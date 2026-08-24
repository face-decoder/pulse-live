import asyncio
import json
import logging
import os
import threading
import time
import uuid

import numpy as np

from src.api.config import Window
from src.api.roi_grid import RoiGrid
from src.api.window_buffers import WindowBuffers

logger = logging.getLogger(__name__)


def _avg_max(xs: list[float]) -> tuple[float, float]:
    return (float(np.mean(xs)), float(np.max(xs))) if xs else (0.0, 0.0)


class AnxietyStreamProcessor:
    QUEUE_LIMIT: int = 3

    def __init__(
        self,
        result_queue: asyncio.Queue[dict[str, object]],
        session_id: str = "unknown",
    ) -> None:
        from src.apex.modules import ApexPhaseSpotterROI
        from src.face.modules import FaceLandmark, FaceRoiPoints
        from src.face.modules.face_aligner import FaceAligner
        from src.optical_flow.modules import TVL1

        self._result_queue = result_queue
        self._session_id = session_id
        self._last_saved_time = 0.0
        self._inference_in_progress = False
        self._lock = threading.Lock()
        self._last_crops: list[np.ndarray] | None = None

        self._landmarker = FaceLandmark()
        self._aligner = FaceAligner()
        self._tvl1 = TVL1(fast_mode=True)
        self._spotter = ApexPhaseSpotterROI()

        rois = [
            FaceRoiPoints.LEFT_EYE_POINTS,
            FaceRoiPoints.RIGHT_EYE_POINTS,
            FaceRoiPoints.LIPS_POINTS,
            FaceRoiPoints.LEFT_EYEBROW_POINTS,
            FaceRoiPoints.RIGHT_EYEBROW_POINTS,
        ]
        self.__grid = RoiGrid(len(rois))
        self.__roi_defs = [frozenset(points) for points in rois]
        self.__buffers = WindowBuffers()

        self._processing_queue: asyncio.Queue[tuple[np.ndarray, float, float]] = (
            asyncio.Queue()
        )
        self.__loop_task = asyncio.create_task(self.__loop())

    @property
    def all_magnitudes(self) -> list[float]:
        return self.__buffers.history

    def push_frame(
        self, img: np.ndarray, received_at: float, webrtc_latency: float = 0.0
    ) -> None:
        while self._processing_queue.qsize() > self.QUEUE_LIMIT - 1:
            try:
                self._processing_queue.get_nowait()
                self._processing_queue.task_done()
            except asyncio.QueueEmpty:
                break
        self._processing_queue.put_nowait((img, received_at, webrtc_latency))

    def summarize_signal(self, mags: list[float]) -> dict:
        return self._spotter.summarize_signal(mags)

    async def flush_pending_inference(self) -> None:
        await self._processing_queue.join()

        if not self._inference_in_progress and len(self.__buffers.mags) > 0:
            snapshot = self.__buffers.snapshot(received_at=time.time())
            logger.info(
                "Session %s: flushing %d buffered flow frames as final window",
                self._session_id,
                len(snapshot.mags),
            )
            self.__trigger(snapshot)

        while self._inference_in_progress:
            await asyncio.sleep(0.1)

    def close(self) -> None:
        if hasattr(self, "__loop_task"):
            self.__loop_task.cancel()

    async def __loop(self) -> None:
        while True:
            try:
                img, received_at, webrtc_latency = await self._processing_queue.get()
            except asyncio.CancelledError:
                break
            except Exception:  # noqa: BLE001
                continue

            try:
                await self.__frame(img, received_at, webrtc_latency)
            except Exception:
                logger.error("Error in process_frame background task", exc_info=True)
            finally:
                self._processing_queue.task_done()

    async def __frame(
        self, img: np.ndarray, received_at: float, webrtc_ms: float
    ) -> None:
        loop = asyncio.get_running_loop()

        start = time.time()
        bbox, crops = await loop.run_in_executor(None, self.__crop, img)
        landmark_ms = (time.time() - start) * 1000

        logger.info(
            "Frame processing: WebRTC latency = %.2f ms | Landmark & ROI latency = %.2f ms",
            webrtc_ms,
            landmark_ms,
        )
        await self._result_queue.put(
            {
                "type": "bbox",
                "bbox": bbox,
                "latency_ms": round((time.time() - received_at) * 1000, 2),
            }
        )

        if self._last_crops is not None:
            await self.__flow(self._last_crops, crops)
        self.__buffers.record_frame(bbox, webrtc_ms, landmark_ms, received_at)
        self._last_crops = crops

        if self.__buffers.ready and not self._inference_in_progress:
            snapshot = self.__buffers.snapshot(received_at)
            logger.info(
                "Triggering background model inference "
                "(window buffer full with %d flow frames)",
                len(snapshot.flows),
            )
            self.__trigger(snapshot)

    def __crop(self, image: np.ndarray) -> tuple[dict | None, list[np.ndarray]]:
        with self._lock:
            landmarks = self._landmarker.detect(image)
            bbox = self.__face_bbox(landmarks)

            try:
                aligned = self._aligner.align(image=image, landmarks=landmarks)
                aligned_landmarks = self._landmarker.detect(aligned)
            except Exception:  # noqa: BLE001
                aligned_landmarks = landmarks

            crops = []
            for roi_points in self.__roi_defs:
                try:
                    roi, _ = self._landmarker.crop_roi(
                        image=image,
                        landmark_result=aligned_landmarks,
                        roi_points=roi_points,
                        margin=RoiGrid.MARGIN,
                        target_size=RoiGrid.TILE,
                    )
                except Exception:  # noqa: BLE001
                    roi = self.__grid.blank
                crops.append(roi)

        return bbox, crops

    async def __flow(
        self,
        prev_crops: list[np.ndarray],
        crops: list[np.ndarray],
    ) -> None:
        loop = asyncio.get_running_loop()

        start = time.time()
        mag, canvas = await loop.run_in_executor(
            None, self.__batch_flow, prev_crops, crops
        )
        flow_ms = (time.time() - start) * 1000
        logger.info(
            "Optical flow (TV-L1) calculation completed. Latency: %.2f ms", flow_ms
        )

        self.__buffers.record_flow(mag, canvas, flow_ms)

    def __batch_flow(
        self, p_crops: list[np.ndarray], c_crops: list[np.ndarray]
    ) -> tuple[float, np.ndarray]:
        with self._lock:
            pairs = list(zip(p_crops, c_crops))
            flows = self._tvl1.compute_batch(pairs, download=True)
        return self.__grid.pack(flows)

    def __face_bbox(self, landmarks) -> dict | None:
        try:
            face = (
                landmarks.face_landmarks[0]
                if landmarks and landmarks.face_landmarks
                else None
            )
            if face is None:
                return None
            xs = [lm.x for lm in face]
            ys = [lm.y for lm in face]
            return {
                "x": float(min(xs)),
                "y": float(min(ys)),
                "width": float(max(xs) - min(xs)),
                "height": float(max(ys) - min(ys)),
            }
        except Exception:  # noqa: BLE001
            return None

    def __trigger(self, snapshot) -> None:
        self._inference_in_progress = True
        asyncio.create_task(self.__infer_bg(snapshot))

    async def __infer_bg(self, snapshot) -> None:
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, self.__infer, snapshot)
            if result is not None:
                self.__attach_fps(result, snapshot)
                self.__persist(result)
                await self._result_queue.put(result)
                if result.get("label") == "anxiety_tinggi":
                    await self._result_queue.put(
                        {
                            "type": "alert",
                            "alert_type": "anxiety_tinggi",
                            "message": "Terdeteksi Tingkat Kecemasan Tinggi",
                        }
                    )
        except Exception:
            logger.exception("Background inference failed")
        finally:
            self._inference_in_progress = False

    @staticmethod
    def __attach_fps(result: dict[str, object], snapshot) -> None:
        latency_ms = result.get("latency_ms", 0)
        if isinstance(latency_ms, (int, float)) and latency_ms > 0:
            fps = len(snapshot.timestamps) / (float(latency_ms) / 1000.0)
            result["fps"] = round(fps, 2)
        else:
            result["fps"] = 0.0

    def __persist(self, result: dict[str, object]) -> None:
        now = time.time()
        if now - self._last_saved_time < Window.SECONDS:
            result["is_logged"] = False
            return

        self._last_saved_time = now
        detection_id = uuid.uuid4().hex
        result["detection_id"] = detection_id
        data = json.dumps(result, indent=2).encode("utf-8")

        session_dir = os.path.join("logs", self._session_id)
        os.makedirs(session_dir, exist_ok=True)
        path = os.path.join(session_dir, f"detection_{detection_id}.json")
        with open(path, "wb") as f:
            f.write(data)

        try:
            from src.storage.modules import get_minio_storage

            get_minio_storage().upload_bytes(
                object_name=f"detections/{self._session_id}/detection_{detection_id}.json",
                data=data,
                content_type="application/json",
            )
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to upload log to MinIO: %s", e)

        result["is_logged"] = True

    def __infer(self, snapshot) -> dict[str, object] | None:
        inferencer = self.__inferencer()
        if inferencer is None or not snapshot.mags:
            return None

        summary = self.summarize_signal(snapshot.mags)

        try:
            flow_array = self.__grid.unpack(snapshot.flows)
            prediction = inferencer.predict_flow(flow_array)
        except Exception:
            logger.error("Inference pipeline failed", exc_info=True)
            return None

        return self.__payload(prediction, snapshot, summary)

    @staticmethod
    def __inferencer():
        from src.models.inferencer import (
            get_loaded_inferencer,
            load_inferencer_from_env,
        )

        loaded = get_loaded_inferencer()
        if loaded is not None:
            return loaded
        try:
            return load_inferencer_from_env()
        except Exception:
            logger.warning(
                "Inferencer not loaded and fallback env loading failed — skipping prediction",
                exc_info=True,
            )
            return None

    def __payload(self, prediction, snapshot, summary: dict) -> dict[str, object]:
        avg_webrtc, max_webrtc = _avg_max(snapshot.webrtc_ms)
        avg_landmark, max_landmark = _avg_max(snapshot.landmark_ms)
        avg_flow, max_flow = _avg_max(snapshot.flow_ms)
        spotting_ms = prediction.spotting_latency_ms or 0.0
        model_ms = prediction.model_inference_latency_ms or 0.0
        total_ms = (time.time() - snapshot.received_at) * 1000

        logger.info(
            "Inference completed:\n"
            "  - WebRTC: avg=%.2f ms max=%.2f ms\n"
            "  - Landmark: avg=%.2f ms max=%.2f ms\n"
            "  - TVL1 flow: avg=%.2f ms max=%.2f ms\n"
            "  - Phase spotting: %.2f ms | Model: %.2f ms\n"
            "  - Total: %.2f ms | label=%s confidence=%.4f",
            avg_webrtc,
            max_webrtc,
            avg_landmark,
            max_landmark,
            avg_flow,
            max_flow,
            spotting_ms,
            model_ms,
            total_ms,
            prediction.label,
            prediction.confidence,
        )

        return {
            "type": "prediction",
            "label": prediction.label,
            "confidence": round(prediction.confidence, 4),
            "prob_high": round(prediction.prob_high, 4),
            "prob_low": round(prediction.prob_low, 4),
            "n_windows": prediction.n_windows,
            "n_frames": snapshot.n_frames,
            "warning": prediction.warning,
            "face_bboxes": snapshot.bboxes,
            "magnitudes": snapshot.mags,
            "smoothed_magnitudes": summary["smoothed_magnitudes"],
            "detected_phases": summary["detected_phases"],
            "latency_ms": round(total_ms, 2),
            "webrtc_latency_avg_ms": round(avg_webrtc, 2),
            "webrtc_latency_max_ms": round(max_webrtc, 2),
            "landmark_latency_avg_ms": round(avg_landmark, 2),
            "landmark_latency_max_ms": round(max_landmark, 2),
            "flow_latency_avg_ms": round(avg_flow, 2),
            "flow_latency_max_ms": round(max_flow, 2),
            "spotting_latency_ms": round(spotting_ms, 2),
            "model_inference_latency_ms": round(model_ms, 2),
        }
