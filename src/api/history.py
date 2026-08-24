import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException

from src.api.config import Window
from src.storage.modules import get_minio_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/history", tags=["History"])


def _list_detections(session_id: str | None = None) -> list[str]:
    prefix = f"detections/{session_id}/" if session_id else "detections/"
    return list(get_minio_storage().list_objects(prefix=prefix, recursive=False))


def _load_detection(storage, object_name: str) -> dict:
    data_bytes = storage.get_object_bytes(object_name)
    return json.loads(data_bytes.decode("utf-8"))


def _iter_detections(objects: list[str]) -> Iterator[dict]:
    storage = get_minio_storage()
    for obj in objects:
        if not obj.endswith(".json"):
            continue
        try:
            yield _load_detection(storage, obj)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to fetch or parse %s: %s", obj, e)


@router.get("")
def list_history() -> dict[str, list[dict]]:
    try:
        session_counts: dict[str, int] = {}
        for obj in _list_detections():
            parts = obj.split("/")
            if len(parts) >= 3 and parts[-1].endswith(".json"):
                session_id = parts[-2]
                session_counts[session_id] = session_counts.get(session_id, 0) + 1

        result = [
            {"session_id": sid, "total_detections": count}
            for sid, count in session_counts.items()
        ]
        return {"sessions": result}
    except Exception as e:
        logger.error("Failed to list history", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latencies/summary")
def get_global_latencies_summary() -> dict:
    try:
        webrtc_lats = []
        landmark_lats = []
        flow_lats = []
        spotting_lats = []
        inference_lats = []
        total_lats = []
        fps_list = []

        for data in _iter_detections(_list_detections()):
            if "webrtc_latency_avg_ms" in data:
                webrtc_lats.append(data["webrtc_latency_avg_ms"])
            if "landmark_latency_avg_ms" in data:
                landmark_lats.append(data["landmark_latency_avg_ms"])
            if "flow_latency_avg_ms" in data:
                flow_lats.append(data["flow_latency_avg_ms"])
            if "spotting_latency_ms" in data:
                spotting_lats.append(data["spotting_latency_ms"])
            if "model_inference_latency_ms" in data:
                inference_lats.append(data["model_inference_latency_ms"])
            if "latency_ms" in data:
                total_lats.append(data["latency_ms"])
            if "fps" in data:
                fps_list.append(data["fps"])
            elif "n_frames" in data:
                fps_list.append(round((data["n_frames"] + 1) / Window.SECONDS, 2))

        def avg(lst: list) -> float:
            return round(sum(lst) / len(lst), 2) if lst else 0.0

        return {
            "total_detections_analyzed": len(total_lats),
            "global_averages": {
                "average_fps": avg(fps_list),
                "webrtc_latency_avg_ms": avg(webrtc_lats),
                "landmark_latency_avg_ms": avg(landmark_lats),
                "flow_latency_avg_ms": avg(flow_lats),
                "spotting_latency_ms": avg(spotting_lats),
                "model_inference_latency_ms": avg(inference_lats),
                "total_latency_ms": avg(total_lats),
            },
        }
    except Exception as e:
        logger.error("Failed to fetch global latencies summary", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}")
def get_session_detections(session_id: str) -> dict[str, list[str]]:
    try:
        detections = []
        for obj in _list_detections(session_id):
            filename = obj.split("/")[-1]
            if filename.endswith(".json"):
                detection_id = filename.replace("detection_", "").replace(".json", "")
                detections.append(detection_id)

        return {"detections": detections}
    except Exception as e:
        logger.error("Failed to list history", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/{detection_id}/summary")
def get_detection_summary(session_id: str, detection_id: str) -> dict:
    try:
        object_name = f"detections/{session_id}/detection_{detection_id}.json"
        try:
            data_bytes = get_minio_storage().get_object_bytes(object_name)
        except Exception:  # noqa: BLE001
            raise HTTPException(status_code=404, detail="Detection not found")
        return json.loads(data_bytes.decode("utf-8"))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch detection summary {detection_id}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/batch")
def get_session_batch(session_id: str) -> dict[str, list[dict]]:
    try:
        results = list(_iter_detections(_list_detections(session_id)))
        return {"detections": results}
    except Exception as e:
        logger.error("Failed to fetch batch for session %s", session_id, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{session_id}/latencies")
def get_session_latencies(session_id: str) -> dict[str, list[dict]]:
    try:
        results = [
            {
                "detection_id": data.get("detection_id"),
                "webrtc_latency_avg_ms": data.get("webrtc_latency_avg_ms", 0),
                "landmark_latency_avg_ms": data.get("landmark_latency_avg_ms", 0),
                "flow_latency_avg_ms": data.get("flow_latency_avg_ms", 0),
                "spotting_latency_ms": data.get("spotting_latency_ms", 0),
                "model_inference_latency_ms": data.get("model_inference_latency_ms", 0),
                "total_latency_ms": data.get("latency_ms", 0),
            }
            for data in _iter_detections(_list_detections(session_id))
        ]
        return {"latencies": results}
    except Exception as e:
        logger.error(
            "Failed to fetch latencies for session %s", session_id, exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))
