from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass

from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)

MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY: str = os.getenv("MINIO_USER") or os.getenv(
    "MINIO_ACCESS_KEY", "minioadmin"
)
MINIO_SECRET_KEY: str = os.getenv("MINIO_PASSWORD") or os.getenv(
    "MINIO_SECRET_KEY", "minioadmin"
)
MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"
MINIO_DEFAULT_BUCKET: str = os.getenv("MINIO_DEFAULT_BUCKET", "pulse-live")


@dataclass
class MinioConfig:
    endpoint: str = MINIO_ENDPOINT
    access_key: str = MINIO_ACCESS_KEY
    secret_key: str = MINIO_SECRET_KEY
    secure: bool = MINIO_SECURE
    default_bucket: str = MINIO_DEFAULT_BUCKET


class MinioStorage:
    def __init__(self, config: MinioConfig | None = None) -> None:
        self._config = config or MinioConfig()
        self._client = Minio(
            endpoint=self._config.endpoint,
            access_key=self._config.access_key,
            secret_key=self._config.secret_key,
            secure=self._config.secure,
        )
        self.__ensure_bucket(self._config.default_bucket)
        logger.info(
            "MinioStorage connected to %s (bucket=%s)",
            self._config.endpoint,
            self._config.default_bucket,
        )

    def __ensure_bucket(self, bucket: str) -> None:
        try:
            if not self._client.bucket_exists(bucket):
                self._client.make_bucket(bucket)
                logger.info("Created bucket: %s", bucket)
        except S3Error:
            logger.error("Failed to ensure bucket '%s'", bucket, exc_info=True)
            raise

    def upload_bytes(
        self,
        object_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        bucket: str | None = None,
    ) -> str:
        bucket = bucket or self._config.default_bucket
        self.__ensure_bucket(bucket)

        stream = io.BytesIO(data)
        self._client.put_object(
            bucket_name=bucket,
            object_name=object_name,
            data=stream,
            length=len(data),
            content_type=content_type,
        )
        logger.info(
            "Uploaded %s (%d bytes) → s3://%s/%s",
            content_type,
            len(data),
            bucket,
            object_name,
        )
        return object_name

    def upload_csv(
        self,
        object_name: str,
        csv_content: str,
        bucket: str | None = None,
    ) -> str:
        return self.upload_bytes(
            object_name=object_name,
            data=csv_content.encode("utf-8"),
            content_type="text/csv",
            bucket=bucket,
        )

    def upload_npz(
        self,
        object_name: str,
        npz_bytes: bytes,
        bucket: str | None = None,
    ) -> str:
        return self.upload_bytes(
            object_name=object_name,
            data=npz_bytes,
            content_type="application/x-npz",
            bucket=bucket,
        )

    def get_object_bytes(
        self,
        object_name: str,
        bucket: str | None = None,
    ) -> bytes:
        bucket = bucket or self._config.default_bucket
        response = self._client.get_object(bucket, object_name)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def presigned_url(
        self,
        object_name: str,
        bucket: str | None = None,
        expires_seconds: int = 3600,
    ) -> str:
        from datetime import timedelta

        bucket = bucket or self._config.default_bucket
        return self._client.presigned_get_object(
            bucket,
            object_name,
            expires=timedelta(seconds=expires_seconds),
        )

    def list_objects(
        self,
        prefix: str = "",
        bucket: str | None = None,
        recursive: bool = True,
    ) -> list[str]:
        bucket = bucket or self._config.default_bucket
        objects = self._client.list_objects(bucket, prefix=prefix, recursive=recursive)
        return [obj.object_name for obj in objects]

    def __repr__(self) -> str:
        return (
            f"MinioStorage(endpoint='{self._config.endpoint}', "
            f"bucket='{self._config.default_bucket}')"
        )


_storage_instance: MinioStorage | None = None


def get_minio_storage(config: MinioConfig | None = None) -> MinioStorage:
    global _storage_instance  # noqa: PLW0603
    if _storage_instance is None:
        _storage_instance = MinioStorage(config)
    return _storage_instance


def reset_minio_storage() -> None:
    global _storage_instance  # noqa: PLW0603
    _storage_instance = None
