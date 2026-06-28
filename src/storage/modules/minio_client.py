"""MinIO object storage client for persisting inference artifacts.

Provides a singleton-style :class:`MinioStorage` service that manages
bucket lifecycle, CSV/NPZ uploads, and pre-signed URL generation.

Usage::

    from src.storage.modules import get_minio_storage

    storage = get_minio_storage()
    storage.upload_bytes("bucket", "key.csv", csv_bytes, "text/csv")
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass

from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)

# ── Environment-driven defaults ────────────────────────────────────────
# Priority: MINIO_USER / MINIO_PASSWORD (.env convention) →
#           MINIO_ACCESS_KEY / MINIO_SECRET_KEY (generic S3 convention)

MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY: str = (
    os.getenv("MINIO_USER")
    or os.getenv("MINIO_ACCESS_KEY", "minioadmin")
)
MINIO_SECRET_KEY: str = (
    os.getenv("MINIO_PASSWORD")
    or os.getenv("MINIO_SECRET_KEY", "minioadmin")
)
MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"
MINIO_DEFAULT_BUCKET: str = os.getenv("MINIO_DEFAULT_BUCKET", "pulse-live")


@dataclass
class MinioConfig:
    """Connection parameters for a MinIO instance.

    Attributes:
        endpoint: ``host:port`` of the MinIO server.
        access_key: Root or IAM access key.
        secret_key: Corresponding secret key.
        secure: Whether to use TLS.
        default_bucket: Bucket created on first connection.
    """

    endpoint: str = MINIO_ENDPOINT
    access_key: str = MINIO_ACCESS_KEY
    secret_key: str = MINIO_SECRET_KEY
    secure: bool = MINIO_SECURE
    default_bucket: str = MINIO_DEFAULT_BUCKET


class MinioStorage:
    """High-level wrapper around the MinIO Python client.

    Handles bucket initialisation, byte-level uploads, and object
    retrieval while exposing a deliberately small surface area.

    Args:
        config: Connection parameters.  Uses environment defaults
            when omitted.
    """

    def __init__(self, config: MinioConfig | None = None) -> None:
        self._config = config or MinioConfig()
        self._client = Minio(
            endpoint=self._config.endpoint,
            access_key=self._config.access_key,
            secret_key=self._config.secret_key,
            secure=self._config.secure,
        )
        self._ensure_bucket(self._config.default_bucket)
        logger.info(
            "MinioStorage connected to %s (bucket=%s)",
            self._config.endpoint,
            self._config.default_bucket,
        )

    # ── Bucket management ──────────────────────────────────────────────

    def _ensure_bucket(self, bucket: str) -> None:
        """Create *bucket* if it does not already exist."""
        try:
            if not self._client.bucket_exists(bucket):
                self._client.make_bucket(bucket)
                logger.info("Created bucket: %s", bucket)
        except S3Error:
            logger.error("Failed to ensure bucket '%s'", bucket, exc_info=True)
            raise

    # ── Upload helpers ─────────────────────────────────────────────────

    def upload_bytes(
        self,
        object_name: str,
        data: bytes,
        content_type: str = "application/octet-stream",
        bucket: str | None = None,
    ) -> str:
        """Upload raw bytes to MinIO.

        Args:
            object_name: S3 key (may include ``/`` path separators).
            data: Raw byte content.
            content_type: MIME type for the stored object.
            bucket: Target bucket.  Defaults to :attr:`MinioConfig.default_bucket`.

        Returns:
            The ``object_name`` that was written (for chaining convenience).
        """
        bucket = bucket or self._config.default_bucket
        self._ensure_bucket(bucket)

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
            content_type, len(data), bucket, object_name,
        )
        return object_name

    def upload_csv(
        self,
        object_name: str,
        csv_content: str,
        bucket: str | None = None,
    ) -> str:
        """Convenience wrapper for uploading CSV text.

        Args:
            object_name: S3 key ending in ``.csv``.
            csv_content: UTF-8 CSV text.
            bucket: Target bucket.

        Returns:
            The ``object_name``.
        """
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
        """Convenience wrapper for uploading NumPy ``.npz`` archives.

        Args:
            object_name: S3 key ending in ``.npz``.
            npz_bytes: Raw bytes of a serialised ``.npz`` archive.
            bucket: Target bucket.

        Returns:
            The ``object_name``.
        """
        return self.upload_bytes(
            object_name=object_name,
            data=npz_bytes,
            content_type="application/x-npz",
            bucket=bucket,
        )

    # ── Download / URL helpers ─────────────────────────────────────────

    def get_object_bytes(
        self,
        object_name: str,
        bucket: str | None = None,
    ) -> bytes:
        """Download an object's raw bytes.

        Args:
            object_name: S3 key.
            bucket: Source bucket.

        Returns:
            The object content as bytes.
        """
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
        """Generate a pre-signed GET URL.

        Args:
            object_name: S3 key.
            bucket: Source bucket.
            expires_seconds: URL lifetime in seconds.

        Returns:
            A pre-signed HTTPS/HTTP URL.
        """
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
        """List object keys under *prefix*.

        Args:
            prefix: Key prefix filter.
            bucket: Source bucket.
            recursive: Whether to recurse into sub-prefixes.

        Returns:
            List of matching object keys.
        """
        bucket = bucket or self._config.default_bucket
        objects = self._client.list_objects(bucket, prefix=prefix, recursive=recursive)
        return [obj.object_name for obj in objects]

    # ── Dunder ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"MinioStorage(endpoint='{self._config.endpoint}', "
            f"bucket='{self._config.default_bucket}')"
        )


# ── Module-level singleton ─────────────────────────────────────────────

_storage_instance: MinioStorage | None = None


def get_minio_storage(config: MinioConfig | None = None) -> MinioStorage:
    """Return (or create) the module-level :class:`MinioStorage` singleton.

    Thread-safe for the common case where the first call happens during
    the FastAPI ``lifespan`` context (single-threaded startup).

    Args:
        config: Override config on first call.  Ignored on subsequent calls.

    Returns:
        The shared :class:`MinioStorage` instance.
    """
    global _storage_instance  # noqa: PLW0603
    if _storage_instance is None:
        _storage_instance = MinioStorage(config)
    return _storage_instance


def reset_minio_storage() -> None:
    """Reset the singleton (useful for testing)."""
    global _storage_instance  # noqa: PLW0603
    _storage_instance = None
