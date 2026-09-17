"""Artifact egress to Backblaze B2 (S3-compatible) — locked decision 5.

The job response carries **URLs and metadata, never raw bytes**. This module is
the only place that knows how to talk to object storage, so the handler stays
about generation and this stays about delivery.

Design notes:
- A fresh `boto3` client per call would re-do TLS setup on every job; we cache
  one client keyed by config. `boto3.client` is not documented thread-safe, but
  the worker is single-job-at-a-time by design (one song at a time on the GPU),
  so this is deliberately not guarded.
- Uploads are presigned with `generate_presigned_url`, which is a *local*
  signature computation — no network round trip, so minting three URLs costs
  nothing after the upload.
- The B2 key needs `writeFiles` on the bucket only; it never needs bucket
  listing, so nothing here calls `list_objects`.
"""

from __future__ import annotations

import logging
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import StorageConfig

log = logging.getLogger(__name__)


class StorageError(RuntimeError):
    """An upload or URL-minting operation failed."""


@dataclass(frozen=True)
class UploadedArtifact:
    """One object that landed in the bucket, and the URLs that reach it."""

    key: str
    url: str
    size_bytes: int
    content_type: str


class B2Storage:
    """Thin, testable wrapper over the S3 surface of Backblaze B2."""

    def __init__(self, config: StorageConfig, client: Any | None = None) -> None:
        """`client` is injectable so tests never touch the network or need creds."""
        self._config: StorageConfig = config
        self._client: Any | None = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._build_client()
        return self._client

    def _build_client(self) -> Any:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - image always has boto3
            raise StorageError("boto3 is not installed") from exc

        from botocore.config import Config as BotoConfig

        cfg = self._config
        return boto3.client(
            "s3",
            endpoint_url=cfg.endpoint_url,
            aws_access_key_id=cfg.key_id,
            aws_secret_access_key=cfg.app_key,
            region_name=cfg.region,
            config=BotoConfig(
                signature_version="s3v4",
                # B2 speaks S3 but is not S3: path-style addressing is the
                # documented form, and retries here are cheap compared to
                # losing a finished song to one flaky TLS handshake.
                s3={"addressing_style": "path"},
                retries={"max_attempts": 4, "mode": "standard"},
                connect_timeout=15,
                read_timeout=120,
            ),
        )

    # -- upload ---------------------------------------------------------------

    def upload_file(self, path: Path, key: str) -> UploadedArtifact:
        """Upload one local file and return its metadata plus a presigned URL.

        `upload_file` transparently switches to multipart above
        `multipart_chunk_bytes`, so a 40 MB FLAC and a 2 KB JSON take the same
        code path.
        """
        path = Path(path)
        if not path.is_file():
            raise StorageError(f"Cannot upload missing file: {path}")

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        extra = {"ContentType": content_type}

        started = time.perf_counter()
        try:
            self.client.upload_file(
                str(path),
                self._config.bucket,
                key,
                ExtraArgs=extra,
                Config=_transfer_config(self._config.multipart_chunk_bytes),
            )
        except Exception as exc:  # boto3 raises a wide family; normalise it
            raise StorageError(f"Upload of {key} failed: {exc}") from exc

        log.info(
            "uploaded %s (%d bytes, %s) in %.2fs",
            key,
            path.stat().st_size,
            content_type,
            time.perf_counter() - started,
        )
        return UploadedArtifact(
            key=key,
            url=self.presign(key),
            size_bytes=path.stat().st_size,
            content_type=content_type,
        )

    def upload_directory(self, directory: Path, prefix: str) -> dict[str, UploadedArtifact]:
        """Upload every regular file under `directory`, keyed by relative name.

        Keys are `{prefix}/{relative/path}`, so a future stage can add nested
        artifact trees without changing this signature.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise StorageError(f"Cannot upload missing directory: {directory}")

        uploaded: dict[str, UploadedArtifact] = {}
        for file_path in sorted(p for p in directory.rglob("*") if p.is_file()):
            relative = file_path.relative_to(directory).as_posix()
            uploaded[relative] = self.upload_file(file_path, f"{prefix.rstrip('/')}/{relative}")
        return uploaded

    # -- delivery -------------------------------------------------------------

    def presign(self, key: str) -> str:
        """Mint a presigned GET URL. Local signing — no network call."""
        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._config.bucket, "Key": key},
                ExpiresIn=self._config.url_ttl_seconds,
            )
        except Exception as exc:
            raise StorageError(f"Could not presign {key}: {exc}") from exc


def _transfer_config(chunk_bytes: int) -> object:
    from boto3.s3.transfer import TransferConfig

    return TransferConfig(
        multipart_threshold=chunk_bytes,
        multipart_chunksize=chunk_bytes,
        # Single-job worker: parallel parts only speed up one upload, and B2
        # rate-limits per key. Four is plenty without risking 503s.
        max_concurrency=4,
        use_threads=True,
    )


def _safe_segment(value: str) -> str:
    """Reduce one path segment to a key-safe token.

    Object keys are opaque, so `..` is not a traversal risk in B2 itself — but a
    key containing `..` or a leading dot is a trap for anything downstream that
    maps keys back onto local paths (a warm-cache restorer, a mirror script).
    Neutralising it here is cheaper than auditing every consumer later.
    """
    safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in str(value))
    # Collapse any surviving dot-run. `strip` alone is not enough:
    # "../../etc" sanitises to "..-..-etc", which still contains "..".
    while ".." in safe:
        safe = safe.replace("..", "-")
    return safe.strip(".-")


def job_prefix(*parts: str) -> str:
    """Build the object-key prefix for one job, sanitising each segment alone.

    Segments are sanitised *separately* and then joined, so the `/` hierarchy is
    preserved: joining first would turn the separator into a dash and flatten
    `jobs/<job>/<request>` into one opaque token.
    """
    segments = [s for s in (_safe_segment(p) for p in parts if p) if s]
    return "jobs/" + "/".join(segments) if segments else "jobs/job"
