"""Ingest one video: fetch -> chunk -> index -> verify.

This is the smallest unit of ingestion work, and everything above it is built
from it. Thread creation resolves a URL into video ids; a worker then calls
`ingest_video` once per id and records the terminal status it returns.

Why per-video rather than the batch path
----------------------------------------
The original flow was `download()` for the whole playlist, then `get_chunks()`
over the entire transcripts folder, then one big `upsert_documents()`. That is
correct -- deterministic `{video_id}#{chunk_index}` vector ids make re-upsert an
overwrite -- but adding one video to a 102-video corpus meant pushing ~3000
chunks back through a CPU embedding model. Threads need to ingest exactly the
videos the index does not already have.

Three states, and only two of them are terminal
-----------------------------------------------
    ok             indexed and verified queryable
    no_transcript  terminal: captions do not exist, retrying cannot help
    error          transient: rate limit, timeout, dropped connection. Retry.

That distinction is the whole point of the status field. Retrying a
`no_transcript` burns API credits forever; giving up on an `error` loses a video
to a blip.

Blocking, not async
-------------------
Every expensive step here blocks: Supadata I/O, then local CPU embedding, then
the Pinecone call. There is nothing for an event loop to interleave, so this is
sync on purpose and callers should run it in a threadpool (FastAPI's
BackgroundTasks already does) rather than awaiting it on the request path. Same
reasoning as `retrieve_node` being sync in the graph.

CLI:
    python -m src.rag.ingest "https://www.youtube.com/watch?v=..."
    python -m src.rag.ingest "https://www.youtube.com/playlist?list=PL..." --limit 20
    python -m src.rag.ingest FqR5vESuKe0 --force
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from enum import Enum

import requests
from supadata.errors import SupadataError

from src.core.config import settings
from src.core.logging import get_logger
from src.rag.fetch_transcripts import (
    TERMINAL_ERRORS,
    FetchError,
    extract_video_id,
    fetch_one,
    get_client,
    read_json,
    resolve_source,
    write_json,
)
from src.rag.splitters import chunk_record, load_transcript, transcript_path
from src.rag.vector_store import (
    delete_video,
    upsert_documents,
    video_vector_count,
)

log = get_logger(__name__)


class VideoStatus(str, Enum):
    """Ingestion outcome for a single video.

    Inherits from str so it compares cleanly against a database column and
    serialises to JSON without a custom encoder.
    """

    OK = "ok"
    NO_TRANSCRIPT = "no_transcript"
    ERROR = "error"

    @property
    def is_terminal(self) -> bool:
        """True when retrying cannot change the answer."""
        return self is not VideoStatus.ERROR


@dataclass(frozen=True)
class IngestResult:
    video_id: str
    status: VideoStatus
    chunks: int = 0
    vectors: int = 0
    title: str | None = None
    # True when the transcript JSON was already on disk (no Supadata call).
    cached_transcript: bool = False
    # True when the index already held the right vectors (no embedding run).
    skipped_embedding: bool = False
    error: str | None = None
    message: str | None = None

    @property
    def ok(self) -> bool:
        return self.status is VideoStatus.OK

    def as_manifest_entry(self) -> dict:
        entry = {"status": self.status.value}
        if self.ok:
            entry.update(
                {"chunks": self.chunks, "vectors": self.vectors, "title": self.title}
            )
        else:
            entry.update({"error": self.error, "message": self.message})
        return entry


# ---------------------------------------------------------------
# index verification
# ---------------------------------------------------------------

def _await_indexed(video_id: str, expected: int) -> int:
    """Poll until the index actually holds `expected` vectors for this video.

    Pinecone serverless upserts are eventually consistent: `add_documents`
    returning is not the same as a query being able to find the vectors. If we
    flip a video to `ok` on the strength of the upsert alone, the very first
    user question can retrieve nothing and hit the graph's refuse branch --
    which is indistinguishable, from the outside, from a retrieval-quality
    problem. Verifying here is what makes `ok` mean "queryable".
    """
    count = 0
    for attempt in range(1, settings.index_verify_attempts + 1):
        count = video_vector_count(video_id)
        if count >= expected:
            return count
        log.debug(
            f"{video_id}: {count}/{expected} vectors visible "
            f"(attempt {attempt}/{settings.index_verify_attempts})"
        )
        time.sleep(settings.index_verify_delay)

    raise FetchError(
        f"{video_id}: only {count}/{expected} vectors visible after "
        f"{settings.index_verify_attempts} attempts"
    )


# ---------------------------------------------------------------
# the unit
# ---------------------------------------------------------------

def ingest_video(
    source: str,
    *,
    force: bool = False,
    lang: str | None = None,
    mode: str = "auto",
) -> IngestResult:
    """Make one video queryable. Idempotent, and cheap when already done.

    `source` may be a bare id or any YouTube URL form -- normalised here so a
    caller cannot accidentally index under a URL-shaped key while retrieval
    filters on the bare id.
    """
    try:
        video_id = extract_video_id(source)
    except FetchError as e:
        return IngestResult(
            video_id=source,
            status=VideoStatus.ERROR,
            error="invalid-source",
            message=str(e),
        )

    try:
        return _ingest(video_id, force=force, lang=lang, mode=mode)

    # Expected failures: the API said no, or the network did.
    except (SupadataError, FetchError, requests.RequestException) as e:
        code = getattr(e, "error", "") or ""
        status = (
            VideoStatus.NO_TRANSCRIPT if code in TERMINAL_ERRORS else VideoStatus.ERROR
        )
        log.warning(f"{video_id}: {status.value} ({code or type(e).__name__}): {e}")
        return IngestResult(
            video_id=video_id, status=status, error=code, message=str(e)
        )

    # Anything else -- a Pinecone client error, a bad transcript record. A
    # worker ingesting 40 videos must not die on video 7, so this is broad on
    # purpose, and logged with a traceback so it is not silently swallowed.
    except Exception as e:  # noqa: BLE001
        log.exception(f"{video_id}: unexpected ingestion failure")
        return IngestResult(
            video_id=video_id,
            status=VideoStatus.ERROR,
            error=type(e).__name__,
            message=str(e),
        )


def _ingest(
    video_id: str, *, force: bool, lang: str | None, mode: str
) -> IngestResult:
    record = load_transcript(video_id)
    cached = record is not None

    # ---- fast paths: is this video already indexed? -----------------------
    #
    # The index, not the local folder, is the shared source of truth. A second
    # user pasting a playlist someone else already ingested should get an
    # instant thread, and a fresh container with an empty data/ directory must
    # not re-fetch and re-embed a corpus that is already there.
    if not force:
        existing = video_vector_count(video_id)

        if cached:
            chunks = chunk_record(record)
            if existing and existing == len(chunks):
                log.info(f"{video_id}: already indexed ({existing} vectors), skipping")
                return IngestResult(
                    video_id=video_id,
                    status=VideoStatus.OK,
                    chunks=len(chunks),
                    vectors=existing,
                    title=record.get("title"),
                    cached_transcript=True,
                    skipped_embedding=True,
                )
        elif existing:
            # Indexed by someone else; we have no local copy to count against,
            # so trust the index rather than spending a Supadata call to check.
            log.info(
                f"{video_id}: {existing} vectors already indexed, no local "
                f"transcript -- trusting the index"
            )
            return IngestResult(
                video_id=video_id,
                status=VideoStatus.OK,
                vectors=existing,
                skipped_embedding=True,
            )

    # ---- fetch ------------------------------------------------------------
    if record is None:
        log.info(f"{video_id}: fetching transcript")
        record = fetch_one(get_client(), video_id, lang=lang, mode=mode)
        write_json(transcript_path(video_id), record)

    # ---- chunk (pure, no model, no network) -------------------------------
    chunks = chunk_record(record)
    if not chunks:
        raise FetchError(f"{video_id}: transcript produced no chunks")

    # ---- clear orphans ----------------------------------------------------
    #
    # Deterministic ids make re-upsert an overwrite, but only for indices that
    # still exist. If a re-chunk produces FEWER chunks than last time -- a
    # bigger chunk_size, a re-fetched transcript with different segmentation --
    # the tail vectors from the previous run are not overwritten by anything
    # and stay in the index forever, retrievable and stale.
    existing = video_vector_count(video_id)
    if existing > len(chunks):
        log.info(
            f"{video_id}: {existing} existing vectors vs {len(chunks)} chunks, "
            f"clearing before re-index"
        )
        delete_video(video_id)

    # ---- embed and upsert -------------------------------------------------
    log.info(f"{video_id}: indexing {len(chunks)} chunks")
    upsert_documents(chunks)

    vectors = _await_indexed(video_id, len(chunks))

    log.info(f"{video_id}: ok ({vectors} vectors)")
    return IngestResult(
        video_id=video_id,
        status=VideoStatus.OK,
        chunks=len(chunks),
        vectors=vectors,
        title=record.get("title"),
        cached_transcript=cached,
    )


# ---------------------------------------------------------------
# playlist resolution
# ---------------------------------------------------------------

def resolve_videos(source: str, limit: int | None = None) -> list[str]:
    """Expand a video or playlist URL into an ordered list of video ids.

    Thread creation calls this before any ingestion so it can write the
    thread's scope immediately. The scope is frozen at that moment: if the
    playlist gains a video next week, this thread will not see it, and that is
    deliberate -- a thread whose corpus shifts under its own conversation
    history cannot be debugged or evaluated.
    """
    return resolve_source(None, source, limit=limit or settings.download_limit)


def ingest_many(
    video_ids: list[str], *, force: bool = False
) -> dict[str, IngestResult]:
    """Ingest a list of videos, recording outcomes in the manifest.

    Manifest writes live here rather than in `ingest_video` on purpose. The
    manifest is a single JSON file updated read-modify-write, so two workers
    ingesting concurrently would lose each other's entries. Keeping the write
    in this single-threaded driver means the race cannot happen from the CLI,
    and the API path (which calls `ingest_video` directly) never touches the
    file at all -- the `videos` table supersedes it.
    """
    manifest_path = settings.transcripts_dir / "manifest.json"
    settings.transcripts_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(manifest_path) if manifest_path.exists() else {}

    results: dict[str, IngestResult] = {}
    total = len(video_ids)

    for i, raw in enumerate(video_ids, start=1):
        vid = extract_video_id(raw)

        # Terminal-negative videos are never retried: the credits spent asking
        # again buy nothing. `force` is the deliberate override.
        if not force and manifest.get(vid, {}).get("status") == "no_transcript":
            log.info(f"[{i}/{total}] {vid} skipped (no transcript)")
            results[vid] = IngestResult(
                video_id=vid,
                status=VideoStatus.NO_TRANSCRIPT,
                error="cached",
                message="previously recorded as having no transcript",
            )
            continue

        result = ingest_video(vid, force=force)
        results[vid] = result
        manifest[vid] = result.as_manifest_entry()
        write_json(manifest_path, manifest)

        log.info(f"[{i}/{total}] {vid} -> {result.status.value}")

    return results


# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="video id, video URL, or playlist URL")
    ap.add_argument(
        "--force",
        action="store_true",
        help="re-fetch and re-embed even when already indexed",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help=f"max playlist videos (default {settings.download_limit})",
    )
    args = ap.parse_args()

    video_ids = resolve_videos(args.source, limit=args.limit)
    print(f"{len(video_ids)} video(s) to ingest\n")

    results = ingest_many(video_ids, force=args.force)

    by_status: dict[str, int] = {}
    for r in results.values():
        by_status[r.status.value] = by_status.get(r.status.value, 0) + 1

    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(by_status.items())))
    for r in results.values():
        if not r.ok:
            print(f"  {r.video_id}  {r.status.value}  {r.error}: {r.message}")

    # Non-zero exit when anything is retryable, so a shell loop or CI step can
    # tell "some videos have no captions" (fine) from "the run half-failed".
    return 1 if any(r.status is VideoStatus.ERROR for r in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
