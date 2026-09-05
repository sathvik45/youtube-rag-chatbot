"""Download YouTube transcripts (video or playlist) via the Supadata SDK.

Writes one JSON file per video into `out_dir` as `{video_id}.json`, plus a
`manifest.json` recording the outcome for every video so re-runs are cheap.

Segment text is normalised before it hits disk: every run of whitespace
(\xa0 non-breaking spaces, \n from caption line-wrapping, doubled spaces)
collapses to a single space. What lands on disk is the canonical string the
anchor table and splitter will operate on, so character positions stay
consistent from here on.

Usage:
    python fetch_transcripts.py "https://www.youtube.com/playlist?list=PL..." data/transcripts
    python fetch_transcripts.py "EDb37y_MhRw" data/transcripts
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests
from supadata import Supadata
from supadata.errors import SupadataError
from supadata.types import BatchJob, Transcript

from dotenv import load_dotenv
load_dotenv()

from src.core.config import settings
from src.core.logging import get_logger

log = get_logger(__name__)

PLAYLIST_ID_PREFIXES = ("PL", "UU", "OL", "RD", "FL", "LL")

# Video IDs are exactly 11 chars, so "PLxxxxxxxxx" is a video, not a playlist.
VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Error codes the API will keep returning no matter how often we retry.
# "transcript-unavailable" is what a 206 becomes inside the SDK.
TERMINAL_ERRORS = frozenset(
    {"transcript-unavailable", "video-not-found", "invalid-request"}
)

# Not whitespace by Python's definition, so str.split() misses them.
ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)


class FetchError(RuntimeError):
    """Problems we detect ourselves.

    SupadataError is a dataclass needing (error, message, details), so it
    cannot be raised with a single string \u2014 doing so throws TypeError, which
    then sails straight past `except SupadataError` and kills the whole run.
    """


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def get_client(api_key: str | None = None) -> Supadata:
    """One shared Supadata client.

    Previously constructed inside download(), so per-video ingestion would have
    built a fresh client -- and a fresh connection pool -- for every video in a
    playlist.
    """
    key = api_key or settings.supadata_api_key or os.environ.get("SUPADATA_API_KEY")
    if not key:
        raise FetchError(
            "SUPADATA_API_KEY is not set - put it in .env or the environment"
        )
    return Supadata(api_key=key)


# --------------------------------------------------------------------------- #
# text normalisation
# --------------------------------------------------------------------------- #
def clean(text: str) -> str:
    """Collapse all whitespace to single spaces and drop zero-width chars.

    str.split() with no argument splits on any Unicode whitespace, which
    covers \xa0 and \n in one pass.
    """
    return " ".join(text.translate(ZERO_WIDTH).split())


# --------------------------------------------------------------------------- #
# JSON helpers — every read and write goes through these
# --------------------------------------------------------------------------- #
def write_json(path: Path, data) -> None:
    """Write through a temp file so a Ctrl-C can't leave half a manifest behind.

    os.replace is atomic on the same filesystem: readers see either the old
    file or the new one, never a truncated one.
    create parent directory
    → write temporary JSON file
    → atomically replace final JSON file
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# source resolution
# --------------------------------------------------------------------------- #
def is_playlist(source: str) -> bool:
    """True when `source` looks like a playlist URL or a bare playlist ID."""
    # Checked first: plenty of legal video IDs start with PL, UU, RD, ...
    if VIDEO_ID_RE.match(source):
        return False
    if "list=" in source:
        return True
    return source.startswith(PLAYLIST_ID_PREFIXES)


def extract_playlist_id(source: str) -> str:
    """Pull the bare list ID out of a playlist URL; pass bare IDs through.

    Storing the URL instead of the ID makes every later "group by playlist"
    a string match against something that varies by how you copied the link.
    """
    values = parse_qs(urlparse(source).query).get("list")
    return values[0] if values else source


def extract_video_id(source: str) -> str:
    """Accept a watch URL, a youtu.be/shorts/embed/live link, or a bare ID."""
    if "youtube.com" not in source and "youtu.be" not in source:
        return source

    parsed = urlparse(source)
    values = parse_qs(parsed.query).get("v")
    if values:
        return values[0]

    # youtu.be/ID, /shorts/ID, /embed/ID and /live/ID all carry the ID last,
    # and none of them has a ?v= to read.
    tail = parsed.path.rstrip("/").split("/")[-1]
    if VIDEO_ID_RE.match(tail):
        return tail
    log.error(f"could not find a video ID in: {source!r}")
    raise FetchError(f"could not find a video ID in: {source!r}")


def resolve_source(
    client: Supadata | None, source: str, limit: int = 200
) -> list[str]:
    """Turn a video or playlist reference into an ordered list of video IDs.

    Thread creation calls this on its own, before any downloading happens: the
    user needs to see "40 videos" immediately, and thread_videos rows have to
    exist before the ingestion worker has done anything. `client` is optional
    so callers that only have a URL don't have to build one.
    """
    if not is_playlist(source):
        return [extract_video_id(source)]

    client = client or get_client()
    videos = client.youtube.playlist.videos(
        id=extract_playlist_id(source), limit=limit
    )

    ids: list[str] = []
    for bucket in (videos.video_ids, videos.short_ids, videos.live_ids):
        ids.extend(bucket or [])

    seen: set[str] = set()
    return [v for v in ids if not (v in seen or seen.add(v))]


# --------------------------------------------------------------------------- #
# fetching
# --------------------------------------------------------------------------- #
def _segment(chunk) -> dict | None:
    """Normalise one transcript chunk. Returns None for empty text.

    Transcript.__post_init__ inside the SDK turns every dict into a
    TranscriptChunk, so only the attribute form ever reaches us.
    """
    text = clean(chunk.text)
    if not text:
        return None  # empty segments would break strictly-increasing char starts

    return {
        "text": text,
        "start_ms": int(chunk.offset),
        "duration_ms": int(chunk.duration),
    }


def _await_job(
    client: Supadata, job_id: str, timeout: int = 300, poll: int = 5
) -> Transcript:
    """Long videos come back as an async job instead of a transcript.

    The SDK exposes no getter for this. `youtube.batch.get_batch_results`
    looks like the right call but polls /youtube/batch/{id} — a different
    endpoint, for jobs created by transcript.batch(), returning BatchResults
    (which has .results, not .content). So we go through the request helper.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client._request("GET", f"/transcript/{job_id}")
        status = payload.get("status")
        if status == "completed":
            # Rebuilt as a Transcript so callers see one shape either way.
            return Transcript(
                content=payload.get("content") or [],
                lang=payload.get("lang", ""),
                available_langs=payload.get("available_langs") or [],
            )
        if status == "failed":
            log.error(f"job {job_id} failed: {payload.get('error')}")
            raise FetchError(f"job {job_id} failed: {payload.get('error')}")
        time.sleep(poll)
    log.error(f"job {job_id} did not finish in {timeout}s")
    raise FetchError(f"job {job_id} did not finish in {timeout}s")


def fetch_one(
    client: Supadata,
    video_id: str,
    *,
    lang: str | None = None,
    mode: str = "auto",
    playlist_id: str | None = None,
) -> dict:
    """Fetch one transcript plus its title, normalised into our own schema."""
    url = f"https://www.youtube.com/watch?v={video_id}"

    # text=False is load-bearing: text=True returns a flat string and the
    # per-segment timestamps are gone for good.
    result = client.transcript(url=url, lang=lang, text=False, mode=mode)

    if isinstance(result, BatchJob):
        result = _await_job(client, result.job_id)

    title = None
    try:
        title = getattr(client.metadata(url=url), "title", None)
    except SupadataError:
        pass  # a missing title shouldn't sink an otherwise good transcript

    # A string here means we somehow got the text=True shape; iterating it
    # would hand _segment single characters.
    if not isinstance(result.content, list):
        raise FetchError(
            f"{video_id}: expected timestamped chunks, got "
            f"{type(result.content).__name__}"
        )

    segments = [s for s in (_segment(c) for c in result.content) if s]
    if not segments:
        log.error(f"{video_id}: transcript was empty after cleaning")
        raise FetchError(f"{video_id}: transcript was empty after cleaning")

    # Cheap sanity check — bisect silently misbehaves on unsorted input.
    if any(a["start_ms"] > b["start_ms"] for a, b in zip(segments, segments[1:])):
        log.warning(f"{video_id}: segments are not in chronological order")
        raise FetchError(f"{video_id}: segments are not in chronological order")

    return {
        "video_id": video_id,
        "title": title,
        "url": url,
        # Transcript.lang defaults to "", so the attribute always exists and
        # a plain getattr default would never fall back to the requested lang.
        "lang": getattr(result, "lang", None) or lang,
        "playlist_id": playlist_id,
        "mode": mode,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "segments": segments,
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def download(
    source: str,
    out_dir: str | Path | None = None,
    *,
    api_key: str | None = None,
    lang: str | None = None,
    mode: str = "auto",
    limit: int = 200,
) -> dict:
    """Batch download. Superseded by rag.ingest for anything thread-driven.

    Kept because it is how the current corpus was built and it is still the
    right tool for seeding a corpus offline. Note it only fetches -- it does
    not chunk or index. rag.ingest.ingest_video does all three per video.
    """
    # Previously this line read `out_dir = Path(...)` unconditionally, which
    # silently discarded the caller's argument -- including the one the CLI
    # passes as argv[2].
    out_dir = Path(out_dir) if out_dir else settings.transcripts_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    client = get_client(api_key)

    playlist_id = extract_playlist_id(source) if is_playlist(source) else None
    video_ids = resolve_source(client, source, limit=limit)
    log.info(f"{len(video_ids)} video(s) to process")
    # print(f"{len(video_ids)} video(s) to process")

    manifest_path = out_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}

    for i, video_id in enumerate(video_ids, 1):
        target = out_dir / f"{video_id}.json"

        if target.exists():
            # Backfill so a deleted manifest doesn't under-report what's on disk.
            manifest.setdefault(video_id, {"status": "ok", "cached": True})
            log.info(f"[{i}/{len(video_ids)}] {video_id} cached")
            # print(f"[{i}/{len(video_ids)}] {video_id} cached")
            continue

        # Terminal state: don't burn credits retrying what can never work.
        if manifest.get(video_id, {}).get("status") == "no_transcript":
            log.info(f"[{i}/{len(video_ids)}] {video_id} skipped (no transcript)")
            # print(f"[{i}/{len(video_ids)}] {video_id} skipped (no transcript)")
            continue

        try:
            record = fetch_one(
                client, video_id, lang=lang, mode=mode, playlist_id=playlist_id
            )
            write_json(target, record)
            manifest[video_id] = {
                "status": "ok",
                "segments": len(record["segments"]),
                "title": record["title"],
                "fetched_at": record["fetched_at"],
            }
            log.info(f"[{i}/{len(video_ids)}] {video_id} ok "
                  f"({len(record['segments'])} segments)")
            # print(f"[{i}/{len(video_ids)}] {video_id} ok "
                #   f"({len(record['segments'])} segments)")

        # One bad video must never end the run: a TimeoutError or a dropped
        # connection here would otherwise abandon everything still queued.
        except (SupadataError, FetchError, requests.RequestException) as e:
            code = getattr(e, "error", "")
            status = "no_transcript" if code in TERMINAL_ERRORS else "error"
            manifest[video_id] = {
                "status": status,
                "error": code,
                "message": str(e),
            }
            log.info(f"[{i}/{len(video_ids)}] {video_id} FAILED ({status}): {e}")
            # print(f"[{i}/{len(video_ids)}] {video_id} FAILED ({status}): {e}")

        write_json(manifest_path, manifest)

    # Also covers the cached/skipped paths, which `continue` past the write.
    write_json(manifest_path, manifest)
    return manifest


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        raise SystemExit(2)
    download(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
