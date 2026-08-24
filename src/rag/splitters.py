"""Turn a transcript's segments into overlapping, timestamped chunks.

Chunking is pure and cheap -- string joins and integer arithmetic, no model,
no network. That matters for ingestion: it means we can chunk a video purely to
find out how many vectors it *should* have, and compare that against how many
the index actually holds, without paying for a single embedding.
"""

from pathlib import Path
import json

from langchain_core.documents import Document

from src.core.config import settings
from src.core.logging import get_logger

log = get_logger(__name__)

def format_ms(ms: int) -> str:
    seconds = ms // 1000
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02}:{seconds:02}"

def _make_chunk(segments, video_id, title, video_url, chunk_index) -> Document:
    start_ms = min(s["start_ms"] for s in segments)
    end_ms = max(s["end_ms"] for s in segments)
    return Document(
        page_content=" ".join(s["text"] for s in segments),
        metadata={
            "video_id": video_id,
            "video_title": title,
            "video_url": video_url,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "start_time": format_ms(start_ms),
            "end_time": format_ms(end_ms),
            "chunk_index": chunk_index,
        },
    )


def chunk_transcript(
    transcript: list,
    video_id: str,
    title: str,
    video_url: str,
    chunk_size: int = settings.chunk_size,
    overlap_segments: int = settings.chunk_overlap_segments,
):
    chunks = []
    current_segments = []
    current_length = 0
    chunk_index = 0

    for segment in transcript:
        text = segment.get("text", "").strip()
        if not text:
            continue

        start_ms = segment["start_ms"]
        duration_ms = segment["duration_ms"]

        segment_data = {
            "text": text,
            "start_ms": start_ms,
            "end_ms": start_ms + duration_ms,
        }

        if current_segments and current_length + len(text) > chunk_size:
            chunks.append(
                _make_chunk(
                    current_segments, video_id, title, video_url, chunk_index
                )
            )
            chunk_index += 1

            current_segments = (
                current_segments[-overlap_segments:] if overlap_segments else []
            )
            current_length = sum(len(i["text"]) for i in current_segments)

        current_segments.append(segment_data)
        current_length += len(text)

    if current_segments:
        chunks.append(
            _make_chunk(
                current_segments, video_id, title, video_url, chunk_index
            )
        )

    return chunks


# ---------------------------------------------------------------
# Loading transcripts off disk
# ---------------------------------------------------------------

def transcript_path(video_id: str) -> Path:
    return settings.transcripts_dir / f"{video_id}.json"


def load_transcript(video_id: str) -> dict | None:
    """Read one cached transcript. None when it has not been fetched yet."""
    path = transcript_path(video_id)
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    if "video_id" not in data or "segments" not in data:
        log.warning(f"{path.name} is not a transcript record, ignoring")
        return None
    return data


def chunk_record(video_data: dict) -> list[Document]:
    """Chunk an already-loaded transcript record."""
    video_id = video_data["video_id"]

    # KNOWN ISSUE, deliberately left alone here: two transcripts in the corpus
    # (lv0DdVLZuHc, JTp0TY_2hXM) carry an explicit null title, and `.get` with
    # a default returns that null rather than the fallback. So those chunks get
    # video_title=None, which _clean_metadata then drops entirely.
    # Switching to `or` fixes it but changes stored metadata, which means
    # re-upserting those videos -- a data migration, not a refactor. Left for a
    # deliberate `ingest_video(..., force=True)` pass so this change stays
    # byte-identical to what is already in the index.
    title = video_data.get("title", "Unknown Video")
    video_url = video_data.get(
        "url", f"https://www.youtube.com/watch?v={video_id}"
    )

    return chunk_transcript(
        transcript=video_data["segments"],
        video_id=video_id,
        title=title,
        video_url=video_url,
    )


def get_chunks_for_video(video_id: str) -> list[Document]:
    """Chunk exactly one video.

    This is the unit ingestion needs. The old path was get_chunks(), which
    globs the whole transcripts folder -- so adding one video to a 102-video
    corpus meant re-chunking and re-embedding all 103. Correct, thanks to the
    deterministic `{video_id}#{chunk_index}` vector ids, but ~3000 needless
    trips through a CPU embedding model.
    """
    record = load_transcript(video_id)
    if record is None:
        return []
    return chunk_record(record)


def get_chunks() -> list[Document]:
    """Chunk every transcript on disk. Batch/backfill use only.

    Kept because it is how the corpus was originally indexed, but new code
    should call get_chunks_for_video().
    """
    all_chunks: list[Document] = []

    for json_file in sorted(settings.transcripts_dir.glob("*.json")):
        if json_file.name == "manifest.json":
            continue  # the fetcher's bookkeeping file, not a transcript
        log.info(f"Processing: {json_file.name}")

        video_data = load_transcript(json_file.stem)
        if video_data is None:
            log.debug(f"Skipping non-transcript file: {json_file.name}")
            continue

        video_chunks = chunk_record(video_data)
        log.info(f"Created {len(video_chunks)} chunks")
        all_chunks.extend(video_chunks)

    log.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    for r in get_chunks():
        print(r)
        print("--" * 100)