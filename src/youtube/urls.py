from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from src.db.models.source import SourceType


@dataclass(frozen=True)
class ParsedYouTubeSource:
    source_type: SourceType
    video_id: str | None = None
    playlist_id: str | None = None


def parse_youtube_url(submitted_value: str) -> ParsedYouTubeSource:
    """Identify whether a YouTube URL represents a video or playlist."""
    parsed = urlparse(submitted_value.strip())
    host = (parsed.hostname or "").lower()
    query = parse_qs(parsed.query)

    if host == "youtu.be":
        video_id = parsed.path.strip("/")
        if video_id:
            return ParsedYouTubeSource(
                source_type=SourceType.VIDEO,
                video_id=video_id,
            )

    if host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        playlist_id = query.get("list", [None])[0]

        if playlist_id:
            return ParsedYouTubeSource(
                source_type=SourceType.PLAYLIST,
                playlist_id=playlist_id,
            )

        video_id = query.get("v", [None])[0]
        if parsed.path == "/watch" and video_id:
            return ParsedYouTubeSource(
                source_type=SourceType.VIDEO,
                video_id=video_id,
            )

        if parsed.path.startswith("/shorts/"):
            video_id = parsed.path.removeprefix("/shorts/").strip("/")
            if video_id:
                return ParsedYouTubeSource(
                    source_type=SourceType.VIDEO,
                    video_id=video_id,
                )

    raise ValueError("Please provide a valid YouTube video or playlist URL.")