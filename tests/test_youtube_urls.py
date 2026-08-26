import pytest

from src.db.models.source import SourceType
from src.youtube.urls import parse_youtube_url


@pytest.mark.parametrize(
    ("url", "expected_type", "expected_video_id", "expected_playlist_id"),
    [
        (
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            SourceType.VIDEO,
            "dQw4w9WgXcQ",
            None,
        ),
        (
            "https://youtu.be/dQw4w9WgXcQ",
            SourceType.VIDEO,
            "dQw4w9WgXcQ",
            None,
        ),
        (
            "https://www.youtube.com/playlist?list=PL123456789",
            SourceType.PLAYLIST,
            None,
            "PL123456789",
        ),
    ],
)
def test_parse_youtube_url(
    url: str,
    expected_type: SourceType,
    expected_video_id: str | None,
    expected_playlist_id: str | None,
) -> None:
    result = parse_youtube_url(url)

    assert result.source_type == expected_type
    assert result.video_id == expected_video_id
    assert result.playlist_id == expected_playlist_id


def test_parse_youtube_url_rejects_invalid_url() -> None:
    with pytest.raises(ValueError, match="valid YouTube"):
        parse_youtube_url("https://example.com/not-a-youtube-url")