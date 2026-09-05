import json
from pathlib import Path

from src.rag.fetch_transcripts import write_json


def test_write_json_creates_missing_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "fresh-container" / "data" / "transcripts" / "video.json"
    payload = {"video_id": "video", "segments": []}

    assert not target.parent.exists()

    write_json(target, payload)

    assert target.parent.is_dir()
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert not target.with_suffix(".json.tmp").exists()