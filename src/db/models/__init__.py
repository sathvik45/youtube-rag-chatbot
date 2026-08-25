"""Import every model so Alembic can discover the complete schema."""

from src.db.models.citation import Citation
from src.db.models.ingestion_job import IngestionJob
from src.db.models.message import Message
from src.db.models.source import Source
from src.db.models.source_video import SourceVideo
from src.db.models.thread import Thread
from src.db.models.user import User
from src.db.models.video import Video

__all__ = [
    "Citation",
    "IngestionJob",
    "Message",
    "Source",
    "SourceVideo",
    "Thread",
    "User",
    "Video",
]
