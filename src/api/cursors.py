"""Encoding and decoding helpers for API list-pagination cursors."""

import base64
import binascii
import json
from datetime import datetime
from uuid import UUID


def encode_source_cursor(
    *,
    created_at: datetime,
    source_id: UUID,
) -> str:
    payload = {
        "created_at": created_at.isoformat(),
        "source_id": str(source_id),
    }
    encoded = json.dumps(
        payload,
        separators=(",", ":"),
    ).encode("utf-8")

    return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")


def decode_source_cursor(value: str) -> tuple[datetime, UUID]:
    try:
        padded_value = value + ("=" * (-len(value) % 4))
        decoded = base64.b64decode(
            padded_value,
            altchars=b"-_",
            validate=True,
        )
        payload = json.loads(decoded.decode("utf-8"))

        if not isinstance(payload, dict):
            raise ValueError("Cursor payload must be an object.")

        created_at = datetime.fromisoformat(payload["created_at"])
        source_id = UUID(payload["source_id"])

        if created_at.tzinfo is None:
            raise ValueError("Cursor timestamp must include a timezone.")

        return created_at, source_id
    except (
        KeyError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        binascii.Error,
    ) as error:
        raise ValueError("Invalid source cursor.") from error


def encode_thread_cursor(
    *,
    updated_at: datetime,
    thread_id: UUID,
) -> str:
    payload = {
        "updated_at": updated_at.isoformat(),
        "thread_id": str(thread_id),
    }
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")


def decode_thread_cursor(value: str) -> tuple[datetime, UUID]:
    try:
        padded_value = value + ("=" * (-len(value) % 4))
        decoded = base64.b64decode(
            padded_value,
            altchars=b"-_",
            validate=True,
        )
        payload = json.loads(decoded.decode("utf-8"))

        if not isinstance(payload, dict):
            raise ValueError("Cursor payload must be an object.")

        updated_at = datetime.fromisoformat(payload["updated_at"])
        thread_id = UUID(payload["thread_id"])

        if updated_at.tzinfo is None:
            raise ValueError("Cursor timestamp must include a timezone.")

        return updated_at, thread_id
    except (
        KeyError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        binascii.Error,
    ) as error:
        raise ValueError("Invalid thread cursor.") from error


def encode_message_cursor(
    *,
    created_at: datetime,
    message_id: UUID,
) -> str:
    payload = {
        "created_at": created_at.isoformat(),
        "message_id": str(message_id),
    }
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    return base64.urlsafe_b64encode(encoded).decode("ascii").rstrip("=")


def decode_message_cursor(value: str) -> tuple[datetime, UUID]:
    try:
        padded_value = value + ("=" * (-len(value) % 4))
        decoded = base64.b64decode(
            padded_value,
            altchars=b"-_",
            validate=True,
        )
        payload = json.loads(decoded.decode("utf-8"))

        if not isinstance(payload, dict):
            raise ValueError("Cursor payload must be an object.")

        created_at = datetime.fromisoformat(payload["created_at"])
        message_id = UUID(payload["message_id"])

        if created_at.tzinfo is None:
            raise ValueError("Cursor timestamp must include a timezone.")

        return created_at, message_id
    except (
        KeyError,
        TypeError,
        ValueError,
        UnicodeDecodeError,
        binascii.Error,
    ) as error:
        raise ValueError("Invalid message cursor.") from error
