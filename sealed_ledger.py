"""
Sealed lines for the public forecast ledger.

The workflows commit forecasts/ledger.jsonl to the public repository after
every run. Other bots may use any forecast they can see while its question is
open, and the Metaculus Terms of Use bar republishing site material without
permission. So a live run writes neither the forecast nor any text copied from
Metaculus into the public files. For each forecast Metaculus accepts it writes
two lines:

1. In forecasts/sealed_boxes.jsonl, {"seal_sha256": ..., "box": ...}: the
   forecast's record, encrypted. This line is written first, so that a sealed
   line never exists without its box.
2. In forecasts/ledger.jsonl, a sealed line holding only ledger_version,
   "sealed": true, question_id, url, recorded_at_utc and seal_sha256.

The record is the forecast's ledger entry (ledger.ledger_entry) without the
text copied from Metaculus (WITHHELD_FIELDS, and the readable form of each
model forecast), plus a fresh random salt of 32 bytes in hex. seal_sha256 is
the SHA-256 of the record as canonical JSON (canonical_json). The salt stops
anyone finding the forecast by hashing guesses, so it stays inside the box
until the reveal. The box is base64 of a random 12-byte nonce followed by the
AES-GCM ciphertext (its tag included) of the canonical record, under the
32-byte key in GLASS_EARTH_LEDGER_KEY, with seal_sha256 (its hex text, as
ASCII bytes) as associated data.

Once Metaculus reports the question closed or resolved, a live run opens the
box and appends to the ledger {"revealed": true, "seal_sha256": ...,
"revealed_at_utc": ..., "salt": ..., "record": {...}}. Anyone can then hash
the record, salt included, and compare it with the earlier sealed line
(verify_reveal; forecasts/README.md shows how).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import re
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger(__name__)

KEY_ENV = "GLASS_EARTH_LEDGER_KEY"
KEY_BYTES = 32
SALT_BYTES = 32
NONCE_BYTES = 12
TAG_BYTES = 16
LEDGER_FILE = "forecasts/ledger.jsonl"
BOXES_FILE = "forecasts/sealed_boxes.jsonl"

# The entry's fields that a sealed line shows (an allow-list), and the
# sealed line itself, in order. Nothing else from the entry is ever in it.
PUBLIC_FIELDS = ("ledger_version", "question_id", "url", "recorded_at_utc")
SEALED_LINE_FIELDS = (
    "ledger_version",
    "sealed",
    "question_id",
    "url",
    "recorded_at_utc",
    "seal_sha256",
)
BOX_LINE_FIELDS = ("seal_sha256", "box")
REVEAL_LINE_FIELDS = ("revealed", "seal_sha256", "revealed_at_utc", "salt", "record")
# Text copied from Metaculus: never in a public file, sealed or revealed.
WITHHELD_FIELDS = frozenset({"title", "group_option", "final_forecast_readable"})
# The entry's fields that go into the record, also an allow-list: every field
# ledger.ledger_entry writes except WITHHELD_FIELDS. A test keeps the two in
# step, so a new field must be placed in one or the other.
RECORD_FIELDS = (
    "ledger_version",
    "recorded_at_utc",
    "question_id",
    "post_id",
    "url",
    "type",
    "tournaments",
    "scheduled_close_time_utc",
    "run_mode",
    "final_forecast",
    "combined_by",
    "model_forecasts",
    "models_without_forecast",
    "forecaster_model",
    "route",
    "forecast_reason",
    "cost_usd",
    "cost_usd_cap_basis",
    "cost_by_model_usd",
    "model_calls",
    "prices_retrieved_at_utc",
    "code_commit",
    "run_url",
)
# Of each model forecast, the record keeps the model and the forecast as sent
# to Metaculus, not its readable form (which, like final_forecast_readable,
# can spell out multiple choice options).
MODEL_FORECAST_FIELDS = ("model", "forecast")

_KEY_TEXT = re.compile(r"[A-Za-z0-9_-]+={0,2}")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_SALT_HEX = re.compile(r"[0-9a-f]{%d}" % (2 * SALT_BYTES))
_QUESTION_URL = re.compile(r"https://www\.metaculus\.com/questions/\d+/?")


class LedgerKeyError(ValueError):
    """The ledger key is missing or unusable. The message never holds the key."""


class SealError(ValueError):
    """A box that does not open, or a record or reveal that does not match its seal."""


def parse_ledger_key(text: str | None) -> bytes:
    """
    The ledger key from its text: URL-safe base64 of 32 bytes, with or
    without its "=" padding.
    """
    if text is None or not text.strip():
        raise LedgerKeyError(
            f"{KEY_ENV} is not set. A live run needs it to seal each forecast in the "
            "public ledger until its question closes (see forecasts/README.md)."
        )
    text = text.strip()
    if not _KEY_TEXT.fullmatch(text):
        raise LedgerKeyError(
            f"{KEY_ENV} is not URL-safe base64 (letters, digits, - and _, then up to two =)."
        )
    data = text.rstrip("=")
    try:
        key = base64.b64decode(data + "=" * (-len(data) % 4), altchars=b"-_", validate=True)
    except (binascii.Error, ValueError):
        raise LedgerKeyError(f"{KEY_ENV} is not URL-safe base64.") from None
    if len(key) != KEY_BYTES:
        raise LedgerKeyError(
            f"{KEY_ENV} decodes to {len(key)} bytes; it must be {KEY_BYTES} random bytes."
        )
    return key


def ledger_key_from_env(environ: Mapping[str, str] | None = None) -> bytes:
    environ = os.environ if environ is None else environ
    return parse_ledger_key(environ.get(KEY_ENV))


def check_ledger_key(key: Any) -> bytes:
    if not isinstance(key, bytes) or len(key) != KEY_BYTES:
        raise LedgerKeyError(f"The ledger key must be {KEY_BYTES} bytes.")
    return key


def canonical_json(value: Any) -> bytes:
    """
    The canonical form that seal_sha256 hashes: keys sorted, no spaces
    (separators "," and ":"), characters as themselves, encoded as UTF-8.
    In Python: json.dumps(value, sort_keys=True, separators=(",", ":"),
    ensure_ascii=False).encode("utf-8").
    """
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc_text(moment: datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def public_url(url: Any, post_id: Any) -> str | None:
    """
    The question's page address with nothing after its number, since an
    address ending in a title slug would copy question text.
    """
    if isinstance(url, str) and _QUESTION_URL.fullmatch(url):
        return url
    if isinstance(post_id, int) and not isinstance(post_id, bool):
        return f"https://www.metaculus.com/questions/{post_id}/"
    return None


def record_for(entry: Mapping[str, Any], salt: str) -> dict[str, Any]:
    """The record a sealed line commits to: the entry's RECORD_FIELDS, then the salt."""
    if not isinstance(salt, str) or not _SALT_HEX.fullmatch(salt):
        raise ValueError(f"The salt must be {SALT_BYTES} bytes in lowercase hex")
    record = {name: entry.get(name) for name in RECORD_FIELDS}
    record["url"] = public_url(entry.get("url"), entry.get("post_id"))
    forecasts = entry.get("model_forecasts")
    record["model_forecasts"] = (
        [
            {name: forecast.get(name) for name in MODEL_FORECAST_FIELDS}
            for forecast in forecasts
            if isinstance(forecast, Mapping)
        ]
        if isinstance(forecasts, list)
        else None
    )
    record["salt"] = salt
    return record


@dataclass(frozen=True)
class Sealed:
    """A sealed forecast: its ledger line, its box line and (kept in memory only) its record."""

    line: dict[str, Any]
    box_line: dict[str, Any]
    record: dict[str, Any]


def seal_entry(entry: Mapping[str, Any], key: bytes) -> Sealed:
    """Seal one ledger entry under the key, with a fresh salt and nonce."""
    key = check_ledger_key(key)
    record = record_for(entry, secrets.token_hex(SALT_BYTES))
    plaintext = canonical_json(record)
    seal_sha256 = sha256_hex(plaintext)
    nonce = secrets.token_bytes(NONCE_BYTES)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, seal_sha256.encode("ascii"))
    values = {
        **{name: record[name] for name in PUBLIC_FIELDS},
        "sealed": True,
        "seal_sha256": seal_sha256,
    }
    return Sealed(
        line={name: values[name] for name in SEALED_LINE_FIELDS},
        box_line={
            "seal_sha256": seal_sha256,
            "box": base64.b64encode(nonce + ciphertext).decode("ascii"),
        },
        record=record,
    )


def open_box(seal_sha256: Any, box: Any, key: bytes) -> dict[str, Any]:
    """
    The record inside a box, after checking that its SHA-256 is seal_sha256
    and that it is in canonical form. Raises SealError otherwise.
    """
    key = check_ledger_key(key)
    if not isinstance(seal_sha256, str) or not _SHA256_HEX.fullmatch(seal_sha256):
        raise SealError("seal_sha256 is not a SHA-256 hash in lowercase hex")
    try:
        data = base64.b64decode(box, validate=True)
    except (binascii.Error, ValueError, TypeError):
        raise SealError("the box is not base64") from None
    if len(data) < NONCE_BYTES + TAG_BYTES:
        raise SealError("the box is too short")
    try:
        plaintext = AESGCM(key).decrypt(
            data[:NONCE_BYTES], data[NONCE_BYTES:], seal_sha256.encode("ascii")
        )
    except InvalidTag:
        raise SealError("the box does not open with this key, or it was changed") from None
    if sha256_hex(plaintext) != seal_sha256:
        raise SealError("the record's SHA-256 does not match seal_sha256")
    try:
        record = json.loads(plaintext.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SealError("the record is not JSON") from None
    if not isinstance(record, dict) or canonical_json(record) != plaintext:
        raise SealError("the record is not in canonical form")
    if not isinstance(record.get("salt"), str) or not _SALT_HEX.fullmatch(record["salt"]):
        raise SealError(f"the record has no salt of {SALT_BYTES} bytes")
    return record


def check_record_matches(line: Mapping[str, Any], record: Mapping[str, Any]) -> None:
    """The fields a sealed line shows must be the record's own."""
    for name in PUBLIC_FIELDS:
        if record.get(name) != line.get(name):
            raise SealError(f"the record's {name} does not match the sealed line")


def is_sealed_line(line: Any) -> bool:
    return (
        isinstance(line, dict)
        and line.get("sealed") is True
        and isinstance(line.get("seal_sha256"), str)
    )


def is_reveal_line(line: Any) -> bool:
    return isinstance(line, dict) and line.get("revealed") is True


def reveal_line_for(
    line: Mapping[str, Any], record: Mapping[str, Any], revealed_at: datetime
) -> dict[str, Any]:
    """The reveal of a sealed line: its record, with the salt also shown on its own."""
    return {
        "revealed": True,
        "seal_sha256": line["seal_sha256"],
        "revealed_at_utc": utc_text(revealed_at),
        "salt": record["salt"],
        "record": dict(record),
    }


def verify_reveal(line: Mapping[str, Any], reveal: Mapping[str, Any]) -> None:
    """
    What anyone can check without the key: the record's SHA-256, salt
    included, is the sealed line's seal_sha256, the salt shown is the
    record's, and the sealed line's fields are the record's. Raises
    SealError if not.
    """
    if not is_sealed_line(line):
        raise SealError("not a sealed line")
    if not is_reveal_line(reveal):
        raise SealError("not a reveal line")
    if reveal.get("seal_sha256") != line["seal_sha256"]:
        raise SealError("the reveal is for another seal")
    record = reveal.get("record")
    if not isinstance(record, dict):
        raise SealError("the reveal has no record")
    if sha256_hex(canonical_json(record)) != line["seal_sha256"]:
        raise SealError("the record's SHA-256 does not match the sealed line")
    if reveal.get("salt") != record.get("salt"):
        raise SealError("the salt shown is not the record's")
    check_record_matches(line, record)


def read_json_lines(path: Path | str) -> list[dict[str, Any]]:
    """The JSON objects in a JSON Lines file, in order. An unreadable line is skipped and logged."""
    path = Path(path)
    if not path.exists():
        return []
    objects: list[dict[str, Any]] = []
    for number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not text.strip():
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Skipping unreadable line {number} of {path}")
            continue
        if isinstance(value, dict):
            objects.append(value)
        else:
            logger.warning(f"Skipping line {number} of {path}: not a JSON object")
    return objects


def append_json_line(path: Path | str, value: Mapping[str, Any]) -> None:
    """Add one line to a JSON Lines file: ASCII only, and LF line endings on every computer."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(value, ensure_ascii=True) + "\n")
