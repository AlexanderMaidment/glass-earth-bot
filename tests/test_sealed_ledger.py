"""
sealed_ledger.py on its own: the ledger key, sealing, the box, the hash and
the reveal format. Every entry, key and salt here is SYNTHETIC.
"""

import base64
import hashlib
import json
import re
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from support import FAKE_LEDGER_KEY, FAKE_LEDGER_KEY_TEXT, REPO  # support also blocks the network

import sealed_ledger as sl
from ledger import ledger_entry

TITLE = "SYNTHETIC: Which made-up colour will the made-up panel pick?"
GROUP_LABEL = "SYNTHETIC sub-question label"
OPTIONS = ("SYNTHETIC Crimson", "SYNTHETIC Viridian", "SYNTHETIC Cobalt")
OTHER_KEY = b"SYNTHETIC-other-key-not-the-one!"


def multiple_choice_entry(**context_changes):
    final = {"probability_yes_per_category": dict(zip(OPTIONS, (0.5, 0.3, 0.2)))}
    readable = "\n".join(f"- {name}: {share}" for name, share in zip(OPTIONS, ("50.0%", "30.0%", "20.0%")))
    context = {
        "post_id": 7007,
        "url": "https://www.metaculus.com/questions/7007",
        "type": "multiple_choice",
        "group_option": GROUP_LABEL,
        "title": TITLE,
        "tournaments": ["fall-futureeval-2026"],
        "scheduled_close_time_utc": "2026-11-04T10:00:00Z",
        "final_forecast_readable": readable,
        "combined_by": "1 forecast from anthropic/claude-opus-5.5, its option probabilities rescaled to sum to 100%.",
        "model_forecasts": [
            {"model": "anthropic/claude-opus-5.5", "forecast": final, "forecast_readable": readable}
        ],
        "models_without_forecast": [],
        "forecaster_model": "anthropic/claude-opus-5.5",
        "route": "SYNTHETIC route",
        "forecast_reason": "first forecast",
        "cost_usd": "0.1162",
        "cost_usd_cap_basis": "0.1162",
        "cost_by_model_usd": {"anthropic/claude-opus-5.5": "0.0760"},
        "model_calls": 5,
        "prices_retrieved_at_utc": "2026-11-03T04:29:51Z",
        **context_changes,
    }
    return ledger_entry(question_id=70007, final_forecast=final, context=context, run_mode="tournament")


def binary_entry(probability=0.37):
    context = {
        "post_id": 7008,
        "url": "https://www.metaculus.com/questions/7008",
        "type": "binary",
        "title": "SYNTHETIC: Will made-up event number 8 happen?",
        "final_forecast_readable": f"{probability * 100:.1f}%",
        "model_forecasts": [
            {
                "model": "anthropic/claude-opus-5.5",
                "forecast": {"probability_yes": probability},
                "forecast_readable": f"{probability * 100:.1f}%",
            }
        ],
    }
    return ledger_entry(
        question_id=70008,
        final_forecast={"probability_yes": probability},
        context=context,
        run_mode="tournament",
    )


def all_keys(value):
    """Every key anywhere in a JSON value."""
    if isinstance(value, dict):
        for key, inner in value.items():
            yield key
            yield from all_keys(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from all_keys(inner)


class KeyTests(unittest.TestCase):
    def test_url_safe_base64_of_32_bytes_with_or_without_padding(self):
        key = bytes(range(200, 232))
        text = base64.urlsafe_b64encode(key).decode("ascii")
        self.assertTrue(text.endswith("="))
        for given in (text, text.rstrip("="), f"  {text}\n"):
            self.assertEqual(sl.parse_ledger_key(given), key)
        self.assertEqual(sl.parse_ledger_key(FAKE_LEDGER_KEY_TEXT), FAKE_LEDGER_KEY)
        self.assertEqual(sl.ledger_key_from_env({sl.KEY_ENV: FAKE_LEDGER_KEY_TEXT}), FAKE_LEDGER_KEY)

    def test_missing_short_long_or_mangled_keys_are_refused_without_showing_them(self):
        short = base64.urlsafe_b64encode(bytes(16)).decode("ascii")
        long = base64.urlsafe_b64encode(bytes(33)).decode("ascii")
        # Standard base64 of these bytes has "+" and "/", which URL-safe base64 never has.
        standard = base64.b64encode(bytes([0xFB, 0xFF] * 16)).decode("ascii")
        self.assertRegex(standard, r"[+/]")
        cases = {
            None: "is not set",
            "": "is not set",
            "   ": "is not set",
            short: "decodes to 16 bytes",
            long: "decodes to 33 bytes",
            standard: "not URL-safe base64",
            "SYNTHETIC not a key!": "not URL-safe base64",
            # 45 characters: one more than a multiple of 4 is never base64.
            FAKE_LEDGER_KEY_TEXT.rstrip("=") + "AA": "not URL-safe base64",
        }
        for text, said in cases.items():
            with self.assertRaises(sl.LedgerKeyError, msg=repr(text)) as caught:
                sl.parse_ledger_key(text)
            message = str(caught.exception)
            self.assertIn(sl.KEY_ENV, message)
            self.assertIn(said, message)
            if text and text.strip():
                self.assertNotIn(text.strip(), message)
        with self.assertRaises(sl.LedgerKeyError):
            sl.ledger_key_from_env({})
        for key in (FAKE_LEDGER_KEY[:31], FAKE_LEDGER_KEY + b"!", FAKE_LEDGER_KEY_TEXT, None):
            with self.assertRaises(sl.LedgerKeyError):
                sl.check_ledger_key(key)


class CryptographyTests(unittest.TestCase):
    def test_the_locked_cryptography_gives_the_published_aes_gcm_answers(self):
        import cryptography
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        lock = (REPO / "uv.lock").read_bytes().decode("utf-8")
        locked = re.search(r'name = "cryptography"\r?\nversion = "([^"]+)"', lock).group(1)
        self.assertEqual(cryptography.__version__, locked)
        # Test cases 14 and 16 (AES-256) of the GCM specification, McGrew and
        # Viega, "The Galois/Counter Mode of Operation (GCM)", 2005: sourced.
        self.assertEqual(
            AESGCM(bytes(32)).encrypt(bytes(12), bytes(16), None).hex(),
            "cea7403d4d606b6e074ec5d3baf39d18" "d0d1c8a799996bf0265b98b5d48ab919",
        )
        key = bytes.fromhex("feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308")
        plaintext = bytes.fromhex(
            "d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a72"
            "1c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39"
        )
        sealed = AESGCM(key).encrypt(
            bytes.fromhex("cafebabefacedbaddecaf888"),
            plaintext,
            bytes.fromhex("feedfacedeadbeeffeedfacedeadbeefabaddad2"),
        )
        self.assertEqual(
            sealed.hex(),
            "522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa"
            "8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662"
            "76fc6ece0f4e1768cddf8853bb2d551b",
        )


class SealTests(unittest.TestCase):
    def test_the_sealed_line_holds_only_its_six_fields(self):
        entry = multiple_choice_entry()
        sealed = sl.seal_entry(entry, FAKE_LEDGER_KEY)
        self.assertEqual(tuple(sealed.line), sl.SEALED_LINE_FIELDS)
        self.assertEqual(
            sealed.line,
            {
                "ledger_version": entry["ledger_version"],
                "sealed": True,
                "question_id": 70007,
                "url": "https://www.metaculus.com/questions/7007",
                "recorded_at_utc": entry["recorded_at_utc"],
                "seal_sha256": sealed.line["seal_sha256"],
            },
        )
        self.assertRegex(sealed.line["seal_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(tuple(sealed.box_line), sl.BOX_LINE_FIELDS)
        self.assertEqual(sealed.box_line["seal_sha256"], sealed.line["seal_sha256"])

    def test_no_text_or_forecast_from_the_question_in_either_public_line(self):
        for entry in (multiple_choice_entry(), binary_entry()):
            sealed = sl.seal_entry(entry, FAKE_LEDGER_KEY)
            line_text = json.dumps(sealed.line)
            # No forecast, model, cost, type or close time in the ledger line.
            forecast_keys = {"final_forecast", "probability_yes", "probability_yes_per_category", "model_forecasts"}
            self.assertFalse(forecast_keys & set(all_keys(sealed.line)))
            for absent in ("title", "group_option", "final_forecast_readable", "cost_usd", "type",
                           "forecaster_model", "scheduled_close_time_utc", "box", "salt", "post_id"):
                self.assertNotIn(absent, sealed.line)
            for text in (entry["title"], GROUP_LABEL, *OPTIONS, "50.0%", "37.0%", "opus"):
                self.assertNotIn(text, line_text)
            # The box line holds only the hash and the encrypted record: no
            # text, salt or record in usable form.
            box = base64.b64decode(sealed.box_line["box"])
            self.assertNotIn(sl.canonical_json(sealed.record), box)
            self.assertNotIn(sealed.record["salt"].encode("ascii"), box)
            self.assertNotIn(bytes.fromhex(sealed.record["salt"]), box)
            for text in (entry["title"], GROUP_LABEL, *OPTIONS):
                self.assertNotIn(text.encode("utf-8"), box)

    def test_the_record_leaves_out_the_text_copied_from_metaculus(self):
        entry = multiple_choice_entry()
        record = sl.seal_entry(entry, FAKE_LEDGER_KEY).record
        self.assertEqual(list(record), [*sl.RECORD_FIELDS, "salt"])
        for name in sl.WITHHELD_FIELDS:
            self.assertNotIn(name, record)
        self.assertNotIn("forecast_readable", set(all_keys(record)))
        self.assertNotIn(TITLE, json.dumps(record))
        self.assertNotIn(GROUP_LABEL, json.dumps(record))
        # What Metaculus was sent is kept, inside the box until the reveal;
        # for multiple choice it carries the option labels.
        self.assertEqual(record["final_forecast"], entry["final_forecast"])
        self.assertEqual(
            record["model_forecasts"],
            [{"model": "anthropic/claude-opus-5.5", "forecast": entry["final_forecast"]}],
        )
        self.assertEqual(record["cost_usd"], "0.1162")

    def test_seal_then_open_gives_back_the_record_and_its_hash(self):
        for entry in (multiple_choice_entry(), binary_entry()):
            sealed = sl.seal_entry(entry, FAKE_LEDGER_KEY)
            opened = sl.open_box(sealed.line["seal_sha256"], sealed.box_line["box"], FAKE_LEDGER_KEY)
            self.assertEqual(opened, sealed.record)
            canonical = json.dumps(opened, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            self.assertEqual(hashlib.sha256(canonical.encode("utf-8")).hexdigest(), sealed.line["seal_sha256"])
            sl.check_record_matches(sealed.line, opened)

    def test_each_forecast_gets_a_fresh_32_byte_salt_and_nonce(self):
        entry = binary_entry()
        first, second = sl.seal_entry(entry, FAKE_LEDGER_KEY), sl.seal_entry(entry, FAKE_LEDGER_KEY)
        for sealed in (first, second):
            self.assertRegex(sealed.record["salt"], r"^[0-9a-f]{64}$")
        self.assertNotEqual(first.record["salt"], second.record["salt"])
        self.assertNotEqual(first.line["seal_sha256"], second.line["seal_sha256"])
        nonces = {base64.b64decode(s.box_line["box"])[: sl.NONCE_BYTES] for s in (first, second)}
        self.assertEqual(len(nonces), 2)

    def test_a_wrong_key_does_not_open_the_box(self):
        sealed = sl.seal_entry(binary_entry(), FAKE_LEDGER_KEY)
        with self.assertRaises(sl.SealError):
            sl.open_box(sealed.line["seal_sha256"], sealed.box_line["box"], OTHER_KEY)

    def test_a_changed_box_or_seal_does_not_open(self):
        sealed = sl.seal_entry(binary_entry(), FAKE_LEDGER_KEY)
        seal, box = sealed.line["seal_sha256"], base64.b64decode(sealed.box_line["box"])

        def flipped(data, index):
            changed = bytearray(data)
            changed[index] ^= 0x01
            return base64.b64encode(bytes(changed)).decode("ascii")

        other_seal = hashlib.sha256(b"SYNTHETIC other record").hexdigest()
        for changed_seal, changed_box in (
            (seal, flipped(box, 0)),  # the nonce
            (seal, flipped(box, sl.NONCE_BYTES + 3)),  # the ciphertext
            (seal, flipped(box, len(box) - 1)),  # the tag
            (seal, base64.b64encode(box[:-1]).decode("ascii")),  # cut short
            (seal, base64.b64encode(box[:20]).decode("ascii")),
            (other_seal, sealed.box_line["box"]),  # a box moved to another seal
            (seal.upper(), sealed.box_line["box"]),
            (seal, "SYNTHETIC not base64!"),
            (seal, None),
        ):
            with self.assertRaises(sl.SealError):
                sl.open_box(changed_seal, changed_box, FAKE_LEDGER_KEY)

    def test_canonical_json_is_sorted_compact_and_utf8(self):
        self.assertEqual(
            sl.canonical_json({"b": [1, 0.5], "a": "café", "c": None}),
            '{"a":"café","b":[1,0.5],"c":null}'.encode("utf-8"),
        )

    def test_an_address_with_a_title_slug_is_cut_to_the_question_number(self):
        entry = multiple_choice_entry(url="https://www.metaculus.com/questions/7007/which-made-up-colour/")
        sealed = sl.seal_entry(entry, FAKE_LEDGER_KEY)
        self.assertEqual(sealed.line["url"], "https://www.metaculus.com/questions/7007/")
        self.assertEqual(sealed.record["url"], "https://www.metaculus.com/questions/7007/")


class RevealFormatTests(unittest.TestCase):
    def setUp(self):
        self.sealed = sl.seal_entry(multiple_choice_entry(), FAKE_LEDGER_KEY)
        self.reveal = sl.reveal_line_for(
            self.sealed.line, self.sealed.record, datetime(2026, 11, 5, 1, 2, 3, tzinfo=timezone.utc)
        )

    def test_anyone_can_check_a_reveal_without_the_key(self):
        self.assertEqual(tuple(self.reveal), sl.REVEAL_LINE_FIELDS)
        self.assertEqual(self.reveal["revealed_at_utc"], "2026-11-05T01:02:03Z")
        self.assertEqual(self.reveal["salt"], self.reveal["record"]["salt"])
        # As written to the ledger and read back by anyone.
        line = json.loads(json.dumps(self.sealed.line, ensure_ascii=True))
        reveal = json.loads(json.dumps(self.reveal, ensure_ascii=True))
        sl.verify_reveal(line, reveal)
        # The same check with nothing but the standard library.
        canonical = json.dumps(reveal["record"], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self.assertEqual(hashlib.sha256(canonical.encode("utf-8")).hexdigest(), line["seal_sha256"])

    def test_a_changed_record_salt_or_line_fails_the_check(self):
        def changed(**fields):
            reveal = json.loads(json.dumps(self.reveal))
            for name, value in fields.items():
                if name == "salt_shown":
                    reveal["salt"] = value
                else:
                    reveal["record"][name] = value
            return reveal

        for reveal in (
            changed(final_forecast={"probability_yes_per_category": dict(zip(OPTIONS, (0.2, 0.3, 0.5)))}),
            changed(salt="0" * 64),
            changed(salt_shown="0" * 64),
            changed(recorded_at_utc="2026-11-01T00:00:00Z"),
        ):
            with self.assertRaises(sl.SealError):
                sl.verify_reveal(self.sealed.line, reveal)
        other_line = dict(self.sealed.line, question_id=70009)
        with self.assertRaises(sl.SealError):
            sl.verify_reveal(other_line, self.reveal)
        with self.assertRaises(sl.SealError):
            sl.verify_reveal(self.sealed.line, dict(self.reveal, seal_sha256="0" * 64))


class JsonLinesTests(unittest.TestCase):
    def test_lines_are_appended_as_ascii_with_lf_and_bad_lines_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "forecasts" / "x.jsonl"
            sl.append_json_line(path, {"a": "café"})
            with open(path, "a", encoding="utf-8", newline="\n") as file:
                file.write("\nnot json\n[1, 2]\n")
            sl.append_json_line(path, {"b": 2})
            raw = path.read_bytes()
            self.assertNotIn(b"\r\n", raw)
            self.assertTrue(raw.isascii())
            with self.assertLogs("sealed_ledger", level="WARNING") as logs:
                self.assertEqual(sl.read_json_lines(path), [{"a": "café"}, {"b": 2}])
            self.assertEqual(len(logs.output), 2)
            self.assertEqual(sl.read_json_lines(Path(folder) / "missing.jsonl"), [])


if __name__ == "__main__":
    unittest.main()
