"""
scripts/backup_salts.py, offline: SYNTHETIC sealed lines and boxes in
temporary folders, read from files and from temporary git repositories (the
"origin" for --fetch is a folder on this computer). Nothing is fetched from
the network and no real key is used.
"""

import base64
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from support import (  # support also blocks the network
    FAKE_LEDGER_KEY,
    FAKE_LEDGER_KEY_TEXT,
    REPO,
    OfflineTestCase,
    load_fixture,
)

import sealed_ledger as sl
from ledger import ledger_entry

SPEC = importlib.util.spec_from_file_location("backup_salts", REPO / "scripts" / "backup_salts.py")
backup_salts = importlib.util.module_from_spec(SPEC)
sys.modules["backup_salts"] = backup_salts
SPEC.loader.exec_module(backup_salts)

OTHER_KEY = b"SYNTHETIC-other-key-not-the-one!"
NOW = datetime(2026, 11, 7, 13, 50, 0, tzinfo=timezone.utc)
GIT_ID = ["-c", "user.name=SYNTHETIC", "-c", "user.email=synthetic@example.invalid"]


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *GIT_ID, *args], capture_output=True, text=True, check=True)


def sealed_forecast(question_id, key=FAKE_LEDGER_KEY):
    entry = ledger_entry(
        question_id=question_id,
        final_forecast={"probability_yes": 0.25 + question_id / 1000},
        context={"post_id": question_id // 10, "type": "binary", "title": "SYNTHETIC: will it happen?"},
        run_mode="tournament",
    )
    return sl.seal_entry(entry, key)


class BackupTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.work = Path(self._tmp.name)
        self.repo = self.work / "repo"
        (self.repo / "forecasts").mkdir(parents=True)
        self.boxes = self.repo / "forecasts" / "sealed_boxes.jsonl"
        self.ledger = self.repo / "forecasts" / "ledger.jsonl"
        self.out = self.work / "backup"
        self.printed = []

    def write(self, forecasts, *, boxes_too=True):
        for sealed in forecasts:
            if boxes_too:
                sl.append_json_line(self.boxes, sealed.box_line)
            sl.append_json_line(self.ledger, sealed.line)

    def back_up(self, *argv, environ=None):
        self.printed.clear()
        return backup_salts.main(
            ["--out", str(self.out), *argv],
            environ={sl.KEY_ENV: FAKE_LEDGER_KEY_TEXT} if environ is None else environ,
            out=self.printed.append,
            now=NOW,
        )

    def read(self, name):
        return json.loads((self.out / name).read_text(encoding="utf-8"))

    def assert_printed_only_counts_and_names(self, forecasts):
        text = "\n".join(self.printed)
        self.assertNotIn(FAKE_LEDGER_KEY_TEXT, text)
        self.assertNotIn(FAKE_LEDGER_KEY_TEXT.rstrip("="), text)
        for sealed in forecasts:
            self.assertNotIn(sealed.record["salt"], text)
            self.assertNotIn(sealed.line["seal_sha256"], text)
            self.assertNotIn(sealed.line["seal_sha256"][:12], text)
        self.assertNotIn("probability", text)

    def test_every_box_is_opened_checked_and_backed_up_with_a_manifest(self):
        forecasts = [sealed_forecast(n) for n in (70001, 70002, 70003)]
        self.write(forecasts)
        self.assertEqual(self.back_up("--boxes", str(self.boxes), "--ledger", str(self.ledger)), 0)
        salts, records, manifest = self.read("salts.json"), self.read("records.json"), self.read("manifest.json")
        self.assertEqual(salts, {s.line["seal_sha256"]: s.record["salt"] for s in forecasts})
        for sealed in forecasts:
            record = records[sealed.line["seal_sha256"]]
            self.assertEqual(record, sealed.record)
            canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            self.assertEqual(hashlib.sha256(canonical.encode("utf-8")).hexdigest(), sealed.line["seal_sha256"])
        # The manifest's SHA-256s are those of the files written and read.
        for name in ("salts.json", "records.json"):
            self.assertEqual(manifest["files"][name], hashlib.sha256((self.out / name).read_bytes()).hexdigest())
        self.assertEqual(manifest["boxes_sha256"], hashlib.sha256(self.boxes.read_bytes()).hexdigest())
        self.assertEqual(manifest["ledger_sha256"], hashlib.sha256(self.ledger.read_bytes()).hexdigest())
        self.assertEqual(
            {k: manifest[k] for k in ("box_lines", "boxes_opened", "boxes_failed", "sealed_lines",
                                      "sealed_lines_without_box", "unreadable_box_lines", "made_at_utc")},
            {"box_lines": 3, "boxes_opened": 3, "boxes_failed": [], "sealed_lines": 3,
             "sealed_lines_without_box": [], "unreadable_box_lines": 0, "made_at_utc": "2026-11-07T13:50:00Z"},
        )
        self.assertIn("Opened 3 box(es)", "\n".join(self.printed))
        self.assertIn(f"Wrote salts.json, records.json and manifest.json to {self.out}.", self.printed)
        self.assert_printed_only_counts_and_names(forecasts)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["manifest.json", "records.json", "salts.json"])

    def test_the_key_can_come_from_an_env_file(self):
        forecasts = [sealed_forecast(70001)]
        self.write(forecasts)
        env_file = self.work / "SYNTHETIC.env"
        env_file.write_text(f"OTHER=1\n{sl.KEY_ENV}={FAKE_LEDGER_KEY_TEXT}\n", encoding="utf-8")
        self.assertEqual(self.back_up("--boxes", str(self.boxes), "--env-file", str(env_file), environ={}), 0)
        self.assertEqual(len(self.read("salts.json")), 1)
        self.assert_printed_only_counts_and_names(forecasts)

    def test_a_lost_box_is_reported_not_silent(self):
        kept, lost = sealed_forecast(70001), sealed_forecast(70002)
        self.write([kept])
        self.write([lost], boxes_too=False)
        self.assertEqual(self.back_up("--boxes", str(self.boxes), "--ledger", str(self.ledger)), 1)
        manifest = self.read("manifest.json")
        self.assertEqual(manifest["sealed_lines_without_box"], [lost.line["seal_sha256"]])
        self.assertEqual(manifest["boxes_opened"], 1)
        self.assertIn("1 sealed line(s) in the ledger have no box", "\n".join(self.printed))
        self.assertEqual(list(self.read("salts.json")), [kept.line["seal_sha256"]])
        self.assert_printed_only_counts_and_names([kept, lost])

    def test_a_box_that_does_not_open_or_a_broken_line_is_reported(self):
        good, other_key, tampered = sealed_forecast(70001), sealed_forecast(70002, OTHER_KEY), sealed_forecast(70003)
        data = bytearray(base64.b64decode(tampered.box_line["box"]))
        data[-1] ^= 0x01
        tampered.box_line["box"] = base64.b64encode(bytes(data)).decode("ascii")
        self.write([good, other_key, tampered])
        with open(self.boxes, "a", encoding="utf-8", newline="\n") as file:
            file.write("SYNTHETIC broken line\n")
        self.assertEqual(self.back_up("--boxes", str(self.boxes), "--ledger", str(self.ledger)), 1)
        manifest = self.read("manifest.json")
        self.assertEqual(
            sorted(manifest["boxes_failed"]),
            sorted([other_key.line["seal_sha256"], tampered.line["seal_sha256"]]),
        )
        self.assertEqual(manifest["unreadable_box_lines"], 1)
        self.assertEqual(list(self.read("salts.json")), [good.line["seal_sha256"]])
        text = "\n".join(self.printed)
        self.assertIn("2 box(es) did not open", text)
        self.assertIn("1 line(s) of the boxes file could not be read", text)

    def test_no_key_or_a_bad_one_stops_before_anything_is_written(self):
        self.write([sealed_forecast(70001)])
        short = base64.urlsafe_b64encode(bytes(16)).decode("ascii")
        for environ in ({}, {sl.KEY_ENV: short}, {sl.KEY_ENV: "SYNTHETIC not a key"}):
            self.assertEqual(self.back_up("--boxes", str(self.boxes), environ=environ), 1)
            self.assertTrue(self.printed[0].startswith(f"Stopped: {sl.KEY_ENV}"))
            self.assertNotIn(short, "\n".join(self.printed))
            self.assertFalse(self.out.exists())
        self.assertEqual(self.back_up("--boxes", str(self.work / "missing.jsonl")), 1)
        self.assertFalse(self.out.exists())

    def test_a_folder_inside_the_repository_or_an_earlier_backup_is_refused(self):
        self.write([sealed_forecast(70001)])
        # Inside the real repository (in its ignored dry-run folder, in case).
        inside = REPO / "forecasts" / "dry_run" / "SYNTHETIC-backup-guard-test"
        self.addCleanup(shutil.rmtree, inside, True)
        self.printed.clear()
        result = backup_salts.main(
            ["--out", str(inside), "--boxes", str(self.boxes)],
            environ={sl.KEY_ENV: FAKE_LEDGER_KEY_TEXT}, out=self.printed.append, now=NOW,
        )
        self.assertEqual(result, 1)
        self.assertIn("inside the repository", self.printed[0])
        self.assertFalse(inside.exists())
        # Inside the clone it reads from, too.
        self.assertEqual(
            backup_salts.main(["--out", str(self.repo / "backup"), "--repo", str(self.repo)],
                              environ={sl.KEY_ENV: FAKE_LEDGER_KEY_TEXT}, out=self.printed.append, now=NOW),
            1,
        )
        self.assertFalse((self.repo / "backup").exists())
        # A second backup into the same folder needs --replace.
        self.assertEqual(self.back_up("--boxes", str(self.boxes)), 0)
        self.assertEqual(self.back_up("--boxes", str(self.boxes)), 1)
        self.assertIn("already holds a backup", self.printed[0])
        self.assertEqual(self.back_up("--boxes", str(self.boxes), "--replace"), 0)

    def test_the_files_are_read_as_they_are_in_a_commit(self):
        git(self.repo, "init", "-q", "-b", "main")
        (self.repo / "README.md").write_text("SYNTHETIC\n", encoding="utf-8")
        git(self.repo, "add", "README.md")
        git(self.repo, "commit", "-q", "-m", "SYNTHETIC first commit, nothing sealed yet")
        forecasts = [sealed_forecast(n) for n in (70001, 70002)]
        self.write(forecasts)
        git(self.repo, "add", "forecasts")
        git(self.repo, "commit", "-q", "-m", "SYNTHETIC sealed forecasts")
        committed = self.boxes.read_bytes()
        # Not yet committed: the backup of HEAD must not see it.
        self.write([sealed_forecast(70003)])
        self.assertEqual(self.back_up("--git-ref", "HEAD", "--repo", str(self.repo)), 0)
        manifest = self.read("manifest.json")
        self.assertEqual(manifest["boxes_source"], "HEAD:forecasts/sealed_boxes.jsonl")
        self.assertEqual(manifest["boxes_sha256"], hashlib.sha256(committed).hexdigest())
        self.assertEqual((manifest["boxes_opened"], manifest["sealed_lines"]), (2, 2))
        self.assertEqual(set(self.read("salts.json")), {s.line["seal_sha256"] for s in forecasts})
        # A commit from before anything was sealed has no boxes: nothing to back up.
        self.assertEqual(self.back_up("--git-ref", "HEAD~1", "--repo", str(self.repo), "--replace"), 0)
        self.assertEqual((self.read("manifest.json")["box_lines"], self.read("salts.json")), (0, {}))
        # A ref that is not a commit stops.
        self.assertEqual(self.back_up("--git-ref", "SYNTHETIC-no-such-ref", "--repo", str(self.repo), "--replace"), 1)
        self.assertIn("is not a commit", self.printed[0])

    def test_fetch_brings_origin_main_up_to_date_first(self):
        origin = self.repo
        git(origin, "init", "-q", "-b", "main")
        forecasts = [sealed_forecast(70001)]
        self.write(forecasts)
        git(origin, "add", "forecasts")
        git(origin, "commit", "-q", "-m", "SYNTHETIC sealed forecast")
        clone = self.work / "clone"
        clone.mkdir()
        git(clone, "init", "-q", "-b", "main")
        git(clone, "remote", "add", "origin", str(origin))
        # Without a fetch the clone has no origin/main yet.
        self.assertEqual(self.back_up("--git-ref", "origin/main", "--repo", str(clone)), 1)
        self.assertFalse(self.out.exists())
        self.assertEqual(self.back_up("--git-ref", "origin/main", "--fetch", "--repo", str(clone)), 0)
        self.assertEqual(self.read("manifest.json")["boxes_source"], "origin/main:forecasts/sealed_boxes.jsonl")
        self.assertEqual(list(self.read("salts.json")), [forecasts[0].line["seal_sha256"]])


class EndToEndTests(OfflineTestCase):
    """Seal, box, backup and reveal, through main.run against the fake Metaculus."""

    def test_seal_box_backup_and_reveal_with_a_lost_box_reported(self):
        import contextlib
        import io

        import main

        self.set_env(OPENROUTER_API_KEY="sk-or-synthetic-backup-key", GLASS_EARTH_LEDGER_KEY=FAKE_LEDGER_KEY_TEXT)
        page = load_fixture("synthetic_group_posts.json")
        for post in page["results"]:
            post["projects"]["tournament"][0]["slug"] = "bot-testing-area"
        self.metaculus.posts_page = page

        def live_run():
            with contextlib.redirect_stdout(io.StringIO()) as console:
                main.run(["--mode", "test_questions"])
            return console.getvalue()

        # 1. A live run seals four forecasts: four boxes, four sealed lines.
        live_run()
        ledger, boxes = Path("forecasts/ledger.jsonl"), Path("forecasts/sealed_boxes.jsonl")
        sealed = {line["question_id"]: line for line in self.read_jsonl(ledger)}
        self.assertEqual(sorted(sealed), [90011, 90012, 90021, 90022])
        # 2. One box goes missing (SYNTHETIC damage to the boxes file).
        lost = sealed[90012]["seal_sha256"]
        kept = [l for l in boxes.read_text(encoding="utf-8").splitlines() if lost not in l]
        boxes.write_text("".join(l + "\n" for l in kept), encoding="utf-8", newline="\n")
        # 3. The backup copies the three salts and records it can, and reports the lost box.
        printed = []
        out = Path("backup-SYNTHETIC")
        code = backup_salts.main(
            ["--out", str(out), "--boxes", str(boxes), "--ledger", str(ledger)], out=printed.append, now=NOW
        )
        self.assertEqual(code, 1)
        self.assertIn("1 sealed line(s) in the ledger have no box", "\n".join(printed))
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["sealed_lines_without_box"], [lost])
        backed_up = json.loads((out / "records.json").read_text(encoding="utf-8"))
        salts = json.loads((out / "salts.json").read_text(encoding="utf-8"))
        self.assertEqual(len(backed_up), 3)
        # 4. Every question closes; the next live run reveals what it can, first.
        for post in page["results"]:
            for question in post["group_of_questions"]["questions"]:
                question["status"] = "closed"
        with self.assertLogs("main", level="WARNING"):
            console = live_run()
        self.assertIn("Ledger reveals: 3 forecast(s) revealed", console)
        self.assertIn("1 sealed line(s) have no box", console)
        reveals = [line for line in self.read_jsonl(ledger) if line.get("revealed")]
        self.assertEqual(sorted(r["record"]["question_id"] for r in reveals), [90011, 90021, 90022])
        for reveal in reveals:
            line = sealed[reveal["record"]["question_id"]]
            sl.verify_reveal(line, reveal)
            # The reveal is exactly what the backup kept.
            self.assertEqual(reveal["record"], backed_up[line["seal_sha256"]])
            self.assertEqual(reveal["salt"], salts[line["seal_sha256"]])
        self.assertNotIn(lost, [r["seal_sha256"] for r in reveals])


if __name__ == "__main__":
    unittest.main()
