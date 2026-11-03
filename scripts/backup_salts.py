"""
Back up the salts and records of the sealed forecast ledger, on this computer.

Every forecast in the public ledger is sealed until its question closes
(sealed_ledger.py): its record, salt included, is public only inside an
encrypted box in forecasts/sealed_boxes.jsonl. This script opens every box
with the ledger key, checks each record against its seal_sha256, and writes
the salts and records to a folder outside this repository, so that they
survive even if the boxes or the key are lost. It reads only local files and
git; it never opens a network connection except `git fetch origin` with
--fetch.

    uv run --locked python scripts/backup_salts.py --out <folder>
        [--boxes <sealed_boxes.jsonl> [--ledger <ledger.jsonl>]]
        [--git-ref origin/main [--fetch]] [--repo <clone>]
        [--env-file <.env>] [--replace]

- With --boxes it reads that file, and --ledger if given. With --git-ref it
  reads both files as they are in that commit (the bytes `git show
  <ref>:forecasts/sealed_boxes.jsonl` prints), after `git fetch origin` with
  --fetch. Otherwise it reads both from the working copy of --repo (this
  repository by default). The ledger, when there is one, shows which sealed
  lines have no box.
- The key is GLASS_EARTH_LEDGER_KEY from --env-file if given, otherwise from
  the environment. It is never printed.
- --out gets salts.json and records.json (each keyed by seal_sha256) and
  manifest.json: the SHA-256 of each file written and of each file read, the
  counts, and the seals of any box that failed or sealed line with no box.
  A folder that already holds a backup is replaced only with --replace, and
  a folder inside this repository is refused, so that salts are never
  committed to it.
- It prints only counts and file names. It ends with exit code 1 if a box
  does not open or does not match its seal_sha256, a line of the boxes file
  cannot be read, or a sealed line in the ledger has no box.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sealed_ledger import (  # noqa: E402
    BOXES_FILE,
    KEY_ENV,
    LEDGER_FILE,
    LedgerKeyError,
    SealError,
    check_record_matches,
    is_sealed_line,
    open_box,
    parse_ledger_key,
    sha256_hex,
    utc_text,
)

BACKUP_VERSION = 1
SALTS_FILE, RECORDS_FILE, MANIFEST_FILE = "salts.json", "records.json", "manifest.json"

Runner = Callable[..., subprocess.CompletedProcess]


class BackupError(RuntimeError):
    """Stops the backup. The message never holds the key, a salt or a record."""


def _json_lines(data: bytes) -> tuple[list[dict[str, Any]], int]:
    """The JSON objects in JSON Lines bytes, and how many lines could not be read."""
    objects, unreadable = [], 0
    for text in data.decode("utf-8", errors="replace").splitlines():
        if not text.strip():
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            unreadable += 1
            continue
        if isinstance(value, dict):
            objects.append(value)
        else:
            unreadable += 1
    return objects, unreadable


def _git(run: Runner, repo: Path, *args: str) -> subprocess.CompletedProcess:
    return run(["git", "-C", str(repo), *args], capture_output=True)


def _git_file(run: Runner, repo: Path, ref: str, path: str) -> bytes | None:
    """A file's bytes as they are in a commit, or None if the commit has no such file."""
    if _git(run, repo, "cat-file", "-e", f"{ref}:{path}").returncode != 0:
        return None
    shown = _git(run, repo, "cat-file", "blob", f"{ref}:{path}")
    if shown.returncode != 0:
        raise BackupError(f"git could not read {ref}:{path}")
    return shown.stdout


def _read_file(path: Path, *, required: bool) -> bytes | None:
    if path.is_file():
        return path.read_bytes()
    if required:
        raise BackupError(f"{path} was not found")
    return None


def read_sources(args: argparse.Namespace, run: Runner) -> dict[str, Any]:
    """The boxes file and the ledger, with a name for each, from the chosen source."""
    repo = Path(args.repo)
    if args.git_ref:
        if args.fetch and _git(run, repo, "fetch", "--quiet", "origin").returncode != 0:
            raise BackupError(f"git fetch origin failed in {repo}")
        ref = args.git_ref
        if _git(run, repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").returncode != 0:
            raise BackupError(f"{ref} is not a commit in {repo}")
        return {
            "boxes": _git_file(run, repo, ref, BOXES_FILE),
            "boxes_source": f"{ref}:{BOXES_FILE}",
            "ledger": _git_file(run, repo, ref, LEDGER_FILE),
            "ledger_source": f"{ref}:{LEDGER_FILE}",
        }
    if args.boxes:
        return {
            "boxes": _read_file(Path(args.boxes), required=True),
            "boxes_source": str(args.boxes),
            "ledger": _read_file(Path(args.ledger), required=True) if args.ledger else None,
            "ledger_source": str(args.ledger) if args.ledger else None,
        }
    return {
        "boxes": _read_file(repo / BOXES_FILE, required=False),
        "boxes_source": str(repo / BOXES_FILE),
        "ledger": _read_file(repo / LEDGER_FILE, required=False),
        "ledger_source": str(repo / LEDGER_FILE),
    }


def read_key(args: argparse.Namespace, environ: Mapping[str, str]) -> bytes:
    """The ledger key, from --env-file or the environment, kept in memory only."""
    if args.env_file:
        from dotenv import dotenv_values

        path = Path(args.env_file)
        if not path.is_file():
            raise BackupError(f"{path} was not found")
        return parse_ledger_key(dotenv_values(path).get(KEY_ENV))
    return parse_ledger_key(environ.get(KEY_ENV))


def check_out_folder(out: Path, repo: Path, replace: bool) -> None:
    resolved = out.resolve()
    for repository in {ROOT.resolve(), repo.resolve()}:
        if resolved == repository or resolved.is_relative_to(repository):
            raise BackupError(
                f"{out} is inside the repository {repository}; write the backup outside it, "
                "so that its salts are never committed"
            )
    if not replace and any((out / name).exists() for name in (SALTS_FILE, RECORDS_FILE, MANIFEST_FILE)):
        raise BackupError(f"{out} already holds a backup; give a new folder, or add --replace")


def _write(path: Path, data: bytes) -> None:
    """Write a whole file, then put it in place, so a file is never left half written."""
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(data)
    os.replace(partial, path)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def back_up(
    sources: Mapping[str, Any], key: bytes, out: Path, now: datetime
) -> tuple[dict[str, Any], list[str]]:
    """Open every box, write the backup, and return the manifest and the problems found."""
    box_lines, unreadable = _json_lines(sources["boxes"] or b"")
    ledger = sources["ledger"]
    sealed = (
        {line["seal_sha256"]: line for line in _json_lines(ledger)[0] if is_sealed_line(line)}
        if ledger is not None
        else None
    )
    records: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    for line in box_lines:
        seal = line.get("seal_sha256")
        if not isinstance(seal, str):
            unreadable += 1
            continue
        if seal in records:
            continue  # the same line twice, as a union merge can leave it
        try:
            record = open_box(seal, line.get("box"), key)
            if sealed is not None and seal in sealed:
                check_record_matches(sealed[seal], record)
        except SealError:
            failed.append(seal)
            continue
        records[seal] = record
    failed = [seal for seal in dict.fromkeys(failed) if seal not in records]
    box_seals = {line.get("seal_sha256") for line in box_lines}
    without_box = [seal for seal in (sealed or {}) if seal not in box_seals]
    without_line = [seal for seal in records if sealed is not None and seal not in sealed]

    salts_bytes = _json_bytes({seal: record["salt"] for seal, record in records.items()})
    records_bytes = _json_bytes(records)
    manifest = {
        "backup_version": BACKUP_VERSION,
        "made_at_utc": utc_text(now),
        "boxes_source": sources["boxes_source"],
        "boxes_sha256": sha256_hex(sources["boxes"]) if sources["boxes"] is not None else None,
        "ledger_source": sources["ledger_source"] if ledger is not None else None,
        "ledger_sha256": sha256_hex(ledger) if ledger is not None else None,
        "box_lines": len(box_lines),
        "unreadable_box_lines": unreadable,
        "boxes_opened": len(records),
        "boxes_failed": failed,
        "sealed_lines": len(sealed) if sealed is not None else None,
        "sealed_lines_without_box": without_box,
        "boxes_without_sealed_line": without_line,
        "files": {SALTS_FILE: sha256_hex(salts_bytes), RECORDS_FILE: sha256_hex(records_bytes)},
    }
    out.mkdir(parents=True, exist_ok=True)
    _write(out / SALTS_FILE, salts_bytes)
    _write(out / RECORDS_FILE, records_bytes)
    _write(out / MANIFEST_FILE, _json_bytes(manifest))

    problems = []
    if failed:
        problems.append(
            f"{len(failed)} box(es) did not open with this key or did not match their seal_sha256 "
            f"(listed in {MANIFEST_FILE})."
        )
    if unreadable:
        problems.append(f"{unreadable} line(s) of the boxes file could not be read.")
    if without_box:
        problems.append(
            f"{len(without_box)} sealed line(s) in the ledger have no box, so they can never be "
            f"revealed (listed in {MANIFEST_FILE})."
        )
    return manifest, problems


def main(
    argv: Sequence[str] | None = None,
    *,
    run: Runner = subprocess.run,
    environ: Mapping[str, str] | None = None,
    out: Callable[[str], None] = print,
    now: datetime | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Back up the sealed ledger's salts and records (see the top of this file)")
    parser.add_argument("--out", required=True, help="The folder to write the backup to, outside this repository")
    parser.add_argument("--boxes", help="A sealed_boxes.jsonl file to read")
    parser.add_argument("--ledger", help="With --boxes: the ledger.jsonl that goes with it")
    parser.add_argument("--git-ref", help="Read both files as they are in this commit, for example origin/main")
    parser.add_argument("--fetch", action="store_true", help="With --git-ref: run git fetch origin first")
    parser.add_argument("--repo", default=str(ROOT), help="The clone to read from (default: this one)")
    parser.add_argument("--env-file", help=f"A .env file holding {KEY_ENV} (default: the environment)")
    parser.add_argument("--replace", action="store_true", help="Replace a backup already in --out")
    args = parser.parse_args(argv)
    environ = os.environ if environ is None else environ
    try:
        if args.boxes and args.git_ref:
            raise BackupError("Give --boxes or --git-ref, not both")
        if args.ledger and not args.boxes:
            raise BackupError("--ledger goes with --boxes")
        if args.fetch and not args.git_ref:
            raise BackupError("--fetch goes with --git-ref")
        folder = Path(args.out)
        check_out_folder(folder, Path(args.repo), args.replace)
        key = read_key(args, environ)
        sources = read_sources(args, run)
        manifest, problems = back_up(sources, key, folder, now or datetime.now(timezone.utc))
    except (BackupError, LedgerKeyError) as error:
        out(f"Stopped: {error}")
        return 1
    out(f"Read {manifest['box_lines']} box line(s) from {manifest['boxes_source']}.")
    out(f"Opened {manifest['boxes_opened']} box(es), each record matching its seal_sha256.")
    if manifest["sealed_lines"] is not None:
        out(
            f"The ledger ({manifest['ledger_source']}) has {manifest['sealed_lines']} sealed line(s), "
            f"{len(manifest['sealed_lines_without_box'])} of them with no box."
        )
    out(f"Wrote {SALTS_FILE}, {RECORDS_FILE} and {MANIFEST_FILE} to {folder}.")
    for problem in problems:
        out(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
