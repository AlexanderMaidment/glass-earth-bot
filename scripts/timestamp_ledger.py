"""
Stamp the forecast ledger with OpenTimestamps, and upgrade earlier receipts.

The daily workflow .github/workflows/timestamp_ledger.yaml runs this from the
top of the repository. Each run:
1. Upgrades earlier receipts. A new receipt is pending until a Bitcoin block
   confirms it, usually within a day. `ots upgrade` then adds that proof.
2. If the ledger has changed since the last receipt, stamps its current bytes
   and records the receipt in forecasts/proofs/index.jsonl.

A receipt proves that the ledger, exactly as it was at the recorded commit,
existed no later than the Bitcoin block the receipt is anchored in.
forecasts/README.md says how to check one.

Needs the `ots` command from opentimestamps-client. Stamping sends only a
salted hash of the ledger to the public OpenTimestamps calendars, and is free.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

LEDGER = Path("forecasts/ledger.jsonl")
PROOFS = Path("forecasts/proofs")
INDEX = PROOFS / "index.jsonl"


def recover_backups() -> None:
    """
    `ots upgrade` renames the old receipt to .bak before writing the new one.
    Drop the .bak once the new receipt exists, or put it back if the new one
    was never written. A leftover .bak would also block later upgrades.
    """
    for backup in sorted(PROOFS.glob("*.ots.bak")):
        receipt = backup.with_name(backup.name.removesuffix(".bak"))
        if receipt.exists():
            backup.unlink()
        else:
            backup.rename(receipt)
            print(f"Restored {receipt.name} from its backup.")


def upgrade_receipts() -> None:
    # One receipt per call: `ots upgrade` stops at the first pending receipt.
    for receipt in sorted(PROOFS.glob("*.ots")):
        result = subprocess.run(
            ["ots", "upgrade", str(receipt)], capture_output=True, text=True
        )
        recover_backups()
        status = "complete" if result.returncode == 0 else "pending"
        print(f"{receipt.name}: {status}")


def last_stamped_digest() -> str | None:
    if not INDEX.exists():
        return None
    lines = INDEX.read_text(encoding="utf-8").splitlines()
    entries = [json.loads(line) for line in lines if line.strip()]
    return entries[-1]["ledger_sha256"] if entries else None


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True)


def stamp_ledger() -> None:
    if not LEDGER.exists():
        print("There is no ledger yet, so there is nothing to stamp.")
        return
    tracked = git("ls-files", "--error-unmatch", str(LEDGER)).returncode == 0
    if not tracked or git("diff", "--quiet", "HEAD", "--", str(LEDGER)).returncode != 0:
        raise SystemExit(
            "The ledger has changes that are not committed. A receipt must "
            "match a committed version, so commit the ledger first."
        )
    data = LEDGER.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest == last_stamped_digest():
        print("The ledger has not changed since the last receipt.")
        return

    stamped_at = datetime.now(timezone.utc)
    name = f"ledger-{stamped_at:%Y%m%dT%H%M%SZ}-{digest[:12]}.jsonl"
    receipt = PROOFS / f"{name}.ots"
    PROOFS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        snapshot = Path(folder) / name
        snapshot.write_bytes(data)
        subprocess.run(["ots", "stamp", "--timeout", "30", str(snapshot)], check=True)
        shutil.move(f"{snapshot}.ots", receipt)

    entry = {
        "stamped_at_utc": stamped_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ledger_sha256": digest,
        "ledger_bytes": len(data),
        "ledger_lines": data.count(b"\n"),
        "ledger_commit": git("rev-parse", "HEAD").stdout.strip(),
        "receipt": receipt.as_posix(),
    }
    with open(INDEX, "a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(entry) + "\n")
    print(f"Stamped the ledger ({len(data)} bytes, sha256 {digest}) as {receipt}.")


def main() -> int:
    if shutil.which("ots") is None:
        print("The ots command is missing. Install opentimestamps-client.", file=sys.stderr)
        return 1
    recover_backups()
    upgrade_receipts()
    stamp_ledger()
    return 0


if __name__ == "__main__":
    sys.exit(main())
