"""Alpha Result Recorder — auto-persist WQ BRAIN submission outcomes to JSONL.

After each WQ BRAIN simulation/submission, this module appends a structured
record to the appropriate JSONL file. Deduplication by alpha_id is enforced.

JSONL file paths are configured via environment variables (see .env):
- ALPHA_JSONL_SC_BLACKLIST: SC FAIL / LOW_SHARPE FAIL records
- ALPHA_JSONL_ACTIVE: Successfully submitted ACTIVE alphas
- ALPHA_JSONL_TIMEOUTS: Submission timeouts pending finalize

Usage (called internally by wq_brain_service.py):
    from quantgpt.alpha_result_recorder import record_alpha_result
    record_alpha_result(alpha_id="xxx", status="SC_FAIL", ...)
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration from .env
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

JSONL_SC_BLACKLIST = os.getenv(
    "ALPHA_JSONL_SC_BLACKLIST",
    str(_PROJECT_ROOT / "docs" / "knowledge" / "data" / "sc_blacklist.jsonl"),
)
JSONL_ACTIVE = os.getenv(
    "ALPHA_JSONL_ACTIVE",
    str(_PROJECT_ROOT / "docs" / "knowledge" / "data" / "active_alphas.jsonl"),
)
JSONL_TIMEOUTS = os.getenv(
    "ALPHA_JSONL_TIMEOUTS",
    str(_PROJECT_ROOT / "docs" / "knowledge" / "data" / "timeouts.jsonl"),
)

# Status → file mapping
_STATUS_FILE_MAP = {
    "SC_FAIL": JSONL_SC_BLACKLIST,
    "LOW_SHARPE_FAIL": JSONL_SC_BLACKLIST,
    "LOW_FITNESS_FAIL": JSONL_SC_BLACKLIST,
    "OTHER_FAIL": JSONL_TIMEOUTS,
    "SC_PENDING": JSONL_TIMEOUTS,
    "ACTIVE": JSONL_ACTIVE,
}


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------


def record_alpha_result(
    alpha_id: str,
    expression: str,
    status: str,
    *,
    region: str = "USA",
    universe: str = "TOP3000",
    neutralization: str = "SUBINDUSTRY",
    decay: int = 0,
    delay: int = 1,
    fitness: float | None = None,
    sharpe: float | None = None,
    returns: float | None = None,
    turnover: float | None = None,
    sc: float | None = None,
    rev_src: str | None = None,
    fund: list[str] | None = None,
    tag: str | None = None,
    note: str | None = None,
) -> bool:
    """Append alpha result to the appropriate JSONL file (with dedup).

    Args:
        alpha_id: WQ BRAIN alpha identifier
        expression: FASTEXPR expression string
        status: One of SC_FAIL, LOW_SHARPE_FAIL, LOW_FITNESS_FAIL, OTHER_FAIL, SC_PENDING, ACTIVE
        region: Market region (e.g. USA)
        universe: Universe (e.g. TOP3000)
        neutralization: Neutralization method
        decay: Decay parameter
        delay: Signal delay
        fitness: WQ fitness value
        sharpe: Sharpe ratio
        returns: Annual returns
        turnover: Turnover ratio
        sc: Self-correlation value (None if not available)
        rev_src: Reversal source type (optional, LLM-provided semantic tag)
        fund: Fundamental fields used (optional, LLM-provided)
        tag: Submission tag for tracking
        note: Optional note

    Returns:
        True if record was appended, False if skipped (duplicate or error)
    """
    target_file = _STATUS_FILE_MAP.get(status)
    if not target_file:
        logger.warning(f"Unknown status '{status}' for alpha {alpha_id}, skipping record")
        return False

    if _alpha_exists_anywhere(alpha_id):
        logger.debug(f"Alpha {alpha_id} already recorded, skipping")
        return False

    record: dict[str, Any] = {
        "alpha_id": alpha_id,
        "expression": expression,
        "status": status,
        "region": region,
        "universe": universe,
        "neutralization": neutralization,
        "decay": decay,
        "delay": delay,
        "tested_date": date.today().isoformat(),
    }

    if fitness is not None:
        record["fitness"] = fitness
    if sharpe is not None:
        record["sharpe"] = sharpe
    if returns is not None:
        record["returns"] = returns
    if turnover is not None:
        record["turnover"] = turnover
    if sc is not None:
        record["sc"] = sc
    if rev_src is not None:
        record["rev_src"] = rev_src
    if fund is not None:
        record["fund"] = fund
    if tag is not None:
        record["tag"] = tag
    if note is not None:
        record["note"] = note

    # ACTIVE-specific field
    if status == "ACTIVE":
        record["date_active"] = date.today().isoformat()

    return _append_jsonl(target_file, record)


def move_timeout_to_resolved(alpha_id: str, final_status: str, sc: float | None = None) -> bool:
    """After finalize, move a timeout record to sc_blacklist or active_alphas.

    Args:
        alpha_id: The alpha to resolve
        final_status: SC_FAIL or ACTIVE
        sc: SC value if available

    Returns:
        True if successfully moved
    """
    record = _remove_from_file(JSONL_TIMEOUTS, alpha_id)
    if record is None:
        logger.warning(f"Alpha {alpha_id} not found in timeouts.jsonl")
        return False

    record["status"] = final_status
    if sc is not None:
        record["sc"] = sc

    if final_status == "ACTIVE":
        record["date_active"] = date.today().isoformat()
        return _append_jsonl(JSONL_ACTIVE, record)
    else:
        return _append_jsonl(JSONL_SC_BLACKLIST, record)


def query_blacklist(
    max_sc: float | None = None,
    neutralization: str | None = None,
    region: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Query sc_blacklist.jsonl with optional filters. Returns list sorted by SC ascending."""
    records = _read_jsonl(JSONL_SC_BLACKLIST)

    if max_sc is not None:
        records = [r for r in records if r.get("sc") is not None and r["sc"] <= max_sc]
    if neutralization:
        records = [r for r in records if r.get("neutralization") == neutralization]
    if region:
        records = [r for r in records if r.get("region") == region]

    records.sort(key=lambda r: r.get("sc") or 999)
    return records[:limit]


def get_active_alphas() -> list[dict]:
    """Return all ACTIVE alpha records."""
    return _read_jsonl(JSONL_ACTIVE)


def get_timeouts() -> list[dict]:
    """Return all timeout/pending records."""
    return _read_jsonl(JSONL_TIMEOUTS)


def annotate_alpha(
    alpha_id: str,
    rev_src: str | None = None,
    fund: list[str] | None = None,
    note: str | None = None,
) -> dict:
    """Annotate an existing alpha record with semantic fields.

    Searches across all 3 JSONL files, updates the record in-place.

    Args:
        alpha_id: The alpha to annotate
        rev_src: Reversal source type
        fund: Fundamental fields list
        note: Free-text note from LLM

    Returns:
        {"ok": True, "file": filename, "record": updated_record} or {"ok": False, "error": ...}
    """
    for filepath in [JSONL_SC_BLACKLIST, JSONL_ACTIVE, JSONL_TIMEOUTS]:
        if _alpha_in_file(filepath, alpha_id):
            updated = _update_record_fields(
                filepath, alpha_id,
                rev_src=rev_src, fund=fund, note=note,
            )
            if updated:
                return {"ok": True, "file": Path(filepath).name, "record": updated}

    return {"ok": False, "error": f"alpha_id {alpha_id} not found in any JSONL file"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _alpha_exists_anywhere(alpha_id: str) -> bool:
    """Check if alpha_id exists in any of the JSONL files."""
    for filepath in [JSONL_SC_BLACKLIST, JSONL_ACTIVE, JSONL_TIMEOUTS]:
        if _alpha_in_file(filepath, alpha_id):
            return True
    return False


def _alpha_in_file(filepath: str, alpha_id: str) -> bool:
    """Check if alpha_id exists in a specific JSONL file."""
    path = Path(filepath)
    if not path.exists():
        return False
    search_str = f'"alpha_id":"{alpha_id}"'
    # Also handle space after colon
    search_str_alt = f'"alpha_id": "{alpha_id}"'
    try:
        content = path.read_text(encoding="utf-8")
        return search_str in content or search_str_alt in content
    except OSError:
        return False


def _append_jsonl(filepath: str, record: dict) -> bool:
    """Append a single JSON record as a new line to the file."""
    path = Path(filepath)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"Recorded alpha {record.get('alpha_id')} → {path.name}")
        return True
    except OSError as e:
        logger.error(f"Failed to write to {filepath}: {e}")
        return False


def _read_jsonl(filepath: str) -> list[dict]:
    """Read all records from a JSONL file."""
    path = Path(filepath)
    if not path.exists():
        return []
    records = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"Error reading {filepath}: {e}")
    return records


def _remove_from_file(filepath: str, alpha_id: str) -> dict | None:
    """Remove a record by alpha_id from a JSONL file. Returns the removed record or None."""
    path = Path(filepath)
    if not path.exists():
        return None

    lines = path.read_text(encoding="utf-8").splitlines()
    remaining = []
    removed = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            if record.get("alpha_id") == alpha_id:
                removed = record
            else:
                remaining.append(line)
        except json.JSONDecodeError:
            remaining.append(line)

    if removed is not None:
        path.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")

    return removed


def _update_record_fields(
    filepath: str,
    alpha_id: str,
    rev_src: str | None = None,
    fund: list[str] | None = None,
    note: str | None = None,
) -> dict | None:
    """Update specific fields of a record in a JSONL file. Returns updated record or None."""
    path = Path(filepath)
    if not path.exists():
        return None

    lines = path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    updated = None

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            if record.get("alpha_id") == alpha_id:
                if rev_src is not None:
                    record["rev_src"] = rev_src
                if fund is not None:
                    record["fund"] = fund
                if note is not None:
                    record["note"] = note
                updated = record
            new_lines.append(json.dumps(record, ensure_ascii=False))
        except json.JSONDecodeError:
            new_lines.append(line)

    if updated is not None:
        path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    return updated
