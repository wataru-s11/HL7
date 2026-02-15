#!/usr/bin/env python3
"""
Validate OCR result frames against monitor_cache snapshots.

Usage:
  python validator.py --ocr dataset/20260215/ocr_results.jsonl --outdir dataset/20260215
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

TEMP_KEYS = {"TSKIN", "TRECT"}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def is_match(vital: str, truth: Any, ocr: Any) -> bool:
    t = safe_float(truth)
    o = safe_float(ocr)
    if t is None or o is None:
        return False
    if vital in TEMP_KEYS:
        return abs(o - t) <= 0.1
    return abs(o - t) == 0


def resolve_snapshot_path(snapshot_path: str, ocr_file: Path) -> Path:
    p = Path(snapshot_path)
    if p.is_absolute():
        return p
    by_ocr_parent = ocr_file.parent / p
    if by_ocr_parent.exists():
        return by_ocr_parent
    return Path.cwd() / p


def validate(ocr_path: Path, outdir: Path) -> tuple[int, int]:
    report_path = outdir / "validation_report.jsonl"
    summary_path = outdir / "validation_summary.csv"
    outdir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "n": 0.0,
            "match": 0.0,
            "null": 0.0,
            "absdiff_sum": 0.0,
            "absdiff_n": 0.0,
            "conf_sum": 0.0,
            "conf_n": 0.0,
        }
    )

    processed, skipped = 0, 0
    with ocr_path.open("r", encoding="utf-8") as src, report_path.open("w", encoding="utf-8") as rep:
        for line_no, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("line %d invalid json: %s", line_no, exc)
                skipped += 1
                continue

            snapshot_ref = record.get("cache_snapshot_path")
            if not snapshot_ref:
                logger.warning("line %d skipped: missing cache_snapshot_path", line_no)
                skipped += 1
                continue

            snapshot_path = resolve_snapshot_path(str(snapshot_ref), ocr_path)
            try:
                snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("line %d skipped: snapshot read failed (%s)", line_no, exc)
                skipped += 1
                continue

            frame_total = 0
            frame_match = 0
            bed_report: dict[str, Any] = {}
            truth_beds = snapshot.get("beds", {})
            ocr_beds = record.get("beds", {})

            for bed_id, bed_payload in truth_beds.items():
                truth_vitals = bed_payload.get("vitals", {}) if isinstance(bed_payload, dict) else {}
                ocr_vitals = ocr_beds.get(bed_id, {}) if isinstance(ocr_beds, dict) else {}
                details: dict[str, Any] = {}

                bed_total = 0
                bed_match = 0
                for vital, truth_payload in truth_vitals.items():
                    truth_value = truth_payload.get("value") if isinstance(truth_payload, dict) else None
                    ocr_payload = ocr_vitals.get(vital, {}) if isinstance(ocr_vitals, dict) else {}
                    ocr_value = ocr_payload.get("value") if isinstance(ocr_payload, dict) else None
                    conf = ocr_payload.get("confidence") if isinstance(ocr_payload, dict) else None

                    t = safe_float(truth_value)
                    o = safe_float(ocr_value)
                    diff = (o - t) if (t is not None and o is not None) else None
                    abs_diff = abs(diff) if diff is not None else None
                    match = is_match(vital, truth_value, ocr_value)

                    item = {
                        "truth": truth_value,
                        "ocr": ocr_value,
                        "diff": diff,
                        "abs_diff": abs_diff,
                        "match": match,
                        "confidence": conf,
                    }
                    details[vital] = item

                    bucket = stats[vital]
                    bucket["n"] += 1
                    if match:
                        bucket["match"] += 1
                        bed_match += 1
                        frame_match += 1
                    if o is None:
                        bucket["null"] += 1
                    if abs_diff is not None:
                        bucket["absdiff_sum"] += abs_diff
                        bucket["absdiff_n"] += 1
                    c = safe_float(conf)
                    if c is not None:
                        bucket["conf_sum"] += c
                        bucket["conf_n"] += 1

                    bed_total += 1
                    frame_total += 1

                bed_report[bed_id] = {
                    "match_rate": (bed_match / bed_total) if bed_total else None,
                    "vitals": details,
                }

            rep.write(
                json.dumps(
                    {
                        "line": line_no,
                        "timestamp": record.get("timestamp"),
                        "cache_snapshot_path": snapshot_ref,
                        "overall_match_rate": (frame_match / frame_total) if frame_total else None,
                        "beds": bed_report,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            rep.flush()
            processed += 1

    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["vital", "n", "match_rate", "mae", "null_rate", "mean_confidence"])
        writer.writeheader()
        for vital in sorted(stats):
            s = stats[vital]
            n = int(s["n"])
            writer.writerow(
                {
                    "vital": vital,
                    "n": n,
                    "match_rate": (s["match"] / n) if n else "",
                    "mae": (s["absdiff_sum"] / s["absdiff_n"]) if s["absdiff_n"] else "",
                    "null_rate": (s["null"] / n) if n else "",
                    "mean_confidence": (s["conf_sum"] / s["conf_n"]) if s["conf_n"] else "",
                }
            )

    return processed, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate OCR jsonl against monitor cache snapshots")
    parser.add_argument("--ocr", required=True, help="Path to ocr_results.jsonl")
    parser.add_argument("--outdir", required=True, help="Output directory for reports")
    args = parser.parse_args()

    ocr_path = Path(args.ocr)
    if not ocr_path.exists():
        raise FileNotFoundError(f"OCR file not found: {ocr_path}")

    processed, skipped = validate(ocr_path, Path(args.outdir))
    logger.info("validation done: processed=%d skipped=%d", processed, skipped)


if __name__ == "__main__":
    main()
