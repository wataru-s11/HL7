#!/usr/bin/env python3
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


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to read snapshot %s: %s", path, exc)
        return None


def is_match(vital: str, truth: float | int | None, ocr: float | int | None) -> bool:
    if truth is None or ocr is None:
        return False
    if vital in TEMP_KEYS:
        return abs(float(ocr) - float(truth)) <= 0.1
    return float(ocr) == float(truth)


def summarize_entry(bucket: dict[str, float], item: dict[str, Any]) -> None:
    bucket["n"] += 1
    if item["match"]:
        bucket["match_count"] += 1
    if item["ocr"] is None:
        bucket["null_count"] += 1
    if item["abs_diff"] is not None:
        bucket["abs_diff_sum"] += float(item["abs_diff"])
        bucket["abs_diff_n"] += 1
    conf = item["ocr_confidence"]
    if conf is not None:
        bucket["conf_sum"] += float(conf)
        bucket["conf_n"] += 1


def resolve_snapshot_path(snapshot_rel: str, ocr_path: Path) -> Path:
    candidate = Path(snapshot_rel)
    if candidate.is_absolute():
        return candidate
    from_ocr_dir = ocr_path.parent / candidate
    if from_ocr_dir.exists():
        return from_ocr_dir
    return Path.cwd() / candidate


def validate_frames(ocr_path: Path, outdir: Path) -> tuple[int, int]:
    report_path = outdir / "validation_report.jsonl"
    summary_csv = outdir / "validation_summary.csv"

    processed = 0
    skipped = 0
    vital_stats: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "n": 0.0,
            "match_count": 0.0,
            "null_count": 0.0,
            "abs_diff_sum": 0.0,
            "abs_diff_n": 0.0,
            "conf_sum": 0.0,
            "conf_n": 0.0,
        }
    )

    with ocr_path.open("r", encoding="utf-8") as src, report_path.open("w", encoding="utf-8") as report:
        for lineno, line in enumerate(src, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            snapshot_rel = record.get("cache_snapshot_path")
            if not snapshot_rel:
                logger.warning("line %d skipped: cache_snapshot_path is missing", lineno)
                skipped += 1
                continue

            snapshot_path = resolve_snapshot_path(snapshot_rel, ocr_path)
            snapshot = load_json(snapshot_path)
            if snapshot is None:
                logger.warning("line %d skipped: cache_snapshot_path unreadable (%s)", lineno, snapshot_rel)
                skipped += 1
                continue

            truth_beds = snapshot.get("beds", {})
            ocr_beds = record.get("beds", {})

            bed_reports: dict[str, Any] = {}
            frame_total = 0
            frame_match = 0

            for bed_id, truth_bed in truth_beds.items():
                truth_vitals = truth_bed.get("vitals", {}) if isinstance(truth_bed, dict) else {}
                ocr_vitals = ocr_beds.get(bed_id, {}) if isinstance(ocr_beds, dict) else {}

                vitals_report: dict[str, Any] = {}
                bed_total = 0
                bed_match = 0

                for vital, truth_payload in truth_vitals.items():
                    truth_value = truth_payload.get("value") if isinstance(truth_payload, dict) else None
                    ocr_payload = ocr_vitals.get(vital, {}) if isinstance(ocr_vitals, dict) else {}
                    ocr_value = ocr_payload.get("value") if isinstance(ocr_payload, dict) else None
                    ocr_confidence = ocr_payload.get("confidence") if isinstance(ocr_payload, dict) else None

                    diff = None
                    abs_diff = None
                    if truth_value is not None and ocr_value is not None:
                        diff = float(ocr_value) - float(truth_value)
                        abs_diff = abs(diff)

                    matched = is_match(vital, truth_value, ocr_value)
                    if matched:
                        bed_match += 1
                        frame_match += 1
                    bed_total += 1
                    frame_total += 1

                    item = {
                        "truth": truth_value,
                        "ocr": ocr_value,
                        "diff": diff,
                        "abs_diff": abs_diff,
                        "match": matched,
                        "ocr_confidence": ocr_confidence,
                    }
                    vitals_report[vital] = item
                    summarize_entry(vital_stats[vital], item)

                bed_reports[bed_id] = {
                    "match_rate": (bed_match / bed_total) if bed_total else None,
                    "vitals": vitals_report,
                }

            frame_report = {
                "line": lineno,
                "timestamp": record.get("timestamp"),
                "cache_snapshot_path": snapshot_rel,
                "overall_match_rate": (frame_match / frame_total) if frame_total else None,
                "beds": bed_reports,
            }
            report.write(json.dumps(frame_report, ensure_ascii=False) + "\n")
            processed += 1

    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["vital", "n", "match_rate", "mae", "null_rate", "mean_confidence"],
        )
        writer.writeheader()
        for vital in sorted(vital_stats):
            stat = vital_stats[vital]
            n = int(stat["n"])
            writer.writerow(
                {
                    "vital": vital,
                    "n": n,
                    "match_rate": (stat["match_count"] / n) if n else "",
                    "mae": (stat["abs_diff_sum"] / stat["abs_diff_n"]) if stat["abs_diff_n"] else "",
                    "null_rate": (stat["null_count"] / n) if n else "",
                    "mean_confidence": (stat["conf_sum"] / stat["conf_n"]) if stat["conf_n"] else "",
                }
            )

    return processed, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate OCR results against monitor cache snapshots")
    parser.add_argument("--ocr", default="dataset/ocr_results.jsonl")
    parser.add_argument("--outdir", default="dataset")
    args = parser.parse_args()

    ocr_path = Path(args.ocr)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not ocr_path.exists():
        raise FileNotFoundError(f"OCR file not found: {ocr_path}")

    processed, skipped = validate_frames(ocr_path, outdir)
    logger.info("validation done: processed=%d skipped=%d", processed, skipped)


if __name__ == "__main__":
    main()
