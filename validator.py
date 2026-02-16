#!/usr/bin/env python3
"""Validate OCR results against point-in-time monitor truth and emit JSONL report."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_VALIDATOR_CONFIG: dict[str, Any] = {
    "tolerances": {
        "HR": 1.0,
        "ART": 1.0,
        "CVP": 1.0,
        "RAP": 1.0,
        "RR": 1.0,
        "EtCO2": 1.0,
        "SpO2": 1.0,
        "PEEP": 1.0,
        "NO": 1.0,
        "BSR": 1.0,
        "TSKIN": 0.1,
        "TRECT": 0.1,
        "VTe": 1.0,
        "VTi": 1.0,
        "Ppeak": 1.0,
        "O2conc": 1.0,
    }
}


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_numeric_with_reason(raw_text: Any, raw_value: Any) -> tuple[float | None, str | None]:
    value = safe_float(raw_value)
    if value is not None:
        return value, None

    text = "" if raw_text is None else str(raw_text).strip()
    if not text:
        return None, "empty"

    return None, f"invalid_numeric:{text}"


def parse_iso8601(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt


def ensure_validator_config(path: Path) -> dict[str, Any]:
    config = dict(DEFAULT_VALIDATOR_CONFIG)
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            config.update(loaded)
    else:
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    tolerances = config.get("tolerances")
    if not isinstance(tolerances, dict):
        config["tolerances"] = {}
    return config


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            row = line.strip()
            if not row:
                continue
            try:
                payload = json.loads(row)
            except json.JSONDecodeError:
                print(f"[WARN] jsonl line parse failed: line={idx}")
                continue
            if isinstance(payload, dict):
                yield payload


def load_recent_records(path: Path, last: int) -> list[dict[str, Any]]:
    if last <= 0:
        return list(iter_jsonl(path))
    buffer: deque[dict[str, Any]] = deque(maxlen=last)
    for record in iter_jsonl(path):
        buffer.append(record)
    return list(buffer)


def extract_truth_map(truth_json: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    beds = truth_json.get("beds")
    if not isinstance(beds, dict):
        return out

    for bed, bed_payload in beds.items():
        if not isinstance(bed_payload, dict):
            continue
        out[bed] = {}

        vitals_section = bed_payload.get("vitals")
        if isinstance(vitals_section, dict):
            for vital, vital_payload in vitals_section.items():
                if isinstance(vital_payload, dict):
                    out[bed][vital] = vital_payload.get("value")
                else:
                    out[bed][vital] = vital_payload

        for vital, vital_payload in bed_payload.items():
            if vital in {"vitals", "values"}:
                continue
            if vital in out[bed]:
                continue
            if isinstance(vital_payload, dict) and "value" in vital_payload:
                out[bed][vital] = vital_payload.get("value")

    return out


def extract_message_datetime(truth_json: dict[str, Any]) -> datetime | None:
    for key in ("message_datetime", "timestamp", "captured_at"):
        dt = parse_iso8601(truth_json.get(key))
        if dt is not None:
            return dt

    meta = truth_json.get("meta")
    if isinstance(meta, dict):
        dt = parse_iso8601(meta.get("message_datetime"))
        if dt is not None:
            return dt

    return None


def resolve_snapshot_path(raw_path: Any, ocr_results_path: Path) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(str(raw_path))
    if candidate.is_absolute() and candidate.exists():
        return candidate

    relative_candidates = [Path.cwd() / candidate, ocr_results_path.parent / candidate]
    for path in relative_candidates:
        if path.exists():
            return path
    return None


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * p
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return sorted_values[low]
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def validate_records(
    ocr_records: list[dict[str, Any]],
    monitor_cache_payload: dict[str, Any],
    monitor_truth: dict[str, dict[str, Any]],
    ocr_results_path: Path,
    tolerances: dict[str, Any],
    dump_mismatch: int,
) -> tuple[list[dict[str, Any]], dict[str, float], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    abs_errors: list[float] = []
    dt_seconds: list[float] = []

    total = 0
    matched = 0
    missing = 0
    invalid = 0
    snapshot_used = 0
    fallback_used = 0
    snapshot_missing = 0
    snapshot_load_error = 0
    dt_skipped = 0

    for rec in ocr_records:
        beds_payload = rec.get("beds", {})
        if not isinstance(beds_payload, dict):
            beds_payload = {}

        snapshot_source = "monitor_cache_fallback"
        truth_payload = monitor_cache_payload
        truth_map = monitor_truth

        snapshot_path = resolve_snapshot_path(rec.get("cache_snapshot_path"), ocr_results_path)
        if snapshot_path is None:
            snapshot_missing += 1
            fallback_used += 1
        else:
            try:
                truth_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
                if not isinstance(truth_payload, dict):
                    raise ValueError("snapshot is not a JSON object")
                truth_map = extract_truth_map(truth_payload)
                snapshot_source = "snapshot"
                snapshot_used += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] failed to load snapshot {snapshot_path}: {exc}")
                snapshot_load_error += 1
                fallback_used += 1
                truth_payload = monitor_cache_payload
                truth_map = monitor_truth

        ocr_dt = parse_iso8601(rec.get("timestamp"))
        truth_dt = extract_message_datetime(truth_payload)
        dt_sec: float | None = None
        if ocr_dt is not None and truth_dt is not None:
            dt_sec = abs((ocr_dt - truth_dt).total_seconds())
            dt_seconds.append(dt_sec)
        else:
            dt_skipped += 1

        per_item: list[dict[str, Any]] = []
        for bed, truth_vitals in truth_map.items():
            if not isinstance(truth_vitals, dict):
                continue
            ocr_bed_payload = beds_payload.get(bed, {})
            if not isinstance(ocr_bed_payload, dict):
                ocr_bed_payload = {}

            for vital, truth_raw in truth_vitals.items():
                total += 1
                truth_value = safe_float(truth_raw)

                ocr_vital_payload = ocr_bed_payload.get(vital, {})
                if not isinstance(ocr_vital_payload, dict):
                    ocr_vital_payload = {}

                ocr_text = ocr_vital_payload.get("text")
                ocr_raw_value = ocr_vital_payload.get("value")
                confidence = safe_float(ocr_vital_payload.get("confidence"))
                ocr_value, invalid_reason = parse_numeric_with_reason(ocr_text, ocr_raw_value)

                tol = safe_float(tolerances.get(vital, 0.0))
                tolerance = tol if tol is not None else 0.0

                is_match = False
                abs_error: float | None = None
                if truth_value is None or ocr_value is None:
                    if ocr_value is None:
                        missing += 1
                    if invalid_reason:
                        invalid += 1
                else:
                    abs_error = abs(ocr_value - truth_value)
                    abs_errors.append(abs_error)
                    is_match = abs_error <= tolerance
                    if is_match:
                        matched += 1

                comparison = {
                    "bed": bed,
                    "vital": vital,
                    "truth": truth_raw,
                    "ocr_text": ocr_text,
                    "ocr_value": ocr_raw_value,
                    "parsed_ocr_value": ocr_value,
                    "tolerance": tolerance,
                    "abs_error": abs_error,
                    "match": is_match,
                    "invalid_reason": invalid_reason,
                    "confidence": confidence,
                    "image_path": rec.get("image_path"),
                    "cache_snapshot_path": rec.get("cache_snapshot_path"),
                }
                per_item.append(comparison)

                if not is_match:
                    mismatch_error = abs_error if abs_error is not None else float("inf")
                    mismatches.append(
                        {
                            "bed": bed,
                            "vital": vital,
                            "ocr_value": ocr_raw_value,
                            "truth_value": truth_raw,
                            "abs_error": abs_error,
                            "_sort_error": mismatch_error,
                            "confidence": confidence,
                            "image_path": rec.get("image_path"),
                            "cache_snapshot_path": rec.get("cache_snapshot_path"),
                        }
                    )

        details.append(
            {
                "timestamp": rec.get("timestamp"),
                "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "truth_source": snapshot_source,
                "delta_t_seconds": dt_sec,
                "comparisons": per_item,
            }
        )

    metrics: dict[str, float] = {
        "total": float(total),
        "match_rate": (matched / total) if total else 0.0,
        "mean_abs_error": (sum(abs_errors) / len(abs_errors)) if abs_errors else 0.0,
        "median_abs_error": statistics.median(abs_errors) if abs_errors else 0.0,
        "missing_rate": (missing / total) if total else 0.0,
        "invalid_rate": (invalid / total) if total else 0.0,
        "snapshot_used": float(snapshot_used),
        "fallback_used": float(fallback_used),
        "snapshot_missing": float(snapshot_missing),
        "snapshot_load_error": float(snapshot_load_error),
        "dt_mean_seconds": (sum(dt_seconds) / len(dt_seconds)) if dt_seconds else 0.0,
        "dt_median_seconds": statistics.median(dt_seconds) if dt_seconds else 0.0,
        "dt_p90_seconds": percentile(dt_seconds, 0.9) if dt_seconds else 0.0,
        "dt_evaluated": float(len(dt_seconds)),
        "dt_skipped": float(dt_skipped),
    }

    if dump_mismatch > 0:
        mismatches.sort(key=lambda item: item["_sort_error"], reverse=True)
        mismatches = mismatches[:dump_mismatch]
    else:
        mismatches = []

    for item in mismatches:
        item.pop("_sort_error", None)

    return details, metrics, mismatches


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate OCR JSONL against cache snapshots (or monitor_cache fallback)"
    )
    parser.add_argument("--ocr-results", required=True, help="Path to dataset/.../ocr_results.jsonl")
    parser.add_argument("--monitor-cache", required=True, help="Path to monitor_cache.json")
    parser.add_argument("--validator-config", default="validator_config.json", help="Path to validator config JSON")
    parser.add_argument("--last", type=int, default=50, help="Use latest N OCR records")
    parser.add_argument(
        "--dump-mismatch",
        type=int,
        default=0,
        help="Print top-N mismatch rows (bed/vital/value/confidence/path)",
    )
    args = parser.parse_args()

    ocr_path = Path(args.ocr_results)
    cache_path = Path(args.monitor_cache)
    config_path = Path(args.validator_config)

    if not ocr_path.exists():
        raise FileNotFoundError(f"OCR file not found: {ocr_path}")
    if not cache_path.exists():
        raise FileNotFoundError(f"monitor_cache file not found: {cache_path}")

    cfg = ensure_validator_config(config_path)
    tolerances = cfg.get("tolerances", {}) if isinstance(cfg, dict) else {}

    records = load_recent_records(ocr_path, args.last)

    monitor_cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(monitor_cache_payload, dict):
        raise ValueError(f"monitor_cache file is not a JSON object: {cache_path}")
    monitor_truth = extract_truth_map(monitor_cache_payload)

    rows, metrics, mismatches = validate_records(
        ocr_records=records,
        monitor_cache_payload=monitor_cache_payload,
        monitor_truth=monitor_truth,
        ocr_results_path=ocr_path,
        tolerances=tolerances,
        dump_mismatch=args.dump_mismatch,
    )

    output_path = ocr_path.parent / "validation_results.jsonl"
    write_jsonl(output_path, rows)

    print(f"[INFO] validated_records={len(rows)} output={output_path}")
    print(f"[INFO] match_rate={metrics['match_rate']:.4f}")
    print(f"[INFO] MAE={metrics['mean_abs_error']:.4f}")
    print(f"[INFO] median_abs_error={metrics['median_abs_error']:.4f}")
    print(f"[INFO] missing_rate={metrics['missing_rate']:.4f}")
    print(f"[INFO] invalid_rate={metrics['invalid_rate']:.4f}")
    print(
        "[INFO] truth_source_stats "
        f"snapshot_used={int(metrics['snapshot_used'])} "
        f"fallback_used={int(metrics['fallback_used'])} "
        f"snapshot_missing={int(metrics['snapshot_missing'])} "
        f"snapshot_load_error={int(metrics['snapshot_load_error'])}"
    )
    print(
        "[INFO] delta_t_seconds "
        f"mean={metrics['dt_mean_seconds']:.3f} "
        f"median={metrics['dt_median_seconds']:.3f} "
        f"p90={metrics['dt_p90_seconds']:.3f} "
        f"evaluated={int(metrics['dt_evaluated'])} "
        f"skipped={int(metrics['dt_skipped'])}"
    )

    if args.dump_mismatch > 0:
        print(f"[INFO] dump_mismatch_top={len(mismatches)}")
        for idx, item in enumerate(mismatches, start=1):
            print(
                f"[MISMATCH {idx}] "
                f"bed={item['bed']} vital={item['vital']} "
                f"ocr_value={item['ocr_value']} truth_value={item['truth_value']} "
                f"abs_error={item['abs_error']} confidence={item['confidence']} "
                f"image_path={item['image_path']} cache_snapshot_path={item['cache_snapshot_path']}"
            )


if __name__ == "__main__":
    main()
