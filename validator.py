#!/usr/bin/env python3
"""Validate OCR results against monitor_cache.json and emit JSONL report."""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

# MOD: デフォルト許容誤差（vital別）
DEFAULT_VALIDATOR_CONFIG: dict[str, Any] = {
    "tolerances": {
        "HR": 1.0,
        "SpO2": 1.0,
        "RR": 1.0,
        "TSKIN": 0.2,
        "TRECT": 0.2,
    }
}


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def parse_numeric_with_reason(raw_text: Any, raw_value: Any) -> tuple[float | None, str | None]:
    value = safe_float(raw_value)
    if value is not None:
        return value, None
    text = "" if raw_text is None else str(raw_text)
    if not text.strip():
        return None, "empty"
    return None, f"invalid_numeric:{text}"


def ensure_validator_config(path: Path) -> dict[str, Any]:
    config = dict(DEFAULT_VALIDATOR_CONFIG)
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            config.update(loaded)
    else:
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    if "tolerances" not in config or not isinstance(config["tolerances"], dict):
        config["tolerances"] = {}
    return config


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                print(f"[WARN] jsonl line parse failed: line={idx}")
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def extract_truth_map(monitor_cache: dict[str, Any]) -> dict[str, dict[str, Any]]:
    beds = monitor_cache.get("beds", {})
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(beds, dict):
        return out
    for bed, payload in beds.items():
        if not isinstance(payload, dict):
            continue
        vitals = payload.get("vitals", payload.get("values", {}))
        if not isinstance(vitals, dict):
            continue
        out[bed] = {}
        for vital, vital_payload in vitals.items():
            if isinstance(vital_payload, dict):
                out[bed][vital] = vital_payload.get("value")
            else:
                out[bed][vital] = vital_payload
    return out


def validate_records(
    ocr_records: list[dict[str, Any]],
    monitor_truth: dict[str, dict[str, Any]],
    tolerances: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    details: list[dict[str, Any]] = []
    abs_errors: list[float] = []
    total = 0
    matched = 0
    missing = 0
    invalid = 0

    for rec in ocr_records:
        beds_payload = rec.get("beds", {})
        if not isinstance(beds_payload, dict):
            continue

        per_item: list[dict[str, Any]] = []
        for bed, truth_vitals in monitor_truth.items():
            ocr_vitals = beds_payload.get(bed, {}) if isinstance(beds_payload.get(bed, {}), dict) else {}
            for vital, truth_raw in truth_vitals.items():
                total += 1
                truth_val = safe_float(truth_raw)
                ocr_payload = ocr_vitals.get(vital, {}) if isinstance(ocr_vitals, dict) else {}
                if not isinstance(ocr_payload, dict):
                    ocr_payload = {}
                ocr_text = ocr_payload.get("text")
                ocr_raw_val = ocr_payload.get("value")
                ocr_val, invalid_reason = parse_numeric_with_reason(ocr_text, ocr_raw_val)

                tol = safe_float(tolerances.get(vital, 0.0))
                tolerance = tol if tol is not None else 0.0

                if truth_val is None or ocr_val is None:
                    is_match = False
                    abs_error = None
                    if ocr_val is None:
                        missing += 1
                    if invalid_reason:
                        invalid += 1
                else:
                    abs_error = abs(ocr_val - truth_val)
                    abs_errors.append(abs_error)
                    is_match = abs_error <= tolerance
                    if is_match:
                        matched += 1

                per_item.append(
                    {
                        "bed": bed,
                        "vital": vital,
                        "truth": truth_raw,
                        "ocr_text": ocr_text,
                        "ocr_value": ocr_raw_val,
                        "parsed_ocr_value": ocr_val,
                        "tolerance": tolerance,
                        "abs_error": abs_error,
                        "match": is_match,
                        "invalid_reason": invalid_reason,
                    }
                )

        details.append(
            {
                "timestamp": rec.get("timestamp"),
                "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "comparisons": per_item,
            }
        )

    metrics = {
        "total": float(total),
        "match_rate": (matched / total) if total else 0.0,
        "mean_abs_error": (sum(abs_errors) / len(abs_errors)) if abs_errors else 0.0,
        "median_abs_error": statistics.median(abs_errors) if abs_errors else 0.0,
        "missing_rate": (missing / total) if total else 0.0,
        "invalid_rate": (invalid / total) if total else 0.0,
    }
    return details, metrics


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate OCR JSONL against monitor_cache.json")
    parser.add_argument("--ocr-results", required=True, help="Path to dataset/.../ocr_results.jsonl")
    parser.add_argument("--monitor-cache", required=True, help="Path to monitor_cache.json")
    parser.add_argument("--validator-config", default="validator_config.json", help="Path to validator config JSON")
    parser.add_argument("--last", type=int, default=50, help="Use latest N OCR records")
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

    records = load_jsonl(ocr_path)
    if args.last > 0:
        records = records[-args.last :]

    cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    truth = extract_truth_map(cache_payload)

    rows, metrics = validate_records(records, truth, tolerances)

    output_path = ocr_path.parent / "validation_results.jsonl"
    write_jsonl(output_path, rows)

    print(f"[INFO] validated_records={len(rows)} output={output_path}")
    print(f"[INFO] match_rate={metrics['match_rate']:.4f}")
    print(f"[INFO] mean_abs_error={metrics['mean_abs_error']:.4f}")
    print(f"[INFO] median_abs_error={metrics['median_abs_error']:.4f}")
    print(f"[INFO] missing_rate={metrics['missing_rate']:.4f}")
    print(f"[INFO] invalid_rate={metrics['invalid_rate']:.4f}")


if __name__ == "__main__":
    main()
