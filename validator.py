#!/usr/bin/env python3
"""Validate OCR results against point-in-time monitor truth and emit JSONL report."""

from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import statistics
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
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

FLOAT_TOLERANCE_VITALS = {"TSKIN", "TRECT"}


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


def parse_score_fields(raw_fields: str | None) -> set[str] | None:
    if raw_fields is None:
        return None
    parsed = {item.strip() for item in raw_fields.split(",") if item.strip()}
    return parsed or None


def _is_integer_like(value: Any) -> bool:
    try:
        return float(value).is_integer()
    except (TypeError, ValueError):
        return False


def mismatch_tolerance(vital: str, ocr_value: float, truth_value: float) -> float:
    if vital in FLOAT_TOLERANCE_VITALS:
        return 0.2
    if not _is_integer_like(ocr_value) or not _is_integer_like(truth_value):
        return 0.2
    return 1.0


JST = timezone(timedelta(hours=9))


def parse_iso8601(value: Any) -> datetime | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    normalized = text
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    if " " in normalized and "T" not in normalized:
        normalized = normalized.replace(" ", "T", 1)

    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=JST)
    return dt
SNAPSHOT_FILENAME_PATTERN = re.compile(r"(\d{8})_(\d{6})_(\d{3})_monitor_cache\.json$")


@dataclass
class SnapshotCandidate:
    path: Path
    timestamp: datetime


@dataclass
class SnapshotData:
    payload: dict[str, Any]
    truth_map: dict[str, dict[str, Any]]


def parse_snapshot_timestamp(path: Path) -> datetime:
    match = SNAPSHOT_FILENAME_PATTERN.search(path.name)
    if match:
        day_part, time_part, ms_part = match.groups()
        parsed = datetime.strptime(day_part + time_part + ms_part, "%Y%m%d%H%M%S%f")
        return parsed.replace(tzinfo=JST)
    return datetime.fromtimestamp(path.stat().st_mtime, tz=JST)


def load_snapshot_candidates(cache_dir: Path) -> list[SnapshotCandidate]:
    if not cache_dir.exists() or not cache_dir.is_dir():
        return []

    candidates: list[SnapshotCandidate] = []
    for path in cache_dir.glob("*_monitor_cache.json"):
        try:
            ts = parse_snapshot_timestamp(path)
            candidates.append(SnapshotCandidate(path=path, timestamp=ts))
        except OSError:
            continue
    candidates.sort(key=lambda item: item.timestamp)
    return candidates


def build_snapshot_timestamps(cache_candidates: list[SnapshotCandidate]) -> list[float]:
    return [candidate.timestamp.timestamp() for candidate in cache_candidates]


def load_snapshot_data(snapshot_path: Path, snapshot_cache: dict[str, SnapshotData]) -> SnapshotData:
    cache_key = str(snapshot_path)
    cached = snapshot_cache.get(cache_key)
    if cached is not None:
        return cached

    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("snapshot is not a JSON object")
    loaded = SnapshotData(payload=payload, truth_map=extract_truth_map(payload))
    snapshot_cache[cache_key] = loaded
    return loaded


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
    for key in ("updated_at", "captured_at", "message_datetime", "timestamp"):
        dt = parse_iso8601(truth_json.get(key))
        if dt is not None:
            return dt

    meta = truth_json.get("meta")
    if isinstance(meta, dict):
        dt = parse_iso8601(meta.get("message_datetime"))
        if dt is not None:
            return dt

    beds = truth_json.get("beds")
    if isinstance(beds, dict):
        for bed_payload in beds.values():
            if not isinstance(bed_payload, dict):
                continue
            dt = parse_iso8601(bed_payload.get("message_datetime"))
            if dt is not None:
                return dt

    return None


def resolve_snapshot_path(raw_path: Any, ocr_results_path: Path) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(str(raw_path)).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate.resolve()

    relative_candidates = [Path.cwd() / candidate, ocr_results_path.parent / candidate]
    for path in relative_candidates:
        if path.exists():
            return path.resolve()
    return None


def get_record_capture_timestamp(rec: dict[str, Any]) -> datetime | None:
    for key in ("capture_timestamp", "timestamp"):
        dt = parse_iso8601(rec.get(key))
        if dt is not None:
            return dt
    return None


def get_record_time(rec: dict[str, Any]) -> tuple[datetime | None, str | None]:
    for key in ("ocr_timestamp", "capture_timestamp", "timestamp"):
        dt = parse_iso8601(rec.get(key))
        if dt is not None:
            return dt, key
    return None, None


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
    monitor_cache_payload: dict[str, Any] | None,
    monitor_truth: dict[str, dict[str, Any]] | None,
    ocr_results_path: Path,
    tolerances: dict[str, Any],
    dump_mismatch: int,
    debug_mismatches: int,
    cache_candidates: list[SnapshotCandidate] | None,
    cache_candidate_timestamps: list[float] | None,
    lookback_sec: float,
    max_candidates: int,
    lag_sec: float,
    fallback_to_nearest_past: bool,
    score_fields: set[str] | None,
    mismatch_details_enabled: bool,
    time_window_sec: float,
) -> tuple[list[dict[str, Any]], dict[str, float], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    abs_errors: list[float] = []
    dt_seconds: list[float] = []

    total = 0
    matched = 0
    missing = 0
    invalid = 0
    snapshot_before_used = 0
    snapshot_used = 0
    explicit_used = 0
    snapshot_missing = 0
    snapshot_load_error = 0
    skipped_truth_unavailable = 0
    dt_skipped = 0
    nearest_fallback_used = 0
    dt_skip_reason_counts: Counter[str] = Counter()
    skip_reason_counts: Counter[str] = Counter()
    snapshot_data_cache: dict[str, SnapshotData] = {}
    mismatch_details_rows: list[dict[str, Any]] = []

    def evaluate_record(
        rec: dict[str, Any],
        truth_map: dict[str, dict[str, Any]],
        truth_payload: dict[str, Any],
        ocr_dt_corrected: datetime | None,
        truth_dt: datetime | None,
        include_mismatch: bool,
    ) -> tuple[list[dict[str, Any]], dict[str, float], list[dict[str, Any]], float | None, str | None]:
        beds_payload = rec.get("beds", {})
        if not isinstance(beds_payload, dict):
            beds_payload = {}

        resolved_truth_dt = truth_dt if truth_dt is not None else extract_message_datetime(truth_payload)
        dt_sec: float | None = None
        dt_skip_reason: str | None = None
        if ocr_dt_corrected is not None and resolved_truth_dt is not None:
            dt_sec = (ocr_dt_corrected - resolved_truth_dt).total_seconds()
        elif ocr_dt_corrected is None and resolved_truth_dt is None:
            dt_skip_reason = "ocr_dt_none+truth_dt_none"
        elif ocr_dt_corrected is None:
            dt_skip_reason = "ocr_dt_none"
        else:
            dt_skip_reason = "truth_dt_none"

        per_item: list[dict[str, Any]] = []
        record_abs_errors: list[float] = []
        record_mismatches: list[dict[str, Any]] = []
        record_total = 0
        record_matched = 0
        for bed, truth_vitals in truth_map.items():
            if not isinstance(truth_vitals, dict):
                continue
            ocr_bed_payload = beds_payload.get(bed, {})
            if not isinstance(ocr_bed_payload, dict):
                ocr_bed_payload = {}

            for vital, truth_raw in truth_vitals.items():
                record_total += 1
                scoring_enabled = score_fields is None or vital in score_fields
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
                    pass
                else:
                    abs_error = abs(ocr_value - truth_value)
                    if scoring_enabled:
                        record_abs_errors.append(abs_error)
                        is_match = abs_error <= tolerance
                        if is_match:
                            record_matched += 1

                    if mismatch_details_enabled:
                        detail_tol = mismatch_tolerance(vital, ocr_value, truth_value)
                        if abs_error > detail_tol:
                            mismatch_details_rows.append(
                                {
                                    "timestamp": rec.get("timestamp"),
                                    "capture_timestamp": rec.get("capture_timestamp", rec.get("timestamp")),
                                    "bed": bed,
                                    "field": vital,
                                    "ocr_value": ocr_raw_value,
                                    "truth_value": truth_raw,
                                    "abs_error": abs_error,
                                }
                            )

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

                if include_mismatch and not is_match:
                    mismatch_error = abs_error if abs_error is not None else float("inf")
                    record_mismatches.append(
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
                            "truth_snapshot_used": None,
                        }
                    )

        mean_abs = (sum(record_abs_errors) / len(record_abs_errors)) if record_abs_errors else float("inf")
        score = {
            "total": float(record_total),
            "matched": float(record_matched),
            "mean_abs_error": mean_abs,
        }
        return per_item, score, record_mismatches, dt_sec, dt_skip_reason

    for rec in ocr_records:
        record_time, record_time_source = get_record_time(rec)
        if record_time is None:
            skip_reason = "record_time_unavailable"
            skip_reason_counts[skip_reason] += 1
            print(
                f"[WARN] skip_record reason={skip_reason} "
                f"timestamp={rec.get('timestamp')} image_path={rec.get('image_path')}"
            )
            continue

        ocr_dt_corrected = record_time + timedelta(seconds=lag_sec)

        selected_payload: dict[str, Any] = monitor_cache_payload or {}
        selected_map: dict[str, dict[str, Any]] = monitor_truth or {}
        selected_snapshot_path: str | None = None
        selected_truth_dt: datetime | None = None
        selected_source = "monitor_cache_fallback"
        selected_dt_source = "none"
        selected_candidate_count = 0

        chosen_per_item: list[dict[str, Any]] = []
        chosen_score = {"matched": -1.0, "mean_abs_error": float("inf"), "delta": float("inf")}
        chosen_dt_sec: float | None = None

        explicit_candidates = [
            ("cache_snapshot_path_before", rec.get("cache_snapshot_path_before")),
            ("cache_snapshot_path", rec.get("cache_snapshot_path")),
        ]
        explicit_deltas: dict[str, float | None] = {
            "cache_snapshot_path_before": None,
            "cache_snapshot_path": None,
        }
        explicit_invalid_reason: str | None = None
        explicit_selected = False
        for explicit_key, explicit_raw_path in explicit_candidates:
            if explicit_raw_path is None:
                continue
            snapshot_path = resolve_snapshot_path(explicit_raw_path, ocr_results_path)
            if snapshot_path is None:
                snapshot_missing += 1
                print(f"[WARN] explicit_truth_load_failed path={explicit_raw_path} error=not_found")
                continue
            try:
                snapshot_ts = parse_snapshot_timestamp(snapshot_path)
                explicit_delta = abs((ocr_dt_corrected - snapshot_ts).total_seconds())
                explicit_deltas[explicit_key] = explicit_delta
                if explicit_delta > time_window_sec:
                    explicit_invalid_reason = "explicit_out_of_window"
                    print(
                        f"[WARN] explicit_truth_skipped path={snapshot_path} "
                        f"delta={explicit_delta:.3f}s window={time_window_sec:.3f}s"
                    )
                    continue

                snapshot_data = load_snapshot_data(snapshot_path, snapshot_data_cache)
                selected_payload = snapshot_data.payload
                selected_map = snapshot_data.truth_map
                selected_snapshot_path = str(snapshot_path)
                selected_truth_dt = snapshot_ts
                selected_source = f"explicit:{explicit_key}"
                selected_dt_source = "explicit_path"
                if explicit_key == "cache_snapshot_path_before":
                    snapshot_before_used += 1
                else:
                    snapshot_used += 1
                explicit_selected = True
                break
            except Exception as exc:  # noqa: BLE001
                snapshot_load_error += 1
                print(f"[WARN] explicit_truth_load_failed path={snapshot_path} error={exc}")

        if not explicit_selected and cache_candidates and cache_candidate_timestamps:
            window_candidates: list[SnapshotCandidate] = []
            if ocr_dt_corrected is not None:
                ocr_ts = ocr_dt_corrected.timestamp()
                end_idx = bisect.bisect_right(cache_candidate_timestamps, ocr_ts)
                start_idx = end_idx - 1

                while start_idx >= 0:
                    candidate = cache_candidates[start_idx]
                    delta = ocr_ts - cache_candidate_timestamps[start_idx]
                    if delta > lookback_sec:
                        break
                    window_candidates.append(candidate)
                    if max_candidates > 0 and len(window_candidates) >= max_candidates:
                        break
                    start_idx -= 1

                if not window_candidates and fallback_to_nearest_past and end_idx > 0:
                    nearest_fallback_used += 1
                    window_candidates = [cache_candidates[end_idx - 1]]

            if max_candidates > 0:
                window_candidates = window_candidates[:max_candidates]
            selected_candidate_count = len(window_candidates)

            for candidate in window_candidates:
                try:
                    snapshot_data = load_snapshot_data(candidate.path, snapshot_data_cache)
                    truth_payload = snapshot_data.payload
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] failed to load snapshot {candidate.path}: {exc}")
                    continue

                truth_map = snapshot_data.truth_map
                per_item, score, _, _, _ = evaluate_record(
                    rec=rec,
                    truth_map=truth_map,
                    truth_payload=truth_payload,
                    ocr_dt_corrected=ocr_dt_corrected,
                    truth_dt=candidate.timestamp,
                    include_mismatch=False,
                )
                candidate_delta = (ocr_dt_corrected - candidate.timestamp).total_seconds() if ocr_dt_corrected else float("inf")
                if (
                    score["matched"] > chosen_score["matched"]
                    or (
                        score["matched"] == chosen_score["matched"]
                        and score["mean_abs_error"] < chosen_score["mean_abs_error"]
                    )
                    or (
                        score["matched"] == chosen_score["matched"]
                        and score["mean_abs_error"] == chosen_score["mean_abs_error"]
                        and candidate_delta < chosen_score["delta"]
                    )
                ):
                    chosen_per_item = per_item
                    chosen_score = {
                        "matched": score["matched"],
                        "mean_abs_error": score["mean_abs_error"],
                        "delta": candidate_delta,
                    }
                    selected_payload = truth_payload
                    selected_map = truth_map
                    selected_snapshot_path = str(candidate.path)
                    selected_truth_dt = candidate.timestamp
                    selected_source = "cache_dir_search"
                    selected_dt_source = "cache_search"

            if selected_snapshot_path is not None:
                selected_source = "cache_dir_search"
                selected_dt_source = "cache_search"

        if not explicit_selected and selected_snapshot_path is None:
            if monitor_cache_payload is not None and monitor_truth is not None:
                explicit_used += 1
                selected_payload = monitor_cache_payload
                selected_map = monitor_truth
                selected_snapshot_path = None
                selected_truth_dt = None
                selected_source = "monitor_cache_fallback"
                selected_dt_source = "monitor_cache"
            else:
                skipped_truth_unavailable += 1
                skip_reason = "truth_unavailable"
                skip_reason_counts[skip_reason] += 1
                print(
                    f"[WARN] skip_record reason={skip_reason} "
                    f"timestamp={rec.get('timestamp')} image_path={rec.get('image_path')}"
                )
                continue

        per_item, score, record_mismatches, dt_sec, dt_skip_reason = evaluate_record(
            rec=rec,
            truth_map=selected_map,
            truth_payload=selected_payload,
            ocr_dt_corrected=ocr_dt_corrected,
            truth_dt=selected_truth_dt,
            include_mismatch=True,
        )
        for mismatch_item in record_mismatches:
            mismatch_item["truth_snapshot_used"] = selected_snapshot_path

        chosen_per_item = per_item
        if score_fields is None:
            total += int(score["total"])
        else:
            total += sum(1 for row in chosen_per_item if row["vital"] in score_fields)
        matched += int(score["matched"])
        for row in chosen_per_item:
            if score_fields is not None and row["vital"] not in score_fields:
                continue
            if row["parsed_ocr_value"] is None:
                missing += 1
                if row["invalid_reason"]:
                    invalid += 1
            elif row["abs_error"] is not None:
                abs_errors.append(row["abs_error"])

        if selected_dt_source == "explicit_path" and dt_sec is None:
            dt_sec = 0.0
            dt_skip_reason = None

        if dt_sec is not None:
            dt_seconds.append(abs(dt_sec))
        else:
            dt_skipped += 1
            dt_skip_reason_counts[dt_skip_reason or "unknown"] += 1

        mismatches.extend(record_mismatches)
        details.append(
            {
                "timestamp": rec.get("timestamp"),
                "capture_timestamp": rec.get("capture_timestamp", rec.get("timestamp")),
                "record_time_source": record_time_source,
                "record_time": record_time.isoformat(),
                "validated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "truth_source": selected_source,
                "truth_snapshot_used": selected_snapshot_path,
                "delta_t_seconds": dt_sec,
                "dt_source": selected_dt_source,
                "candidate_count": selected_candidate_count,
                "explicit_before_delta": explicit_deltas.get("cache_snapshot_path_before"),
                "explicit_after_delta": explicit_deltas.get("cache_snapshot_path"),
                "explicit_invalid_reason": explicit_invalid_reason,
                "comparisons": chosen_per_item,
            }
        )

    metrics: dict[str, float] = {
        "total": float(total),
        "match_rate": (matched / total) if total else 0.0,
        "mean_abs_error": (sum(abs_errors) / len(abs_errors)) if abs_errors else 0.0,
        "median_abs_error": statistics.median(abs_errors) if abs_errors else 0.0,
        "missing_rate": (missing / total) if total else 0.0,
        "invalid_rate": (invalid / total) if total else 0.0,
        "snapshot_before_used": float(snapshot_before_used),
        "snapshot_used": float(snapshot_used),
        "explicit_used": float(explicit_used),
        "snapshot_missing": float(snapshot_missing),
        "snapshot_load_error": float(snapshot_load_error),
        "skipped_truth_unavailable": float(skipped_truth_unavailable),
        "skip_reason_counts": dict(skip_reason_counts),
        "nearest_fallback_used": float(nearest_fallback_used),
        "dt_mean_seconds": (sum(dt_seconds) / len(dt_seconds)) if dt_seconds else 0.0,
        "dt_median_seconds": statistics.median(dt_seconds) if dt_seconds else 0.0,
        "dt_p90_seconds": percentile(dt_seconds, 0.9) if dt_seconds else 0.0,
        "dt_evaluated": float(len(dt_seconds)),
        "dt_skipped": float(dt_skipped),
        "dt_skip_reason_counts": dict(dt_skip_reason_counts),
    }

    max_mismatches = max(dump_mismatch, debug_mismatches)
    if max_mismatches > 0:
        mismatches.sort(key=lambda item: item["_sort_error"], reverse=True)
        mismatches = mismatches[:max_mismatches]
    else:
        mismatches = []

    for item in mismatches:
        item.pop("_sort_error", None)

    if mismatch_details_enabled:
        detail_output_path = ocr_results_path.parent / "mismatch_details.jsonl"
        write_jsonl(detail_output_path, mismatch_details_rows)
        print(f"[INFO] mismatch_details_saved={len(mismatch_details_rows)} output={detail_output_path}")

    return details, metrics, mismatches


def run_self_check(records: list[dict[str, Any]], rows: list[dict[str, Any]], self_check: int) -> None:
    if not records:
        print("[SELF_CHECK] skipped reason=no_records")
        return

    first = records[0]
    capture_dt = parse_iso8601(first.get("capture_timestamp") or first.get("timestamp"))
    ocr_dt = parse_iso8601(first.get("ocr_timestamp"))
    frame_elapsed_ms = safe_float(first.get("frame_elapsed_ms"))
    record_time, record_time_source = get_record_time(first)

    if capture_dt is None or ocr_dt is None or frame_elapsed_ms is None:
        missing_fields: list[str] = []
        if capture_dt is None:
            missing_fields.append("capture_timestamp/timestamp")
        if ocr_dt is None:
            missing_fields.append("ocr_timestamp")
        if frame_elapsed_ms is None:
            missing_fields.append("frame_elapsed_ms")
        print(
            "[SELF_CHECK] skipped reason=missing_fields "
            f"missing={','.join(missing_fields)} "
            f"record_time_source={record_time_source} record_time={record_time.isoformat() if record_time else None} "
            f"capture_timestamp={first.get('capture_timestamp', first.get('timestamp'))} "
            f"ocr_timestamp={first.get('ocr_timestamp')} frame_elapsed_ms={first.get('frame_elapsed_ms')}"
        )
    else:
        diff_ms = (ocr_dt - capture_dt).total_seconds() * 1000.0
        abs_gap_ms = abs(diff_ms - frame_elapsed_ms)
        status = "pass" if abs_gap_ms <= max(5000.0, frame_elapsed_ms * 0.5) else "warn"
        print(
            f"[SELF_CHECK] capture_vs_ocr status={status} "
            f"diff_ms={diff_ms:.1f} frame_elapsed_ms={frame_elapsed_ms:.1f} gap_ms={abs_gap_ms:.1f}"
        )

    if rows:
        row = rows[0]
        delta_t = safe_float(row.get("delta_t_seconds"))
        truth_source = row.get("truth_source")
        if delta_t is None:
            print(f"[SELF_CHECK] validator_delta skipped reason=delta_none truth_source={truth_source}")
        else:
            status = "pass" if abs(delta_t) <= 3.0 else "warn"
            reason = row.get("explicit_invalid_reason")
            if reason == "explicit_out_of_window":
                status = "warn"
            print(
                f"[SELF_CHECK] validator_delta status={status} "
                f"delta_t_seconds={delta_t:.3f} truth_source={truth_source}"
                f" reason={reason if reason else 'none'}"
            )

    if self_check > 0:
        print("[SELF_CHECK] recent_records")
        for row in rows[-self_check:]:
            print(
                "[SELF_CHECK] "
                f"timestamp={row.get('timestamp')} "
                f"record_time_source={row.get('record_time_source')} "
                f"record_time={row.get('record_time')} "
                f"explicit_before_delta={row.get('explicit_before_delta')} "
                f"explicit_after_delta={row.get('explicit_after_delta')} "
                f"chosen_truth_source={row.get('truth_source')}"
            )



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
    parser.add_argument(
        "--monitor-cache",
        default=None,
        help="Path to monitor_cache.json used only as final fallback truth source",
    )
    parser.add_argument("--validator-config", default="validator_config.json", help="Path to validator config JSON")
    parser.add_argument("--last", type=int, default=50, help="Use latest N OCR records")
    parser.add_argument("--cache-dir", default=None, help="Directory containing monitor cache snapshots")
    parser.add_argument("--time-window-sec", type=float, default=10.0, help="Legacy search window in seconds")
    parser.add_argument("--lookback-sec", type=float, default=12.0, help="Search lookback window in seconds (past only)")
    parser.add_argument("--max-candidates", type=int, default=40, help="Max candidate snapshots to evaluate per OCR record")
    parser.add_argument("--lag-sec", type=float, default=0.0, help="Timestamp correction applied to OCR timestamp")
    parser.add_argument(
        "--fallback-to-nearest-past",
        action="store_true",
        default=True,
        help="If no snapshot exists within lookback window, evaluate nearest past snapshot",
    )
    parser.add_argument(
        "--no-fallback-to-nearest-past",
        action="store_false",
        dest="fallback_to_nearest_past",
        help="Disable nearest-past fallback when lookback search returns empty",
    )
    parser.add_argument(
        "--auto-lag",
        action="store_true",
        help="Brute-force lag from -10.0 to +10.0 sec (step 0.5) and choose best by score",
    )
    parser.add_argument("--select-by", default="score", choices=["score"], help="Candidate selection strategy")
    parser.add_argument(
        "--dump-mismatch",
        type=int,
        nargs="?",
        const=-1,
        default=0,
        help="Enable mismatch details dump JSONL; optional value prints top-N mismatch rows",
    )
    parser.add_argument(
        "--debug-mismatches",
        type=int,
        default=0,
        help="Print top-N mismatching fields with OCR vs truth and selected truth snapshot",
    )
    parser.add_argument(
        "--score-fields",
        default=None,
        help="Comma-separated fields used for scoring metrics (e.g. HR,ART_S,ART_D,ART_M,CVP_M,RAP_M,SpO2)",
    )
    parser.add_argument(
        "--self-check",
        type=int,
        default=0,
        help="Print record_time and explicit deltas for the last N validated records",
    )
    args = parser.parse_args()

    ocr_path = Path(args.ocr_results)
    cache_path = Path(args.monitor_cache) if args.monitor_cache else None
    config_path = Path(args.validator_config)

    if not ocr_path.exists():
        raise FileNotFoundError(f"OCR file not found: {ocr_path}")
    if cache_path is not None and not cache_path.exists():
        raise FileNotFoundError(f"monitor_cache file not found: {cache_path}")

    cfg = ensure_validator_config(config_path)
    tolerances = cfg.get("tolerances", {}) if isinstance(cfg, dict) else {}

    records = load_recent_records(ocr_path, args.last)

    monitor_cache_payload: dict[str, Any] | None = None
    monitor_truth: dict[str, dict[str, Any]] | None = None
    if cache_path is not None:
        monitor_cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(monitor_cache_payload, dict):
            raise ValueError(f"monitor_cache file is not a JSON object: {cache_path}")
        monitor_truth = extract_truth_map(monitor_cache_payload)

    cache_candidates: list[SnapshotCandidate] | None = None
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        cache_candidates = load_snapshot_candidates(cache_dir)
        print(f"[INFO] cache_candidates_loaded={len(cache_candidates)} from={cache_dir}")
    cache_candidate_timestamps = (
        build_snapshot_timestamps(cache_candidates) if cache_candidates else None
    )

    lookback_sec = args.lookback_sec
    if args.time_window_sec != 10.0 and args.lookback_sec == 12.0:
        lookback_sec = args.time_window_sec

    score_fields = parse_score_fields(args.score_fields)
    if score_fields is None:
        print("[INFO] score_fields=ALL")
    else:
        print(f"[INFO] score_fields_selected={sorted(score_fields)}")

    mismatch_details_enabled = args.dump_mismatch != 0
    if mismatch_details_enabled:
        print("[INFO] mismatch_detail_logging=enabled")

    selected_lag = args.lag_sec
    if args.auto_lag and cache_candidates:
        lag_values = [x * 0.5 for x in range(-20, 21)]
        best_lag = selected_lag
        best_score = (-1.0, float("inf"), float("inf"))
        for lag in lag_values:
            _, lag_metrics, _ = validate_records(
                ocr_records=records,
                monitor_cache_payload=monitor_cache_payload,
                monitor_truth=monitor_truth,
                ocr_results_path=ocr_path,
                tolerances=tolerances,
                dump_mismatch=0,
                debug_mismatches=0,
                cache_candidates=cache_candidates,
                cache_candidate_timestamps=cache_candidate_timestamps,
                lookback_sec=lookback_sec,
                max_candidates=args.max_candidates,
                lag_sec=lag,
                fallback_to_nearest_past=args.fallback_to_nearest_past,
                score_fields=score_fields,
                mismatch_details_enabled=False,
                time_window_sec=args.time_window_sec,
            )
            score = (lag_metrics["match_rate"], lag_metrics["mean_abs_error"], lag_metrics["dt_mean_seconds"])
            if score[0] > best_score[0] or (score[0] == best_score[0] and score[1] < best_score[1]):
                best_score = score
                best_lag = lag
        selected_lag = best_lag
        print(f"[INFO] auto_lag_selected={selected_lag:.1f}")

    rows, metrics, mismatches = validate_records(
        ocr_records=records,
        monitor_cache_payload=monitor_cache_payload,
        monitor_truth=monitor_truth,
        ocr_results_path=ocr_path,
        tolerances=tolerances,
        dump_mismatch=args.dump_mismatch,
        debug_mismatches=args.debug_mismatches,
        cache_candidates=cache_candidates,
        cache_candidate_timestamps=cache_candidate_timestamps,
        lookback_sec=lookback_sec,
        max_candidates=args.max_candidates,
        lag_sec=selected_lag,
        fallback_to_nearest_past=args.fallback_to_nearest_past,
        score_fields=score_fields,
        mismatch_details_enabled=mismatch_details_enabled,
        time_window_sec=args.time_window_sec,
    )

    run_self_check(records, rows, args.self_check)

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
        f"snapshot_before_used={int(metrics['snapshot_before_used'])} "
        f"snapshot_used={int(metrics['snapshot_used'])} "
        f"nearest_fallback_used={int(metrics['nearest_fallback_used'])} "
        f"explicit_used={int(metrics['explicit_used'])} "
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
    print(f"[INFO] dt_skip_reason_counts={metrics.get('dt_skip_reason_counts', {})}")
    print(f"[INFO] record_skip_reason_counts={metrics.get('skip_reason_counts', {})}")

    if args.dump_mismatch > 0:
        dump_rows = mismatches[: args.dump_mismatch]
        print(f"[INFO] dump_mismatch_top={len(dump_rows)}")
        for idx, item in enumerate(dump_rows, start=1):
            print(
                f"[MISMATCH {idx}] "
                f"bed={item['bed']} vital={item['vital']} "
                f"ocr_value={item['ocr_value']} truth_value={item['truth_value']} "
                f"abs_error={item['abs_error']} confidence={item['confidence']} "
                f"image_path={item['image_path']} cache_snapshot_path={item['cache_snapshot_path']}"
            )

    if args.debug_mismatches > 0:
        debug_rows = mismatches[: args.debug_mismatches]
        print(f"[INFO] debug_mismatches_top={len(debug_rows)}")
        for idx, item in enumerate(debug_rows, start=1):
            print(
                f"[DEBUG_MISMATCH {idx}] "
                f"bed={item['bed']} vital={item['vital']} "
                f"ocr_value={item['ocr_value']} truth_value={item['truth_value']} "
                f"truth_snapshot_used={item.get('truth_snapshot_used')} "
                f"abs_error={item['abs_error']} image_path={item['image_path']}"
            )


if __name__ == "__main__":
    main()
