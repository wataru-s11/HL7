#!/usr/bin/env python3
"""Compare EasyOCR and Tesseract results against snapshot truth on the same frame."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any

DEFAULT_TOLERANCES: dict[str, float] = {
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

NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def parse_numeric_from_text(raw_text: Any) -> float | None:
    if raw_text is None:
        return None
    text = str(raw_text).strip()
    if not text:
        return None
    compact = text.replace(",", "")
    match = NUMERIC_RE.search(compact)
    if not match:
        return None
    return safe_float(match.group(0))


def parse_candidate_value(candidate: dict[str, Any] | None) -> tuple[float | None, str | None, float | None, str]:
    if not isinstance(candidate, dict):
        return None, None, None, "missing"

    text = candidate.get("text")
    value = safe_float(candidate.get("value"))
    if value is None:
        value = parse_numeric_from_text(text)

    confidence = safe_float(candidate.get("confidence"))
    if confidence is None:
        confidence = safe_float(candidate.get("conf"))

    if value is None:
        if text is None or str(text).strip() == "":
            return None, text if text is None else str(text), confidence, "missing"
        return None, str(text), confidence, "invalid"
    return value, text if text is None else str(text), confidence, "valid"


def normalize_engine_name(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in {"easy", "easyocr"}:
        return "easy"
    if text in {"tess", "tesseract", "tesseractocr"}:
        return "tess"
    return None


def as_candidate_dict(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        return dict(payload)
    if payload is None:
        return None
    if isinstance(payload, (int, float, str)):
        return {"value": payload, "text": str(payload)}
    return None


def extract_from_candidates_section(section: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    easy: dict[str, Any] | None = None
    tess: dict[str, Any] | None = None

    if isinstance(section, dict):
        easy = as_candidate_dict(section.get("easy") or section.get("easyocr"))
        tess = as_candidate_dict(section.get("tess") or section.get("tesseract"))
        return easy, tess

    if not isinstance(section, list):
        return None, None

    scored: dict[str, tuple[float, dict[str, Any]]] = {}
    for item in section:
        if not isinstance(item, dict):
            continue
        engine = normalize_engine_name(item.get("engine") or item.get("selected_engine") or item.get("name"))
        if engine is None:
            continue
        conf = safe_float(item.get("confidence"))
        if conf is None:
            conf = safe_float(item.get("conf"))
        score = conf if conf is not None else -1.0
        current = scored.get(engine)
        if current is None or score > current[0]:
            scored[engine] = (score, item)

    if "easy" in scored:
        easy = dict(scored["easy"][1])
    if "tess" in scored:
        tess = dict(scored["tess"][1])
    return easy, tess


def extract_engine_candidates(field_payload: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not isinstance(field_payload, dict):
        cand = as_candidate_dict(field_payload)
        return cand, None

    easy = as_candidate_dict(field_payload.get("easy") or field_payload.get("easyocr"))
    tess = as_candidate_dict(field_payload.get("tess") or field_payload.get("tesseract"))

    if easy is None or tess is None:
        c_easy, c_tess = extract_from_candidates_section(field_payload.get("candidates"))
        if easy is None:
            easy = c_easy
        if tess is None:
            tess = c_tess

    if easy is None and tess is None:
        selected_engine = normalize_engine_name(field_payload.get("selected_engine"))
        flat_candidate = as_candidate_dict(field_payload)
        if selected_engine == "easy":
            easy = flat_candidate
        elif selected_engine == "tess":
            tess = flat_candidate
        elif flat_candidate is not None:
            easy = flat_candidate

    return easy, tess


def extract_truth_map(snapshot_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    beds = snapshot_payload.get("beds")
    if not isinstance(beds, dict):
        return out

    for bed, bed_payload in beds.items():
        if not isinstance(bed_payload, dict):
            continue
        bed_truth: dict[str, Any] = {}

        vitals = bed_payload.get("vitals")
        if isinstance(vitals, dict):
            for field, payload in vitals.items():
                if isinstance(payload, dict):
                    bed_truth[field] = payload.get("value")
                else:
                    bed_truth[field] = payload

        for field, payload in bed_payload.items():
            if field in {"vitals", "values"}:
                continue
            if field in bed_truth:
                continue
            if isinstance(payload, dict) and "value" in payload:
                bed_truth[field] = payload.get("value")

        if bed_truth:
            out[str(bed)] = bed_truth
    return out


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, start=1):
            row = line.strip()
            if not row:
                continue
            try:
                payload = json.loads(row)
            except json.JSONDecodeError as exc:
                print(f"[WARN] JSONL parse failed line={idx}: {exc}")
                continue
            if not isinstance(payload, dict):
                print(f"[WARN] JSONL non-object line={idx}, skipped")
                continue
            yield idx, payload


def resolve_path(raw_path: Any, ocr_results_path: Path) -> Path | None:
    if raw_path is None:
        return None
    raw = str(raw_path).strip()
    if not raw:
        return None

    raw_path_obj = Path(raw)
    if raw_path_obj.is_absolute() and raw_path_obj.exists():
        return raw_path_obj

    normalized_rel = raw.replace("\\", "/")
    win_rel = Path(*PureWindowsPath(raw).parts)
    rel_candidates = [Path(normalized_rel), win_rel, raw_path_obj]

    bases = [Path.cwd(), ocr_results_path.parent]
    for rel in rel_candidates:
        if rel.is_absolute() and rel.exists():
            return rel
        for base in bases:
            cand = base / rel
            if cand.exists():
                return cand
    return None


def load_tolerances(arg_value: str | None) -> dict[str, float]:
    tolerances = dict(DEFAULT_TOLERANCES)
    if not arg_value:
        return tolerances

    path = Path(arg_value)
    loaded: Any
    if path.exists() and path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
    else:
        loaded = json.loads(arg_value)

    if isinstance(loaded, dict) and isinstance(loaded.get("tolerances"), dict):
        loaded = loaded["tolerances"]
    if not isinstance(loaded, dict):
        raise ValueError("tolerances must be JSON object or validator-config JSON")

    for k, v in loaded.items():
        fv = safe_float(v)
        if fv is not None:
            tolerances[str(k)] = fv
    return tolerances


def canonical_truth_text(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value}".rstrip("0").rstrip(".")


def update_digit_confusion(counter: Counter[tuple[str, str]], truth_value: float, pred_text: str | None):
    if pred_text is None:
        return
    truth_digits = "".join(ch for ch in canonical_truth_text(truth_value) if ch.isdigit())
    pred_digits = "".join(ch for ch in str(pred_text) if ch.isdigit())
    if not truth_digits and not pred_digits:
        return
    max_len = max(len(truth_digits), len(pred_digits))
    for idx in range(max_len):
        t = truth_digits[idx] if idx < len(truth_digits) else "_"
        p = pred_digits[idx] if idx < len(pred_digits) else "_"
        counter[(t, p)] += 1


def winner_from_errors(easy_err: float | None, tess_err: float | None, easy_status: str, tess_status: str) -> str:
    easy_ok = easy_err is not None and easy_status == "valid"
    tess_ok = tess_err is not None and tess_status == "valid"

    if not easy_ok and not tess_ok:
        return "none"
    if easy_ok and not tess_ok:
        return "easy"
    if tess_ok and not easy_ok:
        return "tess"
    if easy_err < tess_err:
        return "easy"
    if tess_err < easy_err:
        return "tess"
    return "tie"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-results", required=True, help="Path to OCR JSONL")
    parser.add_argument("--truth-mode", default="snapshot_before", choices=["snapshot_before"], help="Truth source mode")
    parser.add_argument("--out-dir", default="dataset/20260217/compare", help="Output directory")
    parser.add_argument("--tolerances", default=None, help="JSON file path or JSON object string")
    args = parser.parse_args()

    ocr_results_path = Path(args.ocr_results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tolerances = load_tolerances(args.tolerances)

    snapshot_cache: dict[str, dict[str, Any]] = {}
    winloss_rows: list[dict[str, Any]] = []

    field_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0,
            "easy_win": 0,
            "tess_win": 0,
            "tie": 0,
            "none": 0,
            "easy_abs_errors": [],
            "tess_abs_errors": [],
            "easy_match": 0,
            "tess_match": 0,
            "easy_missing": 0,
            "tess_missing": 0,
            "easy_invalid": 0,
            "tess_invalid": 0,
        }
    )

    overall = {
        "n": 0,
        "easy_win": 0,
        "tess_win": 0,
        "tie": 0,
        "both_bad": 0,
        "easy_abs_errors": [],
        "tess_abs_errors": [],
        "winner_abs_errors": [],
        "easy_match": 0,
        "tess_match": 0,
        "easy_missing": 0,
        "tess_missing": 0,
        "easy_invalid": 0,
        "tess_invalid": 0,
    }

    easy_confusion: Counter[tuple[str, str]] = Counter()
    tess_confusion: Counter[tuple[str, str]] = Counter()

    for line_no, rec in iter_jsonl(ocr_results_path):
        snapshot_path = resolve_path(rec.get("cache_snapshot_path_before"), ocr_results_path)
        if snapshot_path is None:
            print(f"[WARN] line={line_no} snapshot_before path not found, skipped")
            continue

        snapshot_key = str(snapshot_path)
        if snapshot_key not in snapshot_cache:
            try:
                payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("snapshot is not object")
                snapshot_cache[snapshot_key] = payload
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] line={line_no} snapshot load failed path={snapshot_path}: {exc}")
                continue

        truth_map = extract_truth_map(snapshot_cache[snapshot_key])
        beds_payload = rec.get("beds") if isinstance(rec.get("beds"), dict) else {}

        for bed, truth_fields in truth_map.items():
            if not isinstance(truth_fields, dict):
                continue
            bed_ocr = beds_payload.get(bed) if isinstance(beds_payload.get(bed), dict) else {}

            for field, truth_raw in truth_fields.items():
                truth_value = safe_float(truth_raw)
                if truth_value is None:
                    continue

                payload = bed_ocr.get(field)
                easy_candidate, tess_candidate = extract_engine_candidates(payload)

                easy_val, easy_text, _, easy_status = parse_candidate_value(easy_candidate)
                tess_val, tess_text, _, tess_status = parse_candidate_value(tess_candidate)

                easy_err = abs(easy_val - truth_value) if easy_val is not None else None
                tess_err = abs(tess_val - truth_value) if tess_val is not None else None

                tol = tolerances.get(field, 0.0)
                easy_match = easy_err is not None and easy_err <= tol
                tess_match = tess_err is not None and tess_err <= tol

                winner = winner_from_errors(easy_err, tess_err, easy_status, tess_status)

                row = {
                    "timestamp": rec.get("timestamp"),
                    "image_path": rec.get("image_path"),
                    "bed": bed,
                    "field": field,
                    "truth": truth_value,
                    "easy_value": easy_val,
                    "tess_value": tess_val,
                    "easy_abs_err": easy_err,
                    "tess_abs_err": tess_err,
                    "winner": winner,
                }
                winloss_rows.append(row)

                s = field_stats[field]
                s["n"] += 1
                overall["n"] += 1

                if easy_status == "missing":
                    s["easy_missing"] += 1
                    overall["easy_missing"] += 1
                elif easy_status == "invalid":
                    s["easy_invalid"] += 1
                    overall["easy_invalid"] += 1

                if tess_status == "missing":
                    s["tess_missing"] += 1
                    overall["tess_missing"] += 1
                elif tess_status == "invalid":
                    s["tess_invalid"] += 1
                    overall["tess_invalid"] += 1

                if easy_err is not None:
                    s["easy_abs_errors"].append(easy_err)
                    overall["easy_abs_errors"].append(easy_err)
                    update_digit_confusion(easy_confusion, truth_value, easy_text)
                if tess_err is not None:
                    s["tess_abs_errors"].append(tess_err)
                    overall["tess_abs_errors"].append(tess_err)
                    update_digit_confusion(tess_confusion, truth_value, tess_text)

                if easy_match:
                    s["easy_match"] += 1
                    overall["easy_match"] += 1
                if tess_match:
                    s["tess_match"] += 1
                    overall["tess_match"] += 1

                if winner == "easy":
                    s["easy_win"] += 1
                    overall["easy_win"] += 1
                    if easy_err is not None:
                        overall["winner_abs_errors"].append(easy_err)
                elif winner == "tess":
                    s["tess_win"] += 1
                    overall["tess_win"] += 1
                    if tess_err is not None:
                        overall["winner_abs_errors"].append(tess_err)
                elif winner == "tie":
                    s["tie"] += 1
                    overall["tie"] += 1
                    if easy_err is not None:
                        overall["winner_abs_errors"].append(easy_err)
                else:
                    s["none"] += 1
                    overall["both_bad"] += 1

    def mean(vals: list[float]) -> float | None:
        return (sum(vals) / len(vals)) if vals else None

    def median(vals: list[float]) -> float | None:
        return statistics.median(vals) if vals else None

    n = overall["n"]
    summary_overall = {
        "n": n,
        "match_rate": (overall["easy_match"] + overall["tess_match"]) / (2 * n) if n else None,
        "MAE": mean(overall["winner_abs_errors"]),
        "median_abs_error": median(overall["winner_abs_errors"]),
        "easy_win": overall["easy_win"],
        "tess_win": overall["tess_win"],
        "tie": overall["tie"],
        "both_bad": overall["both_bad"],
        "easy_MAE": mean(overall["easy_abs_errors"]),
        "tess_MAE": mean(overall["tess_abs_errors"]),
        "easy_match_rate": (overall["easy_match"] / n) if n else None,
        "tess_match_rate": (overall["tess_match"] / n) if n else None,
        "easy_missing_rate": (overall["easy_missing"] / n) if n else None,
        "tess_missing_rate": (overall["tess_missing"] / n) if n else None,
        "easy_invalid_rate": (overall["easy_invalid"] / n) if n else None,
        "tess_invalid_rate": (overall["tess_invalid"] / n) if n else None,
    }
    (out_dir / "summary_overall.json").write_text(json.dumps(summary_overall, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "summary_by_field.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "field",
                "n",
                "easy_win",
                "tess_win",
                "tie",
                "easy_MAE",
                "tess_MAE",
                "easy_match_rate",
                "tess_match_rate",
                "easy_missing_rate",
                "tess_missing_rate",
                "easy_invalid_rate",
                "tess_invalid_rate",
            ],
        )
        writer.writeheader()
        for field in sorted(field_stats):
            s = field_stats[field]
            n_field = s["n"]
            writer.writerow(
                {
                    "field": field,
                    "n": n_field,
                    "easy_win": s["easy_win"],
                    "tess_win": s["tess_win"],
                    "tie": s["tie"],
                    "easy_MAE": mean(s["easy_abs_errors"]),
                    "tess_MAE": mean(s["tess_abs_errors"]),
                    "easy_match_rate": (s["easy_match"] / n_field) if n_field else None,
                    "tess_match_rate": (s["tess_match"] / n_field) if n_field else None,
                    "easy_missing_rate": (s["easy_missing"] / n_field) if n_field else None,
                    "tess_missing_rate": (s["tess_missing"] / n_field) if n_field else None,
                    "easy_invalid_rate": (s["easy_invalid"] / n_field) if n_field else None,
                    "tess_invalid_rate": (s["tess_invalid"] / n_field) if n_field else None,
                }
            )

    with (out_dir / "winloss_table.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "timestamp",
                "image_path",
                "bed",
                "field",
                "truth",
                "easy_value",
                "tess_value",
                "easy_abs_err",
                "tess_abs_err",
                "winner",
            ],
        )
        writer.writeheader()
        writer.writerows(winloss_rows)

    def write_confusion(path: Path, confusion: Counter[tuple[str, str]]):
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["truth_digit", "pred_digit", "count"])
            writer.writeheader()
            for (truth_d, pred_d), count in sorted(confusion.items()):
                writer.writerow({"truth_digit": truth_d, "pred_digit": pred_d, "count": count})

    write_confusion(out_dir / "confusion_digits_easy.csv", easy_confusion)
    write_confusion(out_dir / "confusion_digits_tess.csv", tess_confusion)

    print(f"[INFO] wrote comparison artifacts to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
