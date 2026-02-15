#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import easyocr
import mss
import numpy as np

try:
    import pygetwindow as gw
except Exception:  # pragma: no cover
    gw = None

BED_IDS = [f"BED0{i}" for i in range(1, 7)]
VITAL_ORDER = [
    "HR",
    "ART_S",
    "ART_D",
    "ART_M",
    "CVP_M",
    "RAP_M",
    "SpO2",
    "TSKIN",
    "TRECT",
    "rRESP",
    "EtCO2",
    "RR",
    "VTe",
    "VTi",
    "Ppeak",
    "PEEP",
    "O2conc",
    "NO",
    "BSR1",
    "BSR2",
]

NUMERIC_RE = re.compile(r"^-?\d+(\.\d+)?$")
ALLOWLIST = "0123456789.-"


DEFAULT_CONFIG = {
    "window_title": "HL7 Bed Monitor",
    "header_ratio": 0.04,
    "value_roi": {
        "left_ratio": 0.42,
        "right_ratio": 0.03,
        "top_ratio": 0.10,
        "bottom_ratio": 0.08,
    },
    "preprocess": {
        "grayscale": True,
        "threshold": False,
        "threshold_value": 160,
        "scale": 1.5,
    },
    "fallback_capture": {
        "enabled": True,
        "region": None,
    },
}


@dataclass
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int


@dataclass
class CellROI:
    bed: str
    vital: str
    bbox: tuple[int, int, int, int]


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def ensure_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
        return dict(DEFAULT_CONFIG)
    with path.open("r", encoding="utf-8") as fh:
        loaded = json.load(fh)
    config = dict(DEFAULT_CONFIG)
    for k, v in loaded.items():
        if isinstance(v, dict) and isinstance(config.get(k), dict):
            merged = dict(config[k])
            merged.update(v)
            config[k] = merged
        else:
            config[k] = v
    return config


def find_window_region(title: str) -> CaptureRegion | None:
    if gw is None:
        return None
    try:
        windows = gw.getWindowsWithTitle(title)
    except Exception:
        return None
    if not windows:
        return None
    win = windows[0]
    if win.width <= 0 or win.height <= 0:
        return None
    return CaptureRegion(left=int(win.left), top=int(win.top), width=int(win.width), height=int(win.height))


def choose_region(sct: mss.mss, title: str, monitor_index: int) -> CaptureRegion:
    win_region = find_window_region(title)
    if win_region:
        return win_region

    monitor_count = len(sct.monitors) - 1
    selected_index = monitor_index
    if selected_index < 1 or selected_index > monitor_count:
        print(
            f"[WARN] invalid --monitor-index {monitor_index}; using monitor 1 instead",
            file=sys.stderr,
        )
        selected_index = 1

    monitor = sct.monitors[selected_index]
    return CaptureRegion(
        left=int(monitor["left"]),
        top=int(monitor["top"]),
        width=int(monitor["width"]),
        height=int(monitor["height"]),
    )


def grab_frame(sct: mss.mss, region: CaptureRegion) -> np.ndarray:
    raw = sct.grab({"left": region.left, "top": region.top, "width": region.width, "height": region.height})
    arr = np.array(raw)
    return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)


def build_rois(frame_w: int, frame_h: int, config: dict[str, Any]) -> list[CellROI]:
    header_ratio = float(config.get("header_ratio", 0.04))
    header_h = max(int(frame_h * header_ratio), 0)
    body_top = min(header_h, frame_h - 1)
    body_h = max(frame_h - body_top, 1)

    value_cfg = config.get("value_roi", {})
    left_ratio = float(value_cfg.get("left_ratio", 0.42))
    right_ratio = float(value_cfg.get("right_ratio", 0.03))
    top_ratio = float(value_cfg.get("top_ratio", 0.10))
    bottom_ratio = float(value_cfg.get("bottom_ratio", 0.08))

    rois: list[CellROI] = []
    for bed_idx, bed in enumerate(BED_IDS):
        bed_row = bed_idx // 2
        bed_col = bed_idx % 2

        bx1 = int((bed_col * frame_w) / 2)
        bx2 = int(((bed_col + 1) * frame_w) / 2)
        by1 = body_top + int((bed_row * body_h) / 3)
        by2 = body_top + int(((bed_row + 1) * body_h) / 3)

        bed_w = max(bx2 - bx1, 1)
        bed_h = max(by2 - by1, 1)

        for cell_row in range(5):
            for cell_col in range(4):
                vital = VITAL_ORDER[cell_row * 4 + cell_col]
                cx1 = bx1 + int((cell_col * bed_w) / 4)
                cx2 = bx1 + int(((cell_col + 1) * bed_w) / 4)
                cy1 = by1 + int((cell_row * bed_h) / 5)
                cy2 = by1 + int(((cell_row + 1) * bed_h) / 5)

                cw = max(cx2 - cx1, 1)
                ch = max(cy2 - cy1, 1)
                vx1 = cx1 + int(cw * left_ratio)
                vx2 = cx2 - int(cw * right_ratio)
                vy1 = cy1 + int(ch * top_ratio)
                vy2 = cy2 - int(ch * bottom_ratio)

                vx1 = min(max(vx1, cx1), cx2 - 1)
                vx2 = max(min(vx2, cx2), vx1 + 1)
                vy1 = min(max(vy1, cy1), cy2 - 1)
                vy2 = max(min(vy2, cy2), vy1 + 1)

                rois.append(CellROI(bed=bed, vital=vital, bbox=(vx1, vy1, vx2, vy2)))
    return rois


def preprocess_image(img: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    pp = config.get("preprocess", {})
    out: np.ndarray = img
    if pp.get("grayscale", True):
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    if pp.get("threshold", False):
        thresh = int(pp.get("threshold_value", 160))
        _, out = cv2.threshold(out, thresh, 255, cv2.THRESH_BINARY)
    scale = float(pp.get("scale", 1.5))
    if scale > 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return out


def parse_value(text: str) -> float | None:
    normalized = text.strip()
    if not normalized:
        return None
    if normalized.count(".") > 1:
        return None
    if NUMERIC_RE.match(normalized) is None:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def run_ocr(reader: easyocr.Reader, roi_image: np.ndarray) -> tuple[str, float]:
    results = reader.readtext(roi_image, detail=1, paragraph=False, allowlist=ALLOWLIST)
    if not results:
        return "", 0.0

    best_text = ""
    best_conf = -1.0
    for _, text, conf in results:
        filtered = "".join(ch for ch in str(text) if ch in ALLOWLIST)
        if conf > best_conf and filtered:
            best_text = filtered
            best_conf = float(conf)
    if best_conf < 0:
        return "", 0.0
    return best_text, best_conf


def draw_debug_rois(frame: np.ndarray, rois: list[CellROI], output_path: Path) -> None:
    debug_img = frame.copy()
    for roi in rois:
        x1, y1, x2, y2 = roi.bbox
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), debug_img)


def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture monitor.py screen and run OCR into JSONL")
    parser.add_argument("--cache", default="monitor_cache.json")
    parser.add_argument("--outdir", default="dataset")
    parser.add_argument("--interval-ms", type=int, default=1000)
    parser.add_argument("--fullscreen", type=parse_bool, default=True)
    parser.add_argument("--save-images", type=parse_bool, default=True)
    parser.add_argument("--debug-roi", type=parse_bool, default=False)
    parser.add_argument("--config", default="ocr_capture_config.json")
    parser.add_argument("--monitor-script", default="monitor.py")
    parser.add_argument("--monitor-index", type=int, default=2)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    images_dir = outdir / "images"
    debug_dir = outdir / "debug"
    jsonl_path = outdir / "ocr_results.jsonl"
    outdir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(args.config)
    config = ensure_config(config_path)

    monitor_cmd = [
        sys.executable,
        args.monitor_script,
        "--cache",
        args.cache,
        "--fullscreen",
        str(args.fullscreen).lower(),
    ]
    monitor_proc = subprocess.Popen(monitor_cmd)

    stop = False

    def _stop_handler(signum, frame):
        del signum, frame
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop_handler)
    signal.signal(signal.SIGTERM, _stop_handler)

    print(f"[INFO] monitor started pid={monitor_proc.pid}")
    reader = easyocr.Reader(["en"], gpu=False)

    try:
        with mss.mss() as sct:
            for idx, monitor in enumerate(sct.monitors):
                print(
                    "[INFO] monitor "
                    f"index={idx} width={monitor['width']} height={monitor['height']} "
                    f"left={monitor['left']} top={monitor['top']}"
                )
            time.sleep(1.5)
            while not stop:
                loop_start = time.time()
                try:
                    region = choose_region(
                        sct,
                        str(config.get("window_title", "HL7 Bed Monitor")),
                        args.monitor_index,
                    )
                    frame = grab_frame(sct, region)
                    rois = build_rois(frame.shape[1], frame.shape[0], config)

                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    image_rel = f"images/{stamp}.png"
                    if args.save_images:
                        image_path = outdir / image_rel
                        cv2.imwrite(str(image_path), frame)
                    else:
                        image_rel = ""

                    if args.debug_roi:
                        draw_debug_rois(frame, rois, debug_dir / f"{stamp}_roi.png")

                    beds: dict[str, dict[str, Any]] = {bed: {} for bed in BED_IDS}
                    for roi in rois:
                        x1, y1, x2, y2 = roi.bbox
                        crop = frame[y1:y2, x1:x2]
                        processed = preprocess_image(crop, config)
                        text, conf = run_ocr(reader, processed)
                        beds[roi.bed][roi.vital] = {
                            "text": text,
                            "value": parse_value(text),
                            "confidence": float(conf),
                        }

                    record = {
                        "timestamp": iso_now(),
                        "image_path": f"{args.outdir}/{image_rel}" if image_rel else None,
                        "source": {"app": "monitor.py", "cache": args.cache},
                        "beds": beds,
                    }
                    with jsonl_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fh.flush()

                    elapsed = (time.time() - loop_start) * 1000
                    sleep_ms = max(args.interval_ms - elapsed, 0)
                    time.sleep(sleep_ms / 1000.0)
                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] capture loop error: {exc}", file=sys.stderr)
                    time.sleep(max(args.interval_ms / 1000.0, 0.2))
    finally:
        if monitor_proc.poll() is None:
            monitor_proc.terminate()
            try:
                monitor_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor_proc.kill()


if __name__ == "__main__":
    main()
