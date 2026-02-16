#!/usr/bin/env python3
"""
OCR capture application.

Usage examples:
  # 1) One-time calibration (adjust ROI and save config)
  python ocr_capture_app.py --calibrate --config ocr_capture_config.json --monitor-index 2

  # 2) Periodic capture (10s interval)
  python ocr_capture_app.py --config ocr_capture_config.json --interval-ms 10000 --save-images true --debug-roi false
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
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
import torch

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

DEFAULT_CONFIG: dict[str, Any] = {
    "window_title": "HL7 Bed Monitor",
    "monitor_index": 2,
    "monitor_rect": None,
    "header_crop_px": 60,
    "bed_grid": {"cols": 2, "rows": 3},
    "cell_grid": {"cols": 4, "rows": 5},
    "value_roi": {
        "left_ratio": 0.45,
        "right_pad": 0.06,
        "top_pad": 0.10,
        "bottom_pad": 0.12,
    },
    # MOD: vital別ROI微調整設定を追加（px/ratioの両対応）
    "per_vital_roi_adjust": {},
    "preprocess": {"grayscale": True, "resize": 2.0, "threshold": False, "threshold_value": 160},
}


@dataclass
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def ensure_config(path: Path) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        for key, value in loaded.items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                merged = dict(config[key])
                merged.update(value)
                config[key] = merged
            else:
                config[key] = value
    else:
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # MOD: 旧config互換（right_ratio/top_ratio/bottom_ratio/scale）
    roi_cfg = config.get("value_roi", {})
    if "right_pad" not in roi_cfg and "right_ratio" in roi_cfg:
        roi_cfg["right_pad"] = roi_cfg.get("right_ratio")
    if "top_pad" not in roi_cfg and "top_ratio" in roi_cfg:
        roi_cfg["top_pad"] = roi_cfg.get("top_ratio")
    if "bottom_pad" not in roi_cfg and "bottom_ratio" in roi_cfg:
        roi_cfg["bottom_pad"] = roi_cfg.get("bottom_ratio")
    pp_cfg = config.get("preprocess", {})
    if "resize" not in pp_cfg and "scale" in pp_cfg:
        pp_cfg["resize"] = pp_cfg.get("scale")
    config["value_roi"] = roi_cfg
    config["preprocess"] = pp_cfg
    config.setdefault("per_vital_roi_adjust", {})
    return config


def save_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


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
    return CaptureRegion(int(win.left), int(win.top), int(win.width), int(win.height))


def log_monitors(sct: mss.mss) -> None:
    # MOD: 起動時に常に利用可能monitor情報を表示
    for idx, monitor in enumerate(sct.monitors):
        print(
            "[INFO] mss.monitor "
            f"index={idx} left={monitor['left']} top={monitor['top']} width={monitor['width']} height={monitor['height']}"
        )


def choose_capture_region(
    sct: mss.mss,
    config: dict[str, Any],
    monitor_index: int,
    mss_monitor_index: int | None,
) -> CaptureRegion:
    monitor_count = len(sct.monitors) - 1

    # MOD: --mss-monitor-index指定時はwindow_titleを無視してmonitor領域を直接使用
    if mss_monitor_index is not None:
        selected = mss_monitor_index
        if selected < 1 or selected > monitor_count:
            print(f"[WARN] invalid mss_monitor_index={selected}; fallback to monitor 1", file=sys.stderr)
            selected = 1
        monitor = sct.monitors[selected]
        print(f"[INFO] explicit mss monitor capture selected: index={selected} (window_title ignored)")
        return CaptureRegion(int(monitor["left"]), int(monitor["top"]), int(monitor["width"]), int(monitor["height"]))

    title = str(config.get("window_title", "HL7 Bed Monitor"))
    win_region = find_window_region(title)
    if win_region:
        print(f"[INFO] window-title capture selected: {title}")
        return win_region

    selected = monitor_index
    if selected < 1 or selected > monitor_count:
        print(f"[WARN] invalid monitor_index={monitor_index}; fallback to monitor 1", file=sys.stderr)
        selected = 1

    monitor = sct.monitors[selected]
    print(f"[INFO] monitor-index capture selected: index={selected}")
    return CaptureRegion(int(monitor["left"]), int(monitor["top"]), int(monitor["width"]), int(monitor["height"]))


def grab_frame(sct: mss.mss, region: CaptureRegion) -> np.ndarray:
    raw = sct.grab({"left": region.left, "top": region.top, "width": region.width, "height": region.height})
    return cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def get_monitor_rect(config: dict[str, Any], fallback: CaptureRegion) -> CaptureRegion:
    rect = config.get("monitor_rect")
    if not isinstance(rect, dict):
        return fallback
    try:
        left = int(rect["left"])
        top = int(rect["top"])
        width = int(rect["width"])
        height = int(rect["height"])
        if width > 0 and height > 0:
            return CaptureRegion(left, top, width, height)
    except Exception:
        pass
    return fallback


def preprocess_image(img: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    pp = config.get("preprocess", {})
    out: np.ndarray = img
    if pp.get("grayscale", True):
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    if pp.get("threshold", False):
        _, out = cv2.threshold(out, int(pp.get("threshold_value", 160)), 255, cv2.THRESH_BINARY)
    resize = float(pp.get("resize", 2.0))
    if resize > 1.0:
        out = cv2.resize(out, None, fx=resize, fy=resize, interpolation=cv2.INTER_CUBIC)
    return out


def run_ocr(reader: easyocr.Reader, roi_image: np.ndarray) -> tuple[str, float]:
    results = reader.readtext(roi_image, detail=1, paragraph=False, allowlist=ALLOWLIST)
    if not results:
        return "", 0.0
    best_text, best_conf = "", -1.0
    for _, text, conf in results:
        filtered = "".join(ch for ch in str(text) if ch in ALLOWLIST)
        if filtered and conf > best_conf:
            best_text, best_conf = filtered, float(conf)
    return ("", 0.0) if best_conf < 0 else (best_text, best_conf)


def parse_value(text: str) -> float | None:
    text = text.strip()
    if not text or NUMERIC_RE.match(text) is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _offset_from_adjust(adjust: dict[str, Any], key: str, cell_w: int, cell_h: int) -> int:
    # MOD: vital別オフセットのpx/ratio指定両対応
    mode = str(adjust.get("mode", "px")).strip().lower()
    raw = adjust.get(key, 0)
    try:
        raw_value = float(raw)
    except Exception:
        raw_value = 0.0

    ratio_key = f"{key}_ratio"
    if ratio_key in adjust:
        try:
            ratio_value = float(adjust.get(ratio_key, 0.0))
        except Exception:
            ratio_value = 0.0
        return int(round((cell_w if key in {"dx1", "dx2"} else cell_h) * ratio_value))

    if mode == "ratio":
        return int(round((cell_w if key in {"dx1", "dx2"} else cell_h) * raw_value))
    return int(round(raw_value))


def build_vital_rois(frame_shape: tuple[int, int], config: dict[str, Any]) -> dict[str, dict[str, tuple[int, int, int, int]]]:
    frame_h, frame_w = frame_shape
    header_crop_px = clamp(int(config.get("header_crop_px", 60)), 0, frame_h - 1)
    body_top = header_crop_px
    body_h = max(frame_h - body_top, 1)

    bed_grid = config.get("bed_grid", {})
    cell_grid = config.get("cell_grid", {})
    bed_cols, bed_rows = int(bed_grid.get("cols", 2)), int(bed_grid.get("rows", 3))
    cell_cols, cell_rows = int(cell_grid.get("cols", 4)), int(cell_grid.get("rows", 5))

    roi_cfg = config.get("value_roi", {})
    left_ratio = float(roi_cfg.get("left_ratio", 0.45))
    right_pad = float(roi_cfg.get("right_pad", 0.06))
    top_pad = float(roi_cfg.get("top_pad", 0.10))
    bottom_pad = float(roi_cfg.get("bottom_pad", 0.12))
    adjust_cfg = config.get("per_vital_roi_adjust", {})

    out: dict[str, dict[str, tuple[int, int, int, int]]] = {bed: {} for bed in BED_IDS}

    for bed_idx, bed in enumerate(BED_IDS):
        bed_row = bed_idx // bed_cols
        bed_col = bed_idx % bed_cols

        bx1 = int((bed_col * frame_w) / bed_cols)
        bx2 = int(((bed_col + 1) * frame_w) / bed_cols)
        by1 = body_top + int((bed_row * body_h) / bed_rows)
        by2 = body_top + int(((bed_row + 1) * body_h) / bed_rows)

        bed_w = max(bx2 - bx1, 1)
        bed_h = max(by2 - by1, 1)

        for idx, vital in enumerate(VITAL_ORDER):
            c_row = idx // cell_cols
            c_col = idx % cell_cols

            cx1 = bx1 + int((c_col * bed_w) / cell_cols)
            cx2 = bx1 + int(((c_col + 1) * bed_w) / cell_cols)
            cy1 = by1 + int((c_row * bed_h) / cell_rows)
            cy2 = by1 + int(((c_row + 1) * bed_h) / cell_rows)

            cw, ch = max(cx2 - cx1, 1), max(cy2 - cy1, 1)
            vx1 = clamp(cx1 + int(cw * left_ratio), cx1, cx2 - 1)
            vx2 = clamp(cx2 - int(cw * right_pad), vx1 + 1, cx2)
            vy1 = clamp(cy1 + int(ch * top_pad), cy1, cy2 - 1)
            vy2 = clamp(cy2 - int(ch * bottom_pad), vy1 + 1, cy2)

            # MOD: vital毎のROI補正を適用
            vital_adjust = adjust_cfg.get(vital) if isinstance(adjust_cfg, dict) else None
            if isinstance(vital_adjust, dict):
                vx1 = clamp(vx1 + _offset_from_adjust(vital_adjust, "dx1", cw, ch), cx1, cx2 - 1)
                vy1 = clamp(vy1 + _offset_from_adjust(vital_adjust, "dy1", cw, ch), cy1, cy2 - 1)
                vx2 = clamp(vx2 + _offset_from_adjust(vital_adjust, "dx2", cw, ch), vx1 + 1, cx2)
                vy2 = clamp(vy2 + _offset_from_adjust(vital_adjust, "dy2", cw, ch), vy1 + 1, cy2)

            out[bed][vital] = (vx1, vy1, vx2, vy2)
    return out


def copy_cache_snapshot(cache_path: Path, day_dir: Path, stamp: str) -> str | None:
    out = day_dir / "cache" / f"{stamp}_monitor_cache.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        try:
            shutil.copy2(cache_path, out)
            return out.as_posix()
        except Exception:
            time.sleep(0.05)
    return None


def maybe_launch_monitor(no_launch_monitor: bool, cache_path: Path) -> subprocess.Popen[str] | None:
    # MOD: no-launch-monitorを追加し、既存のmonitor起動挙動を切替可能にする
    if no_launch_monitor:
        print("[INFO] --no-launch-monitor=true: monitor.py launch skipped")
        return None
    cmd = [sys.executable, "monitor.py", "--cache", str(cache_path), "--fullscreen", "true"]
    try:
        print(f"[INFO] launching monitor process: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
        time.sleep(1.5)
        print(f"[INFO] monitor process started pid={proc.pid}")
        return proc
    except Exception as exc:
        print(f"[WARN] failed to launch monitor.py: {exc}", file=sys.stderr)
        return None


def run_validator_once(ocr_jsonl: Path, cache_path: Path, validator_config: Path, last_n: int) -> None:
    # MOD: オプションでcapture毎にvalidatorを実行
    cmd = [
        sys.executable,
        "validator.py",
        "--ocr-results",
        str(ocr_jsonl),
        "--monitor-cache",
        str(cache_path),
        "--validator-config",
        str(validator_config),
        "--last",
        str(last_n),
    ]
    try:
        result = subprocess.run(cmd, check=False, timeout=120)
        print(f"[INFO] validator exit code={result.returncode}")
    except Exception as exc:
        print(f"[WARN] validator execution failed: {exc}", file=sys.stderr)


def run_calibration(
    sct: mss.mss,
    config_path: Path,
    config: dict[str, Any],
    monitor_index: int,
    mss_monitor_index: int | None,
    use_gpu: bool,
) -> None:
    base = choose_capture_region(sct, config, monitor_index, mss_monitor_index)
    frame = grab_frame(sct, base)

    selected = cv2.selectROI("Select monitor.py region", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select monitor.py region")
    x, y, w, h = [int(v) for v in selected]
    if w <= 0 or h <= 0:
        raise RuntimeError("Calibration cancelled: monitor_rect was not selected.")

    config["monitor_index"] = monitor_index
    config["monitor_rect"] = {"left": base.left + x, "top": base.top + y, "width": w, "height": h}
    cropped = frame[y : y + h, x : x + w].copy()
    reader = easyocr.Reader(["en"], gpu=use_gpu)

    win = "Calibration (s=save, q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def add_trackbar(name: str, value: int, max_value: int) -> None:
        cv2.createTrackbar(name, win, value, max_value, lambda _v: None)

    add_trackbar("header_crop_px", int(config.get("header_crop_px", 60)), max(1, h - 1))
    add_trackbar("value_left_ratio(%)", int(float(config["value_roi"].get("left_ratio", 0.45)) * 100), 90)
    add_trackbar("right_pad(%)", int(float(config["value_roi"].get("right_pad", 0.06)) * 100), 50)
    add_trackbar("top_pad(%)", int(float(config["value_roi"].get("top_pad", 0.10)) * 100), 50)
    add_trackbar("bottom_pad(%)", int(float(config["value_roi"].get("bottom_pad", 0.12)) * 100), 50)

    while True:
        config["header_crop_px"] = cv2.getTrackbarPos("header_crop_px", win)
        config["value_roi"]["left_ratio"] = cv2.getTrackbarPos("value_left_ratio(%)", win) / 100.0
        config["value_roi"]["right_pad"] = cv2.getTrackbarPos("right_pad(%)", win) / 100.0
        config["value_roi"]["top_pad"] = cv2.getTrackbarPos("top_pad(%)", win) / 100.0
        config["value_roi"]["bottom_pad"] = cv2.getTrackbarPos("bottom_pad(%)", win) / 100.0

        rois = build_vital_rois((h, w), config)
        x1, y1, x2, y2 = rois["BED01"]["RAP_M"]
        roi_img = cropped[y1:y2, x1:x2]
        text, conf = run_ocr(reader, preprocess_image(roi_img, config))

        left = cropped.copy()
        cv2.rectangle(left, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(left, f"BED01 RAP_M text={text} conf={conf:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        preview_h = left.shape[0]
        preview_w = max(int(preview_h * 0.8), 280)
        preview = np.zeros((preview_h, preview_w, 3), dtype=np.uint8)
        if roi_img.size > 0:
            roi_show = cv2.resize(roi_img, (preview_w - 20, min(preview_h - 80, (preview_w - 20) // 2 + 20)))
            py = 20
            preview[py : py + roi_show.shape[0], 10 : 10 + roi_show.shape[1]] = roi_show
        cv2.putText(preview, f"OCR: {text}", (10, preview_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(preview, f"Conf: {conf:.3f}", (10, preview_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        canvas = np.hstack([left, preview])
        cv2.imshow(win, canvas)
        key = cv2.waitKey(40) & 0xFF
        if key == ord("s"):
            save_config(config_path, config)
            print(f"[INFO] calibration saved: {config_path}")
            break
        if key in (ord("q"), 27):
            print("[INFO] calibration aborted")
            break

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture monitor display and OCR all vitals")
    parser.add_argument("--cache", default="monitor_cache.json")
    parser.add_argument("--config", default="ocr_capture_config.json")
    parser.add_argument("--outdir", default="dataset")
    parser.add_argument("--monitor-index", type=int, default=2)
    parser.add_argument("--mss-monitor-index", type=int, default=None)  # MOD
    parser.add_argument("--interval-ms", type=int, default=10000)
    parser.add_argument("--save-images", type=parse_bool, default=True)
    parser.add_argument("--debug-roi", type=parse_bool, default=False)
    parser.add_argument("--gpu", type=parse_bool, default=True)
    parser.add_argument("--no-launch-monitor", type=parse_bool, default=True)  # MOD: 安全側デフォルト
    parser.add_argument("--run-validator", type=parse_bool, default=False)  # MOD
    parser.add_argument("--validator-last", type=int, default=50)  # MOD
    parser.add_argument("--validator-config", default="validator_config.json")  # MOD
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = ensure_config(config_path)
    cache_path = Path(args.cache)
    validator_config = Path(args.validator_config)

    # MOD: GPU情報を詳細ログ出力
    cuda_available = torch.cuda.is_available()
    print(f"[INFO] torch.__version__={torch.__version__}")
    print(f"[INFO] torch.version.cuda={torch.version.cuda}")
    print(f"[INFO] torch.cuda.is_available()={cuda_available}")
    use_gpu = args.gpu
    if use_gpu and not cuda_available:
        print("[WARN] GPU requested but CUDA is unavailable. Falling back to gpu=False.")
        use_gpu = False

    launched_monitor = maybe_launch_monitor(args.no_launch_monitor, cache_path)

    with mss.mss() as sct:
        log_monitors(sct)
        if args.calibrate:
            run_calibration(sct, config_path, config, args.monitor_index, args.mss_monitor_index, use_gpu)
            return

        reader = easyocr.Reader(["en"], gpu=use_gpu)

        monitor_base = choose_capture_region(sct, config, args.monitor_index, args.mss_monitor_index)
        capture_rect = get_monitor_rect(config, monitor_base)
        print(
            "[INFO] final capture rect "
            f"left={capture_rect.left} top={capture_rect.top} width={capture_rect.width} height={capture_rect.height}"
        )

        try:
            while True:
                tick = time.time()
                print(f"[INFO] capture tick start ts={datetime.now().isoformat(timespec='seconds')}")  # MOD
                try:
                    day = datetime.now().strftime("%Y%m%d")
                    day_dir = Path(args.outdir) / day
                    images_dir = day_dir / "images"
                    debug_dir = day_dir / "debug"
                    day_dir.mkdir(parents=True, exist_ok=True)
                    if args.save_images:
                        images_dir.mkdir(parents=True, exist_ok=True)

                    frame = grab_frame(sct, capture_rect)
                    rois = build_vital_rois((frame.shape[0], frame.shape[1]), config)
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

                    image_path = None
                    if args.save_images:
                        image_path = images_dir / f"{stamp}.png"
                        cv2.imwrite(str(image_path), frame)

                    if args.debug_roi:
                        # MOD: debug-roiにbed/vital名を描画
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        debug_img = frame.copy()
                        for bed in BED_IDS:
                            for vital in VITAL_ORDER:
                                x1, y1, x2, y2 = rois[bed][vital]
                                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                                cv2.putText(
                                    debug_img,
                                    f"{bed}:{vital}",
                                    (x1 + 1, max(y1 + 11, 11)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.28,
                                    (255, 0, 0),
                                    1,
                                    cv2.LINE_AA,
                                )
                        cv2.imwrite(str(debug_dir / f"{stamp}_rois.png"), debug_img)

                    beds: dict[str, dict[str, Any]] = {bed: {} for bed in BED_IDS}
                    for bed in BED_IDS:
                        for vital in VITAL_ORDER:
                            x1, y1, x2, y2 = rois[bed][vital]
                            roi = frame[y1:y2, x1:x2]
                            text, conf = run_ocr(reader, preprocess_image(roi, config))
                            beds[bed][vital] = {"text": text, "value": parse_value(text), "confidence": conf}

                    cache_snapshot = copy_cache_snapshot(cache_path, day_dir, stamp)
                    if not cache_snapshot:
                        print(f"[WARN] cache snapshot failed at {stamp}", file=sys.stderr)

                    record = {
                        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                        "image_path": image_path.as_posix() if image_path else None,
                        "cache_snapshot_path": cache_snapshot,
                        "beds": beds,
                    }
                    jsonl = day_dir / "ocr_results.jsonl"
                    with jsonl.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fh.flush()

                    if args.run_validator:
                        run_validator_once(jsonl, cache_path, validator_config, max(args.validator_last, 1))

                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] frame skipped due to error: {exc}", file=sys.stderr)

                sleep_ms = max(args.interval_ms - (time.time() - tick) * 1000, 0)
                print(f"[INFO] sleeping {sleep_ms:.1f} ms")  # MOD
                time.sleep(sleep_ms / 1000.0)
        finally:
            # MOD: 起動したmonitorプロセスを安全終了
            if launched_monitor is not None:
                try:
                    launched_monitor.terminate()
                except Exception:
                    pass


if __name__ == "__main__":
    main()
