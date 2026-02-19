#!/usr/bin/env python3
"""
OCR capture application.

Usage examples:
  # 1) One-time calibration (adjust ROI and save config)
  python ocr_capture_app.py --calibrate --config ocr_capture_config.json --target-display primary

  # 2) Periodic capture (10s interval)
  python ocr_capture_app.py --config ocr_capture_config.json --interval-ms 10000 --save-images true --debug-roi false
"""

from __future__ import annotations

import argparse
import ctypes
import json
import importlib
import os
import unicodedata
import re
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from ctypes import wintypes

import cv2
import numpy as np

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
NUMERIC_RE = re.compile(r"^\d+(\.\d+)?$")
ALLOWLIST = "0123456789."
MONITORINFOF_PRIMARY = 0x00000001
ROI_MAP_PATH = Path("dataset/layout/monitor_roi_map.json")
CAPTURE_CACHE_FILENAME_PATTERN = re.compile(r"^(\d{8})_(\d{6})_(\d{3})(?:_before)?_monitor_cache\.json$")

TEMPERATURE_VITALS = {"TSKIN", "TRECT"}
INTEGER_PREFERRED_VITALS = {"HR", "SpO2", "RR", "rRESP", "EtCO2", "Ppeak", "PEEP", "O2conc", "NO", "BSR1", "BSR2"}
DEFAULT_VITAL_RANGES: dict[str, tuple[float, float]] = {
    "TSKIN": (30.0, 45.0),
    "TRECT": (30.0, 45.0),
    "HR": (30.0, 250.0),
    "SpO2": (30.0, 100.0),
    "ART_S": (0.0, 250.0),
    "ART_D": (0.0, 250.0),
    "ART_M": (0.0, 250.0),
    "CVP_M": (0.0, 30.0),
    "RAP_M": (0.0, 30.0),
    "rRESP": (0.0, 80.0),
    "EtCO2": (0.0, 120.0),
    "RR": (0.0, 80.0),
    "VTe": (0.0, 2000.0),
    "VTi": (0.0, 2000.0),
    "Ppeak": (0.0, 80.0),
    "PEEP": (0.0, 40.0),
    "O2conc": (0.0, 100.0),
    "NO": (0.0, 100.0),
    "BSR1": (0.0, 100.0),
    "BSR2": (0.0, 100.0),
}

DEFAULT_FIELD_ALLOWLIST = "0123456789"
FIELD_ALLOWLISTS: dict[str, str] = {
    "TSKIN": "0123456789.",
    "TRECT": "0123456789.",
    "CVP_M": "0123456789.-",
    "RAP_M": "0123456789.-",
}
FIELD_PATTERNS: dict[str, re.Pattern[str]] = {
    "TSKIN": re.compile(r"^\d{2}(?:\.\d)?$"),
    "TRECT": re.compile(r"^\d{2}(?:\.\d)?$"),
    "HR": re.compile(r"^\d{1,3}$"),
    "RR": re.compile(r"^\d{1,3}$"),
    "rRESP": re.compile(r"^\d{1,3}$"),
    "SpO2": re.compile(r"^\d{1,3}$"),
    "CVP_M": re.compile(r"^-?\d{1,2}(?:\.\d+)?$"),
    "RAP_M": re.compile(r"^-?\d{1,2}(?:\.\d+)?$"),
}


class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long), ("right", ctypes.c_long), ("bottom", ctypes.c_long)]


class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_ulong), ("rcMonitor", RECT), ("rcWork", RECT), ("dwFlags", ctypes.c_ulong)]

DEFAULT_CONFIG: dict[str, Any] = {
    "window_title": "HL7 Bed Monitor",
    "monitor_index": 2,
    "monitor_rect": None,
    "header_crop_px": 60,
    "bed_grid": {"cols": 2, "rows": 3},
    "cell_grid": {"cols": 4, "rows": 5},
    "roi_strategy": "cell_inner_slice",
    "value_roi": {
        "left_ratio": 0.45,
        "right_pad": 0.06,
        "top_pad": 0.10,
        "bottom_pad": 0.12,
    },
    "value_box": {
        "x1_ratio": 0.60,
        "x2_ratio": 0.98,
        "y1_ratio": 0.20,
        "y2_ratio": 0.88,
        "pad_px": 2,
    },
    "cell_inner_box": {
        "pad_px": 4,
    },
    "cell_value_slice": {
        "x1_ratio": 0.60,
        "x2_ratio": 0.98,
        "y1_ratio": 0.12,
        "y2_ratio": 0.90,
    },
    "value_roi_inset_px": 2,
    # MOD: vital別ROI微調整設定を追加（px/ratioの両対応）
    "per_vital_roi_adjust": {},
    "preprocess": {
        "scale": 4.0,
        "threshold": True,
        "threshold_value": 170,
        "resize": 4.0,
        "trim_px": 3,
        "threshold_mode": "adaptive",
        "adaptive_block_size": 31,
        "adaptive_c": 2,
        "morph_kernel": 2,
        "ocr_variants": ["normal", "invert", "otsu"],
    },
    "ocr_numeric": {
        "morph_kernel": 2,
    },
    "debug_save_failed_roi": True,
    "failed_roi_conf_threshold": 0.70,
    "debug_popup_failed_roi": False,
    "enable_tesseract_fallback": True,
    "ocr_compare_mode": False,
    "tesseract_cmd": "",
    "vital_ranges": DEFAULT_VITAL_RANGES,
}


_TESSERACT_WARNED = False


@dataclass
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def find_monitor_window_hwnd(title: str) -> int | None:
    if not sys.platform.startswith("win"):
        return None

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    query = title.strip()
    if not query:
        return None

    try:
        hwnd = int(user32.FindWindowW(None, query))
        if hwnd:
            return hwnd
    except Exception as exc:
        print(f"[WARN] FindWindowW failed: {exc}", file=sys.stderr)

    matches: list[int] = []

    enum_proc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = int(user32.GetWindowTextLengthW(hwnd))
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        text = buf.value or ""
        if query.lower() in text.lower():
            matches.append(int(hwnd))
            return False
        return True

    try:
        user32.EnumWindows(enum_proc(callback), 0)
    except Exception as exc:
        print(f"[WARN] EnumWindows failed: {exc}", file=sys.stderr)
        return None

    return matches[0] if matches else None


def get_window_capture_region(title: str, capture_client: bool = True) -> CaptureRegion | None:
    if not sys.platform.startswith("win"):
        return None

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    hwnd = find_monitor_window_hwnd(title)
    if not hwnd:
        return None

    if not capture_client:
        print("[WARN] capture_client=false is deprecated. Forcing Win32 client-area capture.", file=sys.stderr)

    client_rect = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(client_rect)):
        return None

    left_top = POINT(client_rect.left, client_rect.top)
    right_bottom = POINT(client_rect.right, client_rect.bottom)
    if not user32.ClientToScreen(hwnd, ctypes.byref(left_top)):
        return None
    if not user32.ClientToScreen(hwnd, ctypes.byref(right_bottom)):
        return None

    left = int(left_top.x)
    top = int(left_top.y)
    width = int(right_bottom.x - left_top.x)
    height = int(right_bottom.y - left_top.y)
    if width <= 0 or height <= 0:
        return None

    return CaptureRegion(left=left, top=top, width=width, height=height)


def dpi_aware() -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        context = ctypes.c_void_p(-4)  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        ok = user32.SetProcessDpiAwarenessContext(context)
        if ok:
            print("[INFO] DPI awareness enabled via SetProcessDpiAwarenessContext(PER_MONITOR_AWARE_V2)")
            return
        print("[WARN] SetProcessDpiAwarenessContext() returned FALSE", file=sys.stderr)
    except Exception as exc:
        print(f"[WARN] SetProcessDpiAwarenessContext() failed: {exc}", file=sys.stderr)

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # type: ignore[attr-defined]
        print("[INFO] DPI awareness enabled via SetProcessDpiAwareness(2)")
        return
    except Exception as exc:
        print(f"[WARN] SetProcessDpiAwareness(2) failed: {exc}", file=sys.stderr)
    try:
        ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
        print("[INFO] DPI awareness enabled via SetProcessDPIAware()")
    except Exception as exc:
        print(f"[WARN] SetProcessDPIAware() failed: {exc}", file=sys.stderr)


def enum_windows_monitors() -> list[dict[str, int | bool]]:
    if not sys.platform.startswith("win"):
        print("[WARN] Windows monitor enumeration is only available on Windows", file=sys.stderr)
        return []

    monitors: list[dict[str, int | bool]] = []
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    callback_type = ctypes.WINFUNCTYPE(
        ctypes.c_int,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(RECT),
        wintypes.LPARAM,
    )

    def callback(hmonitor, _hdc, _lprc, _lparam):
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        ok = user32.GetMonitorInfoW(hmonitor, ctypes.byref(info))
        if not ok:
            print(f"[WARN] GetMonitorInfoW failed for hmonitor={hmonitor}", file=sys.stderr)
            return 1

        left = int(info.rcMonitor.left)
        top = int(info.rcMonitor.top)
        width = int(info.rcMonitor.right - info.rcMonitor.left)
        height = int(info.rcMonitor.bottom - info.rcMonitor.top)
        is_primary = bool(info.dwFlags & MONITORINFOF_PRIMARY)
        index = len(monitors) + 1
        monitors.append(
            {
                "index": index,
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "is_primary": is_primary,
            }
        )
        print(
            "[INFO] monitor "
            f"index={index} left={left} top={top} width={width} height={height} is_primary={is_primary}"
        )
        return 1

    try:
        user32.EnumDisplayMonitors(0, 0, callback_type(callback), 0)
    except Exception as exc:
        print(f"[WARN] EnumDisplayMonitors failed: {exc}", file=sys.stderr)
        return []
    return monitors


def pick_primary_monitor(monitors: list[dict[str, int | bool]]) -> dict[str, int | bool] | None:
    for monitor in monitors:
        if bool(monitor.get("is_primary", False)):
            return monitor
    print("[WARN] Primary monitor not found", file=sys.stderr)
    return monitors[0] if monitors else None


def resolve_mss_monitor_index(
    sct: mss.mss,
    target_display: str,
    windows_monitors: list[dict[str, int | bool]],
    display_index: int | None,
    mss_monitor_index: int | None,
) -> int:
    monitor_count = len(sct.monitors) - 1
    if monitor_count <= 0:
        raise RuntimeError("No mss monitors available")

    if target_display == "mss":
        selected = int(mss_monitor_index or 1)
        if selected < 1 or selected > monitor_count:
            print(f"[WARN] invalid mss-monitor-index={selected}; fallback to 1", file=sys.stderr)
            selected = 1
        monitor = sct.monitors[selected]
        print(
            "[INFO] target-display=mss resolved "
            f"mss-monitor-index={selected} rect=({monitor['left']},{monitor['top']},{monitor['width']},{monitor['height']})"
        )
        return selected

    if target_display == "index":
        selected_window = None
        for monitor in windows_monitors:
            if int(monitor.get("index", 0)) == int(display_index or 1):
                selected_window = monitor
                break
        if selected_window is None:
            print(f"[WARN] display-index={display_index} not found; fallback to primary", file=sys.stderr)
            selected_window = pick_primary_monitor(windows_monitors)
        if selected_window is None:
            return 1
        rect = (
            int(selected_window["left"]),
            int(selected_window["top"]),
            int(selected_window["width"]),
            int(selected_window["height"]),
        )
        best_idx = 1
        best_dist = None
        for idx in range(1, monitor_count + 1):
            mon = sct.monitors[idx]
            dist = abs(mon["left"] - rect[0]) + abs(mon["top"] - rect[1]) + abs(mon["width"] - rect[2]) + abs(mon["height"] - rect[3])
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx
        mon = sct.monitors[best_idx]
        print(
            "[INFO] target-display=index resolved "
            f"display-index={selected_window['index']} mss-monitor-index={best_idx} "
            f"rect=({mon['left']},{mon['top']},{mon['width']},{mon['height']})"
        )
        return best_idx

    primary = pick_primary_monitor(windows_monitors)
    if primary is None:
        print("[WARN] fallback primary monitor unavailable; using mss index=1", file=sys.stderr)
        return 1

    primary_rect = (
        int(primary["left"]),
        int(primary["top"]),
        int(primary["width"]),
        int(primary["height"]),
    )
    exact_match = None
    best_idx = 1
    best_dist = None
    for idx in range(1, monitor_count + 1):
        mon = sct.monitors[idx]
        current_rect = (int(mon["left"]), int(mon["top"]), int(mon["width"]), int(mon["height"]))
        if current_rect == primary_rect:
            exact_match = idx
            break
        dist = abs(current_rect[0] - primary_rect[0]) + abs(current_rect[1] - primary_rect[1]) + abs(current_rect[2] - primary_rect[2]) + abs(current_rect[3] - primary_rect[3])
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = idx

    selected = exact_match if exact_match is not None else best_idx
    mon = sct.monitors[selected]
    print(
        "[INFO] target-display=primary resolved "
        f"mss-monitor-index={selected} rect=({mon['left']},{mon['top']},{mon['width']},{mon['height']})"
    )
    return selected


def _resize_for_monitor_scoring(frame: np.ndarray, max_dim: int = 640) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(1.0, float(max_dim) / float(max(h, w, 1)))
    if scale >= 0.999:
        return frame
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def auto_detect_monitor(sct: mss.mss, debug: bool = False) -> tuple[int | None, float, list[dict[str, float]]]:
    """Estimate the monitor showing fullscreen HL7 monitor UI.

    Heuristics prioritize dark background and grid-like lines, then colorful/white
    text ratios (cyan/red/white) used by the monitor UI.
    """
    monitor_count = len(sct.monitors) - 1
    if monitor_count <= 0:
        return None, 0.0, []

    candidates: list[dict[str, float]] = []
    for idx in range(1, monitor_count + 1):
        mon = sct.monitors[idx]
        try:
            raw = sct.grab({"left": int(mon["left"]), "top": int(mon["top"]), "width": int(mon["width"]), "height": int(mon["height"])})
            frame = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)
        except Exception as exc:
            print(f"[WARN] auto_detect_monitor monitor={idx} capture failed: {exc}", file=sys.stderr)
            continue

        small = _resize_for_monitor_scoring(frame, max_dim=640)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h, w = small.shape[:2]
        px = float(max(h * w, 1))

        black_mask = cv2.inRange(hsv, (0, 0, 0), (180, 80, 45))
        cyan_mask = cv2.inRange(hsv, (75, 70, 110), (105, 255, 255))
        red_mask1 = cv2.inRange(hsv, (0, 80, 90), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (165, 80, 90), (180, 255, 255))
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        white_mask = cv2.inRange(hsv, (0, 0, 170), (180, 45, 255))

        black_ratio = float(cv2.countNonZero(black_mask) / px)
        cyan_ratio = float(cv2.countNonZero(cyan_mask) / px)
        red_ratio = float(cv2.countNonZero(red_mask) / px)
        white_ratio = float(cv2.countNonZero(white_mask) / px)

        line_info = detect_grid_lines(small)
        horizontal_mask = line_info.get("horizontal_mask")
        vertical_mask = line_info.get("vertical_mask")
        line_px = 0.0
        if isinstance(horizontal_mask, np.ndarray):
            line_px += float(cv2.countNonZero(horizontal_mask))
        if isinstance(vertical_mask, np.ndarray):
            line_px += float(cv2.countNonZero(vertical_mask))
        line_mask_ratio = line_px / px
        line_count = float(len(line_info.get("x_lines", [])) + len(line_info.get("y_lines", [])))
        line_strength = min(1.0, line_mask_ratio * 8.0 + min(line_count / 16.0, 1.0) * 0.5)

        score = (
            0.45 * black_ratio
            + 0.35 * line_strength
            + 0.12 * min(cyan_ratio * 12.0, 1.0)
            + 0.05 * min(red_ratio * 12.0, 1.0)
            + 0.03 * min(white_ratio * 8.0, 1.0)
        )
        candidates.append(
            {
                "index": float(idx),
                "score": float(score),
                "black_ratio": black_ratio,
                "line_strength": line_strength,
                "cyan_ratio": cyan_ratio,
                "red_ratio": red_ratio,
                "white_ratio": white_ratio,
            }
        )

    if not candidates:
        return None, 0.0, []

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]
    second = candidates[1] if len(candidates) > 1 else None
    best_score = float(best["score"])
    margin = best_score - float(second["score"]) if second else best_score
    confident = best_score >= 0.20 and margin >= 0.03

    if debug:
        for item in candidates:
            print(
                "[DEBUG] auto_detect_monitor candidate "
                f"mss-monitor-index={int(item['index'])} score={item['score']:.4f} "
                f"black_ratio={item['black_ratio']:.4f} line_strength={item['line_strength']:.4f} "
                f"cyan_ratio={item['cyan_ratio']:.4f} red_ratio={item['red_ratio']:.4f} "
                f"white_ratio={item['white_ratio']:.4f}"
            )

    if not confident:
        if debug:
            print(
                "[DEBUG] auto_detect_monitor low confidence "
                f"best_score={best_score:.4f} margin={margin:.4f}; fallback to primary"
            )
        return None, best_score, candidates

    return int(best["index"]), best_score, candidates


def log_windows_dpi_info() -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        dpi = int(user32.GetDpiForSystem())
        scale = dpi / 96.0
        print(f"[INFO] Windows DPI system_dpi={dpi} estimated_scaling={scale:.3f}x")
        return
    except Exception as exc:
        print(f"[WARN] GetDpiForSystem() unavailable: {exc}", file=sys.stderr)


def _scale_value(value: int, scale: float) -> int:
    return int(round(value * scale))


def normalize_capture_frame(
    frame: np.ndarray,
    capture_region: CaptureRegion,
    monitor_base: CaptureRegion,
) -> np.ndarray:
    """Convert absolute logical capture_rect into frame pixel coordinates.

    Handles DPI mismatch by scaling coordinates when grabbed image size differs
    from logical monitor_base size.
    """
    img_h, img_w = frame.shape[:2]
    base_w = max(monitor_base.width, 1)
    base_h = max(monitor_base.height, 1)
    scale_x = img_w / float(base_w)
    scale_y = img_h / float(base_h)
    print(
        "[INFO] capture_img "
        f"size={img_w}x{img_h} monitor_rect={base_w}x{base_h} "
        f"scale_x={scale_x:.6f} scale_y={scale_y:.6f}"
    )

    rel_x = capture_region.left - monitor_base.left
    rel_y = capture_region.top - monitor_base.top
    rel_w = capture_region.width
    rel_h = capture_region.height

    if abs(scale_x - 1.0) > 1e-6 or abs(scale_y - 1.0) > 1e-6:
        print("[WARN] logical/physical mismatch detected. Applying ROI scale correction.")

    x = clamp(_scale_value(rel_x, scale_x), 0, max(img_w - 1, 0))
    y = clamp(_scale_value(rel_y, scale_y), 0, max(img_h - 1, 0))
    w = max(_scale_value(rel_w, scale_x), 1)
    h = max(_scale_value(rel_h, scale_y), 1)
    x2 = clamp(x + w, x + 1, img_w)
    y2 = clamp(y + h, y + 1, img_h)

    print(
        "[INFO] final crop(pixel) "
        f"x={x} y={y} w={x2 - x} h={y2 - y} "
        f"from capture_rect(left={capture_region.left},top={capture_region.top},"
        f"width={capture_region.width},height={capture_region.height})"
    )
    return frame[y:y2, x:x2].copy()


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
    scale_value = pp_cfg.get("scale", pp_cfg.get("resize", 4.0))
    pp_cfg["scale"] = scale_value
    pp_cfg["resize"] = pp_cfg.get("resize", scale_value)
    pp_cfg["threshold"] = bool(pp_cfg.get("threshold", True))
    try:
        pp_cfg["threshold_value"] = int(pp_cfg.get("threshold_value", 170))
    except Exception:
        pp_cfg["threshold_value"] = 170
    variants_cfg = pp_cfg.get("ocr_variants", ["normal", "invert", "otsu"])
    if not isinstance(variants_cfg, list):
        variants_cfg = ["normal", "invert", "otsu"]
    pp_cfg["ocr_variants"] = [str(v).strip().lower() for v in variants_cfg if str(v).strip()]
    config["value_roi"] = roi_cfg
    vb_cfg = dict(DEFAULT_CONFIG.get("value_box", {}))
    vb_cfg.update(config.get("value_box", {}) if isinstance(config.get("value_box"), dict) else {})
    config["value_box"] = vb_cfg
    cell_inner_cfg = dict(DEFAULT_CONFIG.get("cell_inner_box", {}))
    cell_inner_cfg.update(config.get("cell_inner_box", {}) if isinstance(config.get("cell_inner_box"), dict) else {})
    config["cell_inner_box"] = cell_inner_cfg
    cell_slice_cfg = dict(DEFAULT_CONFIG.get("cell_value_slice", {}))
    cell_slice_cfg.update(config.get("cell_value_slice", {}) if isinstance(config.get("cell_value_slice"), dict) else {})
    config["cell_value_slice"] = cell_slice_cfg
    config["preprocess"] = pp_cfg
    config.setdefault("per_vital_roi_adjust", {})
    ranges = dict(DEFAULT_VITAL_RANGES)
    loaded_ranges = config.get("vital_ranges")
    if isinstance(loaded_ranges, dict):
        ranges.update(loaded_ranges)
    config["vital_ranges"] = ranges
    return config


def save_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def log_monitors(sct: mss.mss) -> None:
    # MOD: 起動時に常に利用可能monitor情報を表示
    for idx, monitor in enumerate(sct.monitors):
        print(
            "[INFO] mss.monitor "
            f"index={idx} left={monitor['left']} top={monitor['top']} width={monitor['width']} height={monitor['height']}"
        )


def choose_capture_region(
    sct: mss.mss,
    target_display: str,
    windows_monitors: list[dict[str, int | bool]],
    display_index: int | None,
    mss_monitor_index: int | None,
) -> CaptureRegion:
    selected = resolve_mss_monitor_index(
        sct=sct,
        target_display=target_display,
        windows_monitors=windows_monitors,
        display_index=display_index,
        mss_monitor_index=mss_monitor_index,
    )
    monitor = sct.monitors[selected]
    return CaptureRegion(int(monitor["left"]), int(monitor["top"]), int(monitor["width"]), int(monitor["height"]))


def get_primary_mss_monitor_base(
    sct: mss.mss,
    windows_monitors: list[dict[str, int | bool]],
) -> CaptureRegion:
    selected = resolve_mss_monitor_index(
        sct=sct,
        target_display="primary",
        windows_monitors=windows_monitors,
        display_index=None,
        mss_monitor_index=None,
    )
    monitor = sct.monitors[selected]
    return CaptureRegion(int(monitor["left"]), int(monitor["top"]), int(monitor["width"]), int(monitor["height"]))


def grab_frame(sct: mss.mss, region: CaptureRegion) -> np.ndarray:
    raw = sct.grab({"left": region.left, "top": region.top, "width": region.width, "height": region.height})
    return cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)


def write_debug_window_rect_image(path: Path, frame: np.ndarray) -> None:
    debug = frame.copy()
    h, w = debug.shape[:2]
    cv2.rectangle(debug, (0, 0), (max(w - 1, 0), max(h - 1, 0)), (255, 0, 0), 3)
    cv2.imwrite(str(path), debug)


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(v, hi))


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_exported_roi_map(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            return None
        items = payload.get("items")
        if not isinstance(items, dict):
            return None
        return payload
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] failed to load ROI map: {exc}", file=sys.stderr)
        return None


def build_exported_rois(
    frame: np.ndarray,
    capture_rect: CaptureRegion,
    roi_map: dict[str, Any],
    x1_ratio: float,
    x2_ratio: float,
    y1_ratio: float,
    y2_ratio: float,
) -> tuple[
    dict[str, dict[str, tuple[int, int, int, int]]],
    dict[str, dict[str, tuple[int, int, int, int]]],
    dict[str, dict[str, Any]],
]:
    frame_h, frame_w = frame.shape[:2]
    x1_ratio = min(max(x1_ratio, 0.0), 1.0)
    x2_ratio = min(max(x2_ratio, 0.0), 1.0)
    y1_ratio = min(max(y1_ratio, 0.0), 1.0)
    y2_ratio = min(max(y2_ratio, 0.0), 1.0)
    sx1, sx2 = min(x1_ratio, x2_ratio), max(x1_ratio, x2_ratio)
    sy1, sy2 = min(y1_ratio, y2_ratio), max(y1_ratio, y2_ratio)

    items = roi_map.get("items", {}) if isinstance(roi_map, dict) else {}
    roi_type = str(roi_map.get("roi_type", "")).strip().lower() if isinstance(roi_map, dict) else ""
    use_widget_bbox_as_value_roi = roi_type == "value_widget_bbox_screen"
    blue_rois: dict[str, dict[str, tuple[int, int, int, int]]] = {bed: {} for bed in BED_IDS}
    red_rois: dict[str, dict[str, tuple[int, int, int, int]]] = {bed: {} for bed in BED_IDS}
    debug_meta: dict[str, dict[str, Any]] = {bed: {"source": "exported", "cell_boxes": {}, "value_rois": {}} for bed in BED_IDS}

    for bed in BED_IDS:
        bed_items = items.get(bed, {}) if isinstance(items, dict) else {}
        for vital in VITAL_ORDER:
            rect = bed_items.get(vital) if isinstance(bed_items, dict) else None
            if not isinstance(rect, dict):
                continue
            try:
                bx1 = int(rect["x"]) - capture_rect.left
                by1 = int(rect["y"]) - capture_rect.top
                bw = int(rect["w"])
                bh = int(rect["h"])
            except Exception:
                continue
            if bw <= 0 or bh <= 0:
                continue
            bx2 = bx1 + bw
            by2 = by1 + bh
            if bx2 <= 0 or by2 <= 0 or bx1 >= frame_w or by1 >= frame_h:
                print(f"[WARN] exported ROI out of frame, skip {bed}.{vital}", file=sys.stderr)
                continue

            bx1c = clamp(bx1, 0, max(frame_w - 1, 0))
            by1c = clamp(by1, 0, max(frame_h - 1, 0))
            bx2c = clamp(bx2, bx1c + 1, frame_w)
            by2c = clamp(by2, by1c + 1, frame_h)

            if use_widget_bbox_as_value_roi:
                # monitor.py exports value_areas as roi_type=value_widget_bbox_screen.
                # Use that box as-is so red ROI matches the exact on-screen numeric area.
                rx1, ry1, rx2, ry2 = bx1c, by1c, bx2c, by2c
            else:
                bwc = max(bx2c - bx1c, 1)
                bhc = max(by2c - by1c, 1)
                rx1 = bx1c + int(bwc * sx1)
                rx2 = bx1c + int(bwc * sx2)
                ry1 = by1c + int(bhc * sy1)
                ry2 = by1c + int(bhc * sy2)
                rx1 = clamp(rx1, bx1c, bx2c - 1)
                ry1 = clamp(ry1, by1c, by2c - 1)
                rx2 = clamp(rx2, rx1 + 1, bx2c)
                ry2 = clamp(ry2, ry1 + 1, by2c)

            blue_rois[bed][vital] = (bx1c, by1c, bx2c, by2c)
            red_rois[bed][vital] = (rx1, ry1, rx2, ry2)
            debug_meta[bed]["cell_boxes"][vital] = (bx1c, by1c, bx2c, by2c)
            debug_meta[bed]["value_rois"][vital] = (rx1, ry1, rx2, ry2)
    return blue_rois, red_rois, debug_meta




def build_rois_for_frame(
    frame: np.ndarray,
    config: dict[str, Any],
    roi_source: str,
    capture_rect: CaptureRegion,
) -> tuple[
    dict[str, dict[str, tuple[int, int, int, int]]],
    dict[str, dict[str, tuple[int, int, int, int]]],
    dict[str, dict[str, Any]],
]:
    """Build ROI boxes in captured-frame coordinates only."""
    frame_h, frame_w = frame.shape[:2]
    if frame_w <= 0 or frame_h <= 0:
        empty = {bed: {} for bed in BED_IDS}
        return empty, empty, {bed: {} for bed in BED_IDS}

    if roi_source == "exported":
        roi_map = load_exported_roi_map(ROI_MAP_PATH)
        if roi_map is not None:
            value_slice_cfg = config.get("cell_value_slice") if isinstance(config.get("cell_value_slice"), dict) else {}
            slice_x1 = float(value_slice_cfg.get("x1_ratio", 0.60))
            slice_x2 = float(value_slice_cfg.get("x2_ratio", 0.98))
            slice_y1 = float(value_slice_cfg.get("y1_ratio", 0.10))
            slice_y2 = float(value_slice_cfg.get("y2_ratio", 0.92))
            return build_exported_rois(
                frame=frame,
                capture_rect=capture_rect,
                roi_map=roi_map,
                x1_ratio=slice_x1,
                x2_ratio=slice_x2,
                y1_ratio=slice_y1,
                y2_ratio=slice_y2,
            )
        print("[WARN] ROI map not found, fallback to legacy", file=sys.stderr)

    red_rois, roi_debug = build_vital_rois(frame, config, return_debug=True)
    return red_rois, red_rois, roi_debug


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


def preprocess_roi(img: np.ndarray, vital_name: str, config: dict[str, Any], variant: str = "normal") -> np.ndarray:
    pp = config.get("preprocess", {}) if isinstance(config.get("preprocess"), dict) else {}
    out: np.ndarray = img.copy()

    scale = float(pp.get("scale", pp.get("resize", 4.0)))
    if vital_name in TEMPERATURE_VITALS:
        scale = max(scale, 4.0)
    if scale > 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if out.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    trim_px = int(pp.get("trim_px", 3))
    if vital_name in TEMPERATURE_VITALS:
        trim_px = max(trim_px, 2)
    max_trim = max(min(out.shape[0], out.shape[1]) // 4, 0)
    trim_px = clamp(trim_px, 0, max_trim)
    if trim_px > 0 and out.shape[0] > trim_px * 2 and out.shape[1] > trim_px * 2:
        out = out[trim_px:-trim_px, trim_px:-trim_px]

    use_threshold = bool(pp.get("threshold", True))
    threshold_mode = str(pp.get("threshold_mode", "adaptive")).lower()
    if variant == "otsu" or threshold_mode == "otsu":
        _, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif variant == "adaptive" or threshold_mode == "adaptive":
        block_size = int(pp.get("adaptive_block_size", 31))
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(block_size, 3)
        c_value = int(pp.get("adaptive_c", 2))
        out = cv2.adaptiveThreshold(out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_value)
    elif use_threshold:
        threshold_value = int(pp.get("threshold_value", 170))
        threshold_value = clamp(threshold_value, 0, 255)
        _, out = cv2.threshold(out, threshold_value, 255, cv2.THRESH_BINARY)
    else:
        _, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_size = int(pp.get("morph_kernel", 2))
    if kernel_size > 0:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)

    if variant == "invert":
        out = cv2.bitwise_not(out)

    if np.mean(out) < 127:
        out = cv2.bitwise_not(out)

    return out


def add_padding(img: np.ndarray, pad: int = 30, color: int = 255) -> np.ndarray:
    pad_px = max(int(pad), 0)
    if pad_px <= 0:
        return img
    return cv2.copyMakeBorder(img, pad_px, pad_px, pad_px, pad_px, cv2.BORDER_CONSTANT, value=color)


def upscale(img: np.ndarray, scale: float = 3.0) -> np.ndarray:
    scale_v = float(scale)
    if scale_v <= 1.0:
        return img
    return cv2.resize(img, None, fx=scale_v, fy=scale_v, interpolation=cv2.INTER_CUBIC)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def binarize(img: np.ndarray, invert: bool = False) -> np.ndarray:
    gray = to_gray(img)
    mode = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, out = cv2.threshold(gray, 0, 255, mode + cv2.THRESH_OTSU)
    return out


def _morph(img: np.ndarray, kernel_size: int = 2) -> np.ndarray:
    k = max(int(kernel_size), 0)
    if k <= 0:
        return img
    kernel = np.ones((k, k), np.uint8)
    out = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    return out


def _field_allowlist(field_name: str) -> str:
    if field_name in {"HR", "RR", "NO", "SpO2"}:
        return "0123456789"
    if field_name in {"TEMP", "TSKIN", "TRECT"}:
        return "0123456789."
    if field_name in {"NIBP", "ART"}:
        return "0123456789/"
    return FIELD_ALLOWLISTS.get(field_name, DEFAULT_FIELD_ALLOWLIST)


def _field_pattern(field_name: str, allowlist: str) -> re.Pattern[str]:
    if field_name in {"HR", "RR", "NO", "SpO2"}:
        return re.compile(r"^\d{1,3}$")
    if field_name in {"TEMP", "TSKIN", "TRECT"}:
        return re.compile(r"^\d{2}(?:\.\d)?$")
    if field_name in {"NIBP", "ART"}:
        return re.compile(r"^\d{2,3}/\d{2,3}$")
    explicit = FIELD_PATTERNS.get(field_name)
    if explicit is not None:
        return explicit
    if "/" in allowlist:
        return re.compile(r"^\d{2,3}/\d{2,3}$")
    if "." in allowlist:
        return re.compile(r"^\d{1,4}(?:\.\d{1,2})?$")
    return re.compile(r"^\d{1,4}$")


def _normalize_ocr_text(text: str, allowlist: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = normalized.strip().replace(" ", "")
    normalized = "".join(ch for ch in normalized if ch in allowlist)
    if normalized.count(".") > 1:
        head, *tail = normalized.split(".")
        normalized = head + "." + "".join(tail)
    if normalized.count("/") > 1:
        head, *tail = normalized.split("/")
        normalized = head + "/" + "".join(tail)
    return normalized


def foreground_auto_tighten(
    img: np.ndarray,
    threshold: int = 200,
    min_black_pixels: int = 12,
    margin: int = 20,
) -> tuple[np.ndarray, dict[str, Any]]:
    gray = to_gray(img)
    mask = (gray < threshold).astype(np.uint8)
    black_pixel_count = int(mask.sum())
    debug: dict[str, Any] = {
        "threshold": int(threshold),
        "min_black_pixels": int(min_black_pixels),
        "black_pixel_count": black_pixel_count,
        "margin": int(margin),
        "boundingRect": None,
        "tightened": False,
    }
    if black_pixel_count < min_black_pixels:
        return img, debug

    points = cv2.findNonZero((mask * 255).astype(np.uint8))
    if points is None:
        return img, debug

    x, y, w, h = cv2.boundingRect(points)
    debug["boundingRect"] = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}

    x1 = clamp(x - margin, 0, img.shape[1] - 1)
    y1 = clamp(y - margin, 0, img.shape[0] - 1)
    x2 = clamp(x + w + margin, x1 + 1, img.shape[1])
    y2 = clamp(y + h + margin, y1 + 1, img.shape[0])
    subimg = img[y1:y2, x1:x2].copy()
    debug["tightened"] = True
    debug["tight_rect"] = {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
    return subimg, debug


def _parse_value_with_allowlist(text: str, field_name: str, config: dict[str, Any]) -> float | None:
    if "/" in text:
        return None

    normalized = text.strip()
    if normalized.count("-") > 1:
        return None
    if "-" in normalized and not normalized.startswith("-"):
        return None

    try:
        value = float(normalized)
    except ValueError:
        return parse_numeric(normalized, field_name, config)

    if field_name in INTEGER_PREFERRED_VITALS:
        value = float(int(round(value)))
    if _is_in_range(value, field_name, config):
        return value
    return None


def _continuity_bonus(value: float | None, prior_value: float | int | None) -> float:
    if value is None or prior_value is None:
        return 0.0
    prior = safe_float(prior_value)
    if prior is None:
        return 0.0
    base = max(abs(prior), 1.0)
    delta_ratio = abs(value - prior) / base
    if delta_ratio <= 0.03:
        return 0.25
    if delta_ratio <= 0.10:
        return 0.15
    if delta_ratio <= 0.30:
        return 0.05
    return -0.10


def _score_candidate(
    text: str,
    conf: float | None,
    format_ok: bool,
    value: float | None,
    prior_value: float | int | None,
) -> float:
    score = (conf if conf is not None else 0.35) * 1.5
    score += 0.8 if format_ok else -0.4
    score += min(len(text), 4) * 0.08
    score += 0.2 if value is not None else -0.2
    score += _continuity_bonus(value, prior_value)
    return score


def _easyocr_read_numeric(
    reader: easyocr.Reader,
    image: np.ndarray,
    allowlist: str,
) -> list[tuple[str, float | None]]:
    results = reader.readtext(
        image,
        detail=1,
        paragraph=False,
        allowlist=allowlist,
        decoder="greedy",
    )
    parsed: list[tuple[str, float | None]] = []
    for row in results:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        _bbox, text, conf = row
        conf_v = safe_float(conf)
        parsed.append((str(text), conf_v))
    return parsed


def _get_pytesseract(config: dict[str, Any]) -> Any | None:
    global _TESSERACT_WARNED
    try:
        pytesseract = importlib.import_module("pytesseract")
    except Exception:
        if not _TESSERACT_WARNED:
            print("[WARN] pytesseract is unavailable. Tesseract fallback is disabled.", file=sys.stderr)
            _TESSERACT_WARNED = True
        return None

    tesseract_cmd = str(config.get("tesseract_cmd", "") or "").strip()
    if tesseract_cmd:
        try:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        except Exception as exc:  # noqa: BLE001
            if not _TESSERACT_WARNED:
                print(f"[WARN] failed to set tesseract_cmd='{tesseract_cmd}': {exc}", file=sys.stderr)
                _TESSERACT_WARNED = True
    return pytesseract


def _normalize_tesseract_text(text: str, allowlist: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    normalized = normalized.strip().replace(" ", "").replace("\n", "")
    normalized = "".join(ch for ch in normalized if not ch.isspace())
    filtered = "".join(ch for ch in normalized if ch in allowlist)
    if not filtered and "O" in normalized and "0" in allowlist:
        maybe = normalized.replace("O", "0").replace("o", "0")
        filtered = "".join(ch for ch in maybe if ch in allowlist)

    cleaned = _normalize_ocr_text(filtered, allowlist)
    if "." in allowlist and cleaned:
        # Tesseractの端数記号ゆらぎを吸収: "7." -> "7", ".7" -> "0.7", "-.7" -> "-0.7"
        if cleaned.endswith("."):
            cleaned = cleaned[:-1]
        if cleaned.startswith("-."):
            cleaned = "-0" + cleaned[1:]
        elif cleaned.startswith("."):
            cleaned = "0" + cleaned
    return cleaned


def _build_tesseract_pass_images(image: np.ndarray) -> dict[str, np.ndarray]:
    gray = to_gray(image)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_inv = cv2.bitwise_not(otsu)
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    adaptive_inv = cv2.bitwise_not(adaptive)
    return {
        "gray": gray,
        "otsu": otsu,
        "otsu_inv": otsu_inv,
        "adaptive": adaptive,
        "adaptive_inv": adaptive_inv,
    }


def _tesseract_read_numeric(
    image: np.ndarray,
    *,
    allowlist: str,
    pattern: re.Pattern[str],
    field_name: str,
    prior_value: float | int | None,
    config: dict[str, Any],
    tighten_debug: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    global _TESSERACT_WARNED
    pytesseract = _get_pytesseract(config)
    debug: dict[str, Any] = {
        "tesseract_used": False,
        "tesseract_text": "",
        "tesseract_raw": "",
        "tesseract_config": None,
        "tesseract_exception": None,
    }
    if pytesseract is None:
        return None, debug

    whitelist = "".join(ch for ch in allowlist if ch in "0123456789.-") or "0123456789."
    psm_candidates = [7, 8, 6, 13]
    oem_candidates = [1, 3]
    tess_cfg = ""
    debug["tesseract_used"] = True
    debug["psm_candidates"] = psm_candidates
    debug["oem_candidates"] = oem_candidates
    debug["whitelist"] = whitelist
    pass_images = _build_tesseract_pass_images(image)
    debug["passes"] = list(pass_images.keys())
    best_attempt: dict[str, Any] | None = None

    try:
        output_cls = getattr(pytesseract, "Output", None)
        output_dict = getattr(output_cls, "DICT", None) if output_cls is not None else None

        for pass_name, pass_image in pass_images.items():
            for oem in oem_candidates:
                for psm in psm_candidates:
                    tess_cfg = f"--oem {oem} --psm {psm} -c tessedit_char_whitelist={whitelist}"
                    debug["tesseract_config"] = tess_cfg

                    raw_text = str(pytesseract.image_to_string(pass_image, config=tess_cfg))
                    debug["tesseract_raw"] = raw_text
                    norm_text = _normalize_tesseract_text(raw_text, allowlist)

                    conf_v: float | None = None
                    if output_dict is not None:
                        data = pytesseract.image_to_data(pass_image, config=tess_cfg, output_type=output_dict)
                        if isinstance(data, dict):
                            text_rows = data.get("text", [])
                            conf_rows = data.get("conf", [])
                            accepted_text: list[str] = []
                            accepted_conf: list[float] = []
                            if isinstance(text_rows, list) and isinstance(conf_rows, list):
                                for txt, conf in zip(text_rows, conf_rows):
                                    token = _normalize_tesseract_text(txt, allowlist)
                                    if not token:
                                        continue
                                    conf_token = safe_float(conf)
                                    if conf_token is None or conf_token < 0:
                                        continue
                                    accepted_text.append(token)
                                    accepted_conf.append(conf_token)
                            if accepted_text:
                                joined = "".join(accepted_text)
                                joined_norm = _normalize_tesseract_text(joined, allowlist)
                                if joined_norm:
                                    norm_text = joined_norm
                                if accepted_conf:
                                    conf_v = sum(accepted_conf) / len(accepted_conf) / 100.0

                    debug["tesseract_text"] = norm_text
                    if not norm_text or pattern.match(norm_text) is None:
                        continue
                    value = _parse_value_with_allowlist(norm_text, field_name, config)
                    if value is None:
                        continue
                    score = _score_candidate(norm_text, conf_v if conf_v is not None else 0.85, True, value, prior_value)
                    result = {
                        "text": norm_text,
                        "value": value,
                        "conf": conf_v,
                        "method": "fallback_tesseract",
                        "ocr_pass": f"tesseract_{pass_name}",
                        "imputed": False,
                        "score": score,
                        "psm": psm,
                        "oem": oem,
                        "whitelist": whitelist,
                    }
                    if best_attempt is None or score > float(best_attempt.get("score", -1.0)):
                        best_attempt = result

        if best_attempt is not None:
            return best_attempt, debug
        return None, debug
    except Exception as exc:  # noqa: BLE001
        debug["tesseract_exception"] = str(exc)
        if not _TESSERACT_WARNED:
            print(f"[WARN] Tesseract fallback failed: {exc}", file=sys.stderr)
            _TESSERACT_WARNED = True
        return None, debug


def ocr_numeric_roi(
    img: np.ndarray,
    reader: easyocr.Reader,
    field_name: str | None = None,
    prior_value: float | int | None = None,
    config: dict[str, Any] | None = None,
    ocr_engine: str = "easyocr",
    run_tesseract_for_field: bool = False,
    low_conf_threshold: float = 0.5,
) -> dict[str, Any]:
    field = str(field_name or "")
    allowlist = _field_allowlist(field)
    pattern = _field_pattern(field, allowlist)
    cfg = config if isinstance(config, dict) else DEFAULT_CONFIG
    compare_mode = bool(cfg.get("ocr_compare_mode", False))

    if field in {"CVP_M", "RAP_M"}:
        subimg, tighten_debug = foreground_auto_tighten(img)
    else:
        subimg = img
        tighten_debug = {
            "boundingRect": None,
            "black_pixel_count": None,
            "tightened": False,
        }

    padded = add_padding(subimg, pad=30)
    upscale_scale = 4 if field in {"CVP_M", "RAP_M"} else 3
    upscaled = upscale(padded, scale=upscale_scale)
    gray = to_gray(upscaled)

    pass_image_map: dict[str, np.ndarray] = {
        "gray": gray,
        "otsu": binarize(gray, invert=False),
        "otsu_inv": binarize(gray, invert=True),
    }

    candidates: list[dict[str, Any]] = []
    raw_easyocr: dict[str, Any] = {}
    chosen_pass = "none"
    chosen_method = "none"
    preprocessed_image = gray
    attempted_passes: list[str] = []
    pass_errors: dict[str, Any] = {}
    tesseract_debug: dict[str, Any] = {
        "tesseract_used": False,
        "tesseract_text": "",
        "tesseract_raw": "",
        "tesseract_config": None,
        "tesseract_exception": None,
    }
    easy_elapsed_ms: float = 0.0
    tess_elapsed_ms: float | None = None

    def run_pass(pass_name: str, method: str) -> None:
        nonlocal preprocessed_image, chosen_pass, chosen_method
        pass_img = pass_image_map[pass_name]
        preprocessed_image = pass_img
        attempted_passes.append(pass_name)
        try:
            ocr_rows = _easyocr_read_numeric(reader, pass_img, allowlist)
        except Exception as exc:  # noqa: BLE001
            pass_errors[pass_name] = {
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
            raw_easyocr[pass_name] = []
            return

        raw_easyocr[pass_name] = [{"text": text, "conf": conf, "method": method} for text, conf in ocr_rows]
        for raw_text, conf in ocr_rows:
            norm_text = _normalize_ocr_text(raw_text, allowlist)
            if not norm_text or not pattern.match(norm_text):
                continue
            parsed_value = _parse_value_with_allowlist(norm_text, field, cfg)
            score = _score_candidate(norm_text, conf, True, parsed_value, prior_value)
            candidates.append(
                {
                    "text": norm_text,
                    "value": parsed_value,
                    "conf": conf,
                    "method": method,
                    "ocr_pass": pass_name,
                    "score": score,
                }
            )
        if candidates and chosen_pass == "none":
            chosen_pass = pass_name
            chosen_method = method

    easy_start = time.perf_counter()
    for pass_name in ("gray", "otsu"):
        run_pass(pass_name, method="preprocess_multi_pass")
        if candidates:
            break

    if not candidates:
        run_pass("otsu_inv", method="preprocess_multi_pass")
    easy_elapsed_ms = (time.perf_counter() - easy_start) * 1000.0

    easyocr_winner = max(candidates, key=lambda c: c.get("score", -1.0)) if candidates else None
    easyocr_failed = (
        easyocr_winner is None
        or str(easyocr_winner.get("text") or "") == ""
        or easyocr_winner.get("value") is None
        or bool(pass_errors)
        or _raw_easyocr_all_empty(raw_easyocr)
    )

    enable_tesseract = bool(cfg.get("enable_tesseract_fallback", True))
    selected_engine = "easyocr"
    easyocr_conf = safe_float(easyocr_winner.get("conf")) if easyocr_winner else None
    easyocr_low_conf = easyocr_conf is None or easyocr_conf < float(low_conf_threshold)
    easyocr_unusable = easyocr_failed or easyocr_low_conf

    should_run_tesseract = False
    if compare_mode:
        should_run_tesseract = True
    elif ocr_engine == "tesseract":
        should_run_tesseract = True
    elif ocr_engine == "both":
        should_run_tesseract = run_tesseract_for_field
    elif ocr_engine == "easyocr" and enable_tesseract and easyocr_unusable:
        should_run_tesseract = True

    final_winner = easyocr_winner
    tesseract_result = None
    if should_run_tesseract:
        tess_start = time.perf_counter()
        tesseract_result, t_debug = _tesseract_read_numeric(
            preprocessed_image,
            allowlist=allowlist,
            pattern=pattern,
            field_name=field,
            prior_value=prior_value,
            config=cfg,
            tighten_debug=tighten_debug,
        )
        tess_elapsed_ms = (time.perf_counter() - tess_start) * 1000.0
        tesseract_debug.update(t_debug)
    if ocr_engine == "tesseract":
        final_winner = tesseract_result
        selected_engine = "tesseract"
        if tesseract_result is not None:
            chosen_pass = "tesseract"
            chosen_method = "fallback_tesseract"
    elif ocr_engine == "both":
        if easyocr_winner is not None and not easyocr_unusable:
            final_winner = easyocr_winner
            selected_engine = "easyocr"
        elif tesseract_result is not None:
            final_winner = tesseract_result
            selected_engine = "tesseract"
            chosen_pass = "tesseract"
            chosen_method = "fallback_tesseract"
    elif tesseract_result is not None:
        final_winner = tesseract_result
        selected_engine = "tesseract"
        chosen_pass = "tesseract"
        chosen_method = "fallback_tesseract"

    ocr_exception = None
    ocr_traceback = None
    if pass_errors:
        ocr_exception = "; ".join(f"{k}: {v.get('exception')}" for k, v in pass_errors.items())
        ocr_traceback = "\n\n".join(str(v.get("traceback", "")) for v in pass_errors.values())

    debug_payload = {
        "preprocess_passes_tried": attempted_passes,
        "chosen_pass": chosen_pass,
        "chosen_method": chosen_method,
        "raw_easyocr": raw_easyocr,
        "pass_errors": pass_errors,
        "boundingRect": tighten_debug.get("boundingRect"),
        "black_pixel_count": tighten_debug.get("black_pixel_count"),
        "tightened": bool(tighten_debug.get("tightened", False)),
        "allowlist": allowlist,
        "regex": pattern.pattern,
        "exception": ocr_exception,
        "traceback": ocr_traceback,
        "tesseract_used": bool(tesseract_debug.get("tesseract_used", False)),
        "tesseract_text": tesseract_debug.get("tesseract_text", ""),
        "tesseract_raw": tesseract_debug.get("tesseract_raw", ""),
        "tesseract_config": tesseract_debug.get("tesseract_config"),
        "tesseract_exception": tesseract_debug.get("tesseract_exception"),
        "tesseract_psm": tesseract_result.get("psm") if isinstance(tesseract_result, dict) else None,
        "tesseract_oem": tesseract_result.get("oem") if isinstance(tesseract_result, dict) else None,
        "tesseract_whitelist": tesseract_result.get("whitelist") if isinstance(tesseract_result, dict) else None,
        "easyocr_candidate": easyocr_winner,
        "selected_engine": selected_engine,
    }
    debug_images = {
        "raw": img,
        "tight": subimg,
        "pre": preprocessed_image,
        "tess": preprocessed_image,
        "passes": pass_image_map,
    }

    if final_winner is not None and final_winner.get("value") is not None:
        return {
            "text": final_winner["text"],
            "value": final_winner["value"],
            "conf": final_winner["conf"],
            "method": final_winner["method"],
            "ocr_pass": final_winner["ocr_pass"],
            "imputed": False,
            "selected_engine": selected_engine,
            "easyocr": easyocr_winner,
            "tesseract": tesseract_result,
            "easy_elapsed_ms": easy_elapsed_ms,
            "tess_elapsed_ms": tess_elapsed_ms,
            "debug": debug_payload,
            "debug_images": debug_images,
        }

    if prior_value is not None:
        return {
            "text": str(prior_value),
            "value": prior_value,
            "conf": None,
            "method": "hold_last",
            "ocr_pass": "none",
            "imputed": True,
            "selected_engine": selected_engine,
            "easyocr": easyocr_winner,
            "tesseract": tesseract_result,
            "easy_elapsed_ms": easy_elapsed_ms,
            "tess_elapsed_ms": tess_elapsed_ms,
            "debug": debug_payload,
            "debug_images": debug_images,
        }

    return {
        "text": "",
        "value": None,
        "conf": None,
        "method": "no_match",
        "ocr_pass": "none",
        "imputed": False,
        "selected_engine": selected_engine,
        "easyocr": easyocr_winner,
        "tesseract": tesseract_result,
        "easy_elapsed_ms": easy_elapsed_ms,
        "tess_elapsed_ms": tess_elapsed_ms,
        "debug": debug_payload,
        "debug_images": debug_images,
    }


def _raw_easyocr_all_empty(raw_easyocr: Any) -> bool:
    if not isinstance(raw_easyocr, dict) or not raw_easyocr:
        return False
    return all(isinstance(rows, list) and len(rows) == 0 for rows in raw_easyocr.values())


def should_save_failed_roi(
    ocr_result: dict[str, Any],
    text: str,
    value: Any,
    *,
    confidence_threshold: float,
) -> tuple[bool, str]:
    if text == "" or value is None:
        return True, "empty"

    method = str(ocr_result.get("method") or "")
    if method in {"no_match", "fallback_tesseract"}:
        return True, method

    conf_value = safe_float(ocr_result.get("conf"))
    conf = conf_value if conf_value is not None else 0.0
    if conf < float(confidence_threshold):
        return True, "low_conf"

    debug = ocr_result.get("debug") if isinstance(ocr_result.get("debug"), dict) else {}
    if debug.get("exception"):
        return True, "exception"
    if _raw_easyocr_all_empty(debug.get("raw_easyocr")):
        return True, "parse_fail"
    return False, ""


def _safe_write_image(path: Path, image: np.ndarray) -> bool:
    try:
        ok = cv2.imwrite(str(path), image)
        if not ok:
            print(f"[WARN] failed to write image path={path}", file=sys.stderr)
        return bool(ok)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] failed to write image path={path} err={exc}", file=sys.stderr)
        return False


def _show_failed_roi_popup(image: np.ndarray, *, bed: str, field: str) -> None:
    window_name = f"failed_roi_{bed}_{field}"
    try:
        cv2.imshow(window_name, image)
        cv2.waitKey(500)
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] debug popup failed bed={bed} field={field} err={exc}", file=sys.stderr)


def save_failed_roi_artifacts(
    fail_roi_dir: Path,
    *,
    timestamp: str,
    image_basename: str,
    image_path: str | None,
    bed: str,
    field: str,
    reason: str,
    roi_coords: tuple[int, int, int, int],
    ocr_result: dict[str, Any],
    debug_popup_failed_roi: bool,
    elapsed_ms: float,
) -> list[str]:
    fail_roi_dir.mkdir(parents=True, exist_ok=True)
    safe_reason = re.sub(r"[^A-Za-z0-9_.-]", "_", reason)
    prefix = f"{image_basename}_{bed}_{field}"
    debug_images = ocr_result.get("debug_images") if isinstance(ocr_result.get("debug_images"), dict) else {}
    debug_meta = ocr_result.get("debug") if isinstance(ocr_result.get("debug"), dict) else {}
    saved_files: list[str] = []

    paths: dict[str, Any] = {"raw": None, "passes": {}, "meta": None}

    raw_img = debug_images.get("raw")
    if isinstance(raw_img, np.ndarray) and raw_img.size > 0:
        raw_path = fail_roi_dir / f"{prefix}_raw.png"
        if _safe_write_image(raw_path, raw_img):
            paths["raw"] = raw_path.as_posix()
            saved_files.append(raw_path.name)
        if debug_popup_failed_roi:
            _show_failed_roi_popup(raw_img, bed=bed, field=field)

    pass_images = debug_images.get("passes") if isinstance(debug_images.get("passes"), dict) else {}
    tess_img = debug_images.get("tess")
    if isinstance(tess_img, np.ndarray) and tess_img.size > 0 and bool(debug_meta.get("tesseract_used", False)):
        tess_path = fail_roi_dir / f"{prefix}_tess.png"
        if _safe_write_image(tess_path, tess_img):
            paths["passes"]["tess"] = tess_path.as_posix()
            saved_files.append(tess_path.name)

    for pass_name, pass_img in pass_images.items():
        if not isinstance(pass_img, np.ndarray) or pass_img.size == 0:
            continue
        pass_path = fail_roi_dir / f"{prefix}_pre_{pass_name}.png"
        if _safe_write_image(pass_path, pass_img):
            paths["passes"][pass_name] = pass_path.as_posix()
            saved_files.append(pass_path.name)

    meta_path = fail_roi_dir / f"{prefix}_meta.json"
    paths["meta"] = meta_path.as_posix()

    meta = {
        "timestamp": timestamp,
        "image_path": image_path,
        "bed": bed,
        "field": field,
        "reason": safe_reason,
        "roi_coords": [int(v) for v in roi_coords],
        "easyocr_result": {
            "text": (ocr_result.get("easyocr") or {}).get("text", "") if isinstance(ocr_result.get("easyocr"), dict) else "",
            "value": (ocr_result.get("easyocr") or {}).get("value") if isinstance(ocr_result.get("easyocr"), dict) else None,
            "conf": (ocr_result.get("easyocr") or {}).get("conf") if isinstance(ocr_result.get("easyocr"), dict) else None,
            "pass": (ocr_result.get("easyocr") or {}).get("ocr_pass") if isinstance(ocr_result.get("easyocr"), dict) else None,
        },
        "tesseract_result": {
            "text": (ocr_result.get("tesseract") or {}).get("text", "") if isinstance(ocr_result.get("tesseract"), dict) else "",
            "value": (ocr_result.get("tesseract") or {}).get("value") if isinstance(ocr_result.get("tesseract"), dict) else None,
            "conf": None,
            "psm": debug_meta.get("tesseract_psm"),
            "oem": debug_meta.get("tesseract_oem"),
            "whitelist": debug_meta.get("tesseract_whitelist"),
        },
        "selected_result": {
            "engine": ocr_result.get("selected_engine", "easyocr"),
            "text": ocr_result.get("text", ""),
            "value": ocr_result.get("value"),
            "conf": ocr_result.get("conf"),
            "ocr_method": ocr_result.get("method", "none"),
            "ocr_pass": ocr_result.get("ocr_pass", "none"),
        },
        "elapsed_ms": elapsed_ms,
        "preprocess_passes_tried": debug_meta.get("preprocess_passes_tried", []),
        "chosen_pass": debug_meta.get("chosen_pass", "none"),
        "chosen_method": debug_meta.get("chosen_method", "none"),
        "raw_easyocr": debug_meta.get("raw_easyocr", {}),
        "pass_errors": debug_meta.get("pass_errors", {}),
        "exception": debug_meta.get("exception"),
        "traceback": debug_meta.get("traceback"),
        "boundingRect": debug_meta.get("boundingRect"),
        "black_pixel_count": debug_meta.get("black_pixel_count"),
        "tightened": debug_meta.get("tightened", False),
        "allowlist": debug_meta.get("allowlist"),
        "regex": debug_meta.get("regex"),
        "tesseract_used": debug_meta.get("tesseract_used", False),
        "tesseract_text": debug_meta.get("tesseract_text", ""),
        "tesseract_raw": debug_meta.get("tesseract_raw", ""),
        "tesseract_config": debug_meta.get("tesseract_config"),
        "tesseract_exception": debug_meta.get("tesseract_exception"),
        "paths": paths,
    }
    try:
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        saved_files.append(meta_path.name)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] failed to write metadata path={meta_path} err={exc}", file=sys.stderr)
    return saved_files

def _sanitize_numeric_text(text: str) -> str:
    sanitized = "".join(ch for ch in str(text) if ch in ALLOWLIST)
    if sanitized.count(".") > 1:
        head, *tail = sanitized.split(".")
        sanitized = head + "." + "".join(tail)
    return sanitized


def _is_in_range(value: float, vital_name: str, config: dict[str, Any]) -> bool:
    ranges = config.get("vital_ranges", {}) if isinstance(config.get("vital_ranges"), dict) else {}
    bounds = ranges.get(vital_name, DEFAULT_VITAL_RANGES.get(vital_name))
    if not bounds:
        return True
    try:
        low, high = float(bounds[0]), float(bounds[1])
    except Exception:
        return True
    return low <= value <= high


def parse_numeric(text: str, vital_name: str, config: dict[str, Any]) -> float | None:
    sanitized = _sanitize_numeric_text(text.strip())
    if not sanitized:
        return None

    candidate_texts = [sanitized]
    if vital_name in TEMPERATURE_VITALS and "." not in sanitized and sanitized.isdigit() and len(sanitized) == 3:
        candidate_texts.insert(0, f"{sanitized[:2]}.{sanitized[2]}")

    for candidate in candidate_texts:
        if NUMERIC_RE.match(candidate) is None:
            continue
        try:
            value = float(candidate)
        except ValueError:
            continue

        if vital_name in INTEGER_PREFERRED_VITALS:
            value = float(int(round(value)))

        if _is_in_range(value, vital_name, config):
            return value
    return None


def ocr_one_roi(reader: easyocr.Reader, roi_crop_final: np.ndarray) -> tuple[str, float, list[np.ndarray], list[tuple[np.ndarray, str, float]]]:
    """Run OCR against the final ROI image only."""
    best_text, best_conf = "", -1.0
    best_bbox_local: list[np.ndarray] = []

    # IMPORTANT: EasyOCR input is always roi_crop_final only.
    results = reader.readtext(roi_crop_final, detail=1, paragraph=False, allowlist=ALLOWLIST)
    for bbox, text, conf in results:
        filtered = _sanitize_numeric_text(text)
        if not filtered:
            continue
        try:
            bbox_array = np.asarray(bbox, dtype=np.int32)
        except Exception:
            bbox_array = np.zeros((0, 2), dtype=np.int32)
        if conf > best_conf:
            best_text, best_conf = filtered, float(conf)
            best_bbox_local = [bbox_array]

    if best_conf < 0:
        return "", 0.0, [], []
    return best_text, best_conf, best_bbox_local, results


def _get_ocr_variants(config: dict[str, Any]) -> list[str]:
    pp = config.get("preprocess", {}) if isinstance(config.get("preprocess"), dict) else {}
    variants_cfg = pp.get("ocr_variants", ["normal", "invert", "otsu"])
    if not isinstance(variants_cfg, list):
        return ["normal", "invert", "otsu"]
    variants: list[str] = []
    for item in variants_cfg:
        name = str(item).strip().lower()
        if not name:
            continue
        if name not in variants:
            variants.append(name)
    return variants or ["normal"]


def ocr_with_fallback_variants(
    reader: easyocr.Reader,
    roi_crop_raw: np.ndarray,
    vital_name: str,
    config: dict[str, Any],
) -> tuple[str, float | None, float, list[np.ndarray], str]:
    best_any_text, best_any_conf = "", -1.0
    best_any_bbox: list[np.ndarray] = []
    best_any_variant = "normal"

    best_valid_text, best_valid_conf = "", -1.0
    best_valid_value: float | None = None
    best_valid_bbox: list[np.ndarray] = []
    best_valid_variant = "normal"

    for variant in _get_ocr_variants(config):
        roi_crop_final = preprocess_roi(roi_crop_raw, vital_name, config, variant=variant)
        if roi_crop_final.size == 0:
            continue

        text, conf, bbox_local, _ocr_results = ocr_one_roi(reader, roi_crop_final)
        value = parse_numeric(text, vital_name, config)

        if conf > best_any_conf:
            best_any_text, best_any_conf = text, conf
            best_any_bbox = bbox_local
            best_any_variant = variant

        if value is not None and conf > best_valid_conf:
            best_valid_text, best_valid_conf = text, conf
            best_valid_value = value
            best_valid_bbox = bbox_local
            best_valid_variant = variant

    if best_valid_value is not None:
        return best_valid_text, best_valid_value, best_valid_conf, best_valid_bbox, best_valid_variant

    if best_any_conf < 0:
        return "", None, 0.0, [], "normal"
    return best_any_text, None, best_any_conf, best_any_bbox, best_any_variant


def draw_ocr_bbox_overlay(roi_crop_final: np.ndarray, ocr_results: list[tuple[np.ndarray, str, float]]) -> np.ndarray:
    overlay = roi_crop_final.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    for bbox, _text, _conf in ocr_results:
        try:
            bbox_array = np.asarray(bbox, dtype=np.int32)
        except Exception:
            continue
        if bbox_array.size == 0:
            continue
        cv2.polylines(overlay, [bbox_array.reshape(-1, 1, 2)], True, (0, 255, 255), 1)
    return overlay


def get_preprocess_geometry(vital_name: str, config: dict[str, Any], roi_crop_raw: np.ndarray) -> tuple[float, int]:
    """Return (resize_scale, trim_px_after_resize) used by preprocess_roi for coordinate mapping."""
    pp = config.get("preprocess", {}) if isinstance(config.get("preprocess"), dict) else {}
    resize = float(pp.get("scale", pp.get("resize", 4.0)))
    if vital_name in TEMPERATURE_VITALS:
        resize = max(resize, 4.0)
    scale = resize if resize > 1.0 else 1.0

    resized_h = int(round(roi_crop_raw.shape[0] * scale))
    resized_w = int(round(roi_crop_raw.shape[1] * scale))

    trim_px = int(pp.get("trim_px", 3))
    if vital_name in TEMPERATURE_VITALS:
        trim_px = max(trim_px, 2)
    max_trim = max(min(resized_h, resized_w) // 4, 0)
    trim_px = clamp(trim_px, 0, max_trim)
    return scale, trim_px


def map_bbox_to_raw_roi(
    bbox_local_final: np.ndarray,
    scale: float,
    trim_px: int,
    raw_shape: tuple[int, int],
) -> np.ndarray:
    """Map bbox from roi_crop_final coordinates back into roi_crop_raw coordinates."""
    if bbox_local_final.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    raw_h, raw_w = raw_shape
    bbox = bbox_local_final.astype(np.float32)
    bbox[:, 0] = (bbox[:, 0] + float(trim_px)) / max(scale, 1e-6)
    bbox[:, 1] = (bbox[:, 1] + float(trim_px)) / max(scale, 1e-6)
    bbox[:, 0] = np.clip(bbox[:, 0], 0, max(raw_w - 1, 0))
    bbox[:, 1] = np.clip(bbox[:, 1], 0, max(raw_h - 1, 0))
    return np.rint(bbox).astype(np.int32)


def crop_red_roi(frame: np.ndarray, roi_box: tuple[int, int, int, int], bed: str, vital: str) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None]:
    frame_h, frame_w = frame.shape[:2]
    vx1, vy1, vx2, vy2 = (int(roi_box[0]), int(roi_box[1]), int(roi_box[2]), int(roi_box[3]))

    if vx1 >= vx2 or vy1 >= vy2:
        print(f"[WARN] invalid red ROI shape, skip {bed}.{vital}: ({vx1}, {vy1}, {vx2}, {vy2})", file=sys.stderr)
        return None, None
    if vx1 < 0 or vy1 < 0 or vx2 > frame_w or vy2 > frame_h:
        print(
            f"[WARN] red ROI out of capture-image bounds, skip {bed}.{vital}: ({vx1}, {vy1}, {vx2}, {vy2}) frame=({frame_w}x{frame_h})",
            file=sys.stderr,
        )
        return None, None

    roi_crop = frame[vy1:vy2, vx1:vx2].copy()
    if roi_crop.size == 0:
        print(f"[WARN] empty red ROI crop, skip {bed}.{vital}: ({vx1}, {vy1}, {vx2}, {vy2})", file=sys.stderr)
        return None, None
    return roi_crop, (vx1, vy1, vx2, vy2)


def inset_roi_box(
    roi_box: tuple[int, int, int, int],
    inset_px: int,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = (int(roi_box[0]), int(roi_box[1]), int(roi_box[2]), int(roi_box[3]))
    inset = max(int(inset_px), 0)
    nx1 = clamp(x1 + inset, 0, max(frame_w - 1, 0))
    ny1 = clamp(y1 + inset, 0, max(frame_h - 1, 0))
    nx2 = clamp(x2 - inset, 1, frame_w)
    ny2 = clamp(y2 - inset, 1, frame_h)
    if nx1 >= nx2 or ny1 >= ny2:
        return None
    return nx1, ny1, nx2, ny2


def apply_inset_to_roi_map(
    roi_map: dict[str, dict[str, tuple[int, int, int, int]]],
    frame_shape: tuple[int, int, int],
    inset_px: int,
) -> dict[str, dict[str, tuple[int, int, int, int]]]:
    frame_h, frame_w = frame_shape[:2]
    adjusted: dict[str, dict[str, tuple[int, int, int, int]]] = {bed: {} for bed in BED_IDS}
    for bed in BED_IDS:
        for vital in VITAL_ORDER:
            box = roi_map.get(bed, {}).get(vital)
            if box is None:
                continue
            inset_box = inset_roi_box(box, inset_px, frame_w, frame_h)
            if inset_box is None:
                continue
            adjusted[bed][vital] = inset_box
    return adjusted


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


def detect_bed_grid_top(bed_img: np.ndarray, fallback_ratio: float = 0.22) -> tuple[int, int | None, float]:
    """Detect the header bottom horizontal line inside one bed image.

    Returns:
        grid_top: y offset where 5x4 vital grid starts (relative to bed image)
        header_bottom_line_y: detected line y, or None when fallback is used
        line_ratio: black-pixel ratio at detected row (0.0 on fallback)
    """
    bed_h, bed_w = bed_img.shape[:2]
    if bed_h <= 2 or bed_w <= 2:
        fallback_top = clamp(int(round(bed_h * fallback_ratio)), 0, max(bed_h - 1, 0))
        return fallback_top, None, 0.0

    gray = cv2.cvtColor(bed_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    y_start = clamp(int(bed_h * 0.10), 0, bed_h - 1)
    y_end = clamp(int(bed_h * 0.40), y_start + 1, bed_h)

    black = (binary == 0).astype(np.uint8)
    row_black_ratio = black[y_start:y_end].sum(axis=1) / float(bed_w)
    best_idx = int(np.argmax(row_black_ratio))
    best_ratio = float(row_black_ratio[best_idx])
    detected_y = y_start + best_idx

    min_line_ratio = 0.08
    if best_ratio < min_line_ratio:
        fallback_top = clamp(int(round(bed_h * fallback_ratio)), 0, bed_h - 1)
        return fallback_top, None, best_ratio

    grid_top = clamp(detected_y + 2, 0, bed_h - 1)
    return grid_top, detected_y, best_ratio


def _cluster_line_positions(indices: np.ndarray, projection: np.ndarray) -> list[int]:
    if indices.size == 0:
        return []

    lines: list[int] = []
    group_start = int(indices[0])
    prev = int(indices[0])

    for raw_idx in indices[1:]:
        idx = int(raw_idx)
        if idx == prev + 1:
            prev = idx
            continue

        segment = np.arange(group_start, prev + 1)
        weights = projection[group_start : prev + 1] + 1e-6
        center = int(round(np.average(segment, weights=weights)))
        lines.append(center)
        group_start = idx
        prev = idx

    segment = np.arange(group_start, prev + 1)
    weights = projection[group_start : prev + 1] + 1e-6
    center = int(round(np.average(segment, weights=weights)))
    lines.append(center)
    return lines


def _fit_expected_lines(lines: list[int], expected_count: int, start: int, end: int) -> list[int] | None:
    if len(lines) < expected_count:
        return None
    if expected_count <= 1:
        return [clamp(lines[0], start, end)]

    sorted_lines = sorted(clamp(v, start, end) for v in lines)
    if len(sorted_lines) == expected_count:
        return sorted_lines

    targets = np.linspace(start, end, expected_count)
    selected: list[int] = []
    cursor = 0
    for target in targets:
        best_idx: int | None = None
        best_dist = None
        for idx in range(cursor, len(sorted_lines)):
            dist = abs(sorted_lines[idx] - target)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = idx
            elif sorted_lines[idx] > target and best_dist is not None:
                break
        if best_idx is None:
            return None
        cursor = best_idx + 1
        selected.append(sorted_lines[best_idx])

    if len(selected) != expected_count:
        return None

    # enforce strictly increasing order by 1px minimum separation
    for idx in range(1, len(selected)):
        if selected[idx] <= selected[idx - 1]:
            selected[idx] = min(end, selected[idx - 1] + 1)
    if selected[-1] > end:
        return None
    return selected


def detect_grid_lines(img: np.ndarray) -> dict[str, Any]:
    """Detect thick horizontal/vertical grid lines from black table borders."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive_inv = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7,
    )
    line_src = cv2.bitwise_or(otsu_inv, adaptive_inv)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 20), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 20)))
    horizontal = cv2.morphologyEx(line_src, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(line_src, cv2.MORPH_OPEN, vertical_kernel)

    x_projection = vertical.sum(axis=0) / 255.0
    y_projection = horizontal.sum(axis=1) / 255.0
    x_threshold = max(float(h) * 0.18, float(x_projection.max()) * 0.35 if x_projection.size else 0.0)
    y_threshold = max(float(w) * 0.18, float(y_projection.max()) * 0.35 if y_projection.size else 0.0)

    x_indices = np.where(x_projection >= x_threshold)[0]
    y_indices = np.where(y_projection >= y_threshold)[0]

    x_lines = _cluster_line_positions(x_indices, x_projection)
    y_lines = _cluster_line_positions(y_indices, y_projection)
    return {
        "x_lines": sorted(x_lines),
        "y_lines": sorted(y_lines),
        "horizontal_mask": horizontal,
        "vertical_mask": vertical,
        "x_threshold": float(x_threshold),
        "y_threshold": float(y_threshold),
    }


def detect_bed_panels(
    frame: np.ndarray,
    header_crop_px: int,
    bed_cols: int,
    bed_rows: int,
) -> tuple[dict[str, tuple[int, int, int, int]], dict[str, Any], bool]:
    frame_h, frame_w = frame.shape[:2]
    body_top = clamp(header_crop_px, 0, frame_h - 1)
    body_img = frame[body_top:frame_h, :]
    body_h = max(frame_h - body_top, 1)

    detected = detect_grid_lines(body_img)
    fit_x = _fit_expected_lines(detected["x_lines"], bed_cols + 1, 0, max(frame_w - 1, 0))
    fit_y = _fit_expected_lines(detected["y_lines"], bed_rows + 1, 0, max(body_h - 1, 0))

    panels: dict[str, tuple[int, int, int, int]] = {}
    used_detected = fit_x is not None and fit_y is not None
    if used_detected:
        assert fit_x is not None
        assert fit_y is not None
        for bed_idx, bed in enumerate(BED_IDS):
            bed_row = bed_idx // bed_cols
            bed_col = bed_idx % bed_cols
            bx1 = fit_x[bed_col]
            bx2 = fit_x[bed_col + 1]
            by1 = body_top + fit_y[bed_row]
            by2 = body_top + fit_y[bed_row + 1]
            panels[bed] = (bx1, by1, max(bx1 + 1, bx2), max(by1 + 1, by2))
    else:
        for bed_idx, bed in enumerate(BED_IDS):
            bed_row = bed_idx // bed_cols
            bed_col = bed_idx % bed_cols
            bx1 = int((bed_col * frame_w) / bed_cols)
            bx2 = int(((bed_col + 1) * frame_w) / bed_cols)
            by1 = body_top + int((bed_row * body_h) / bed_rows)
            by2 = body_top + int(((bed_row + 1) * body_h) / bed_rows)
            panels[bed] = (bx1, by1, bx2, by2)

    debug = {
        "x_lines": detected["x_lines"],
        "y_lines": [body_top + y for y in detected["y_lines"]],
        "fit_x": fit_x,
        "fit_y": [body_top + y for y in fit_y] if fit_y is not None else None,
        "used_detected": used_detected,
    }
    return panels, debug, used_detected


def detect_cells_in_bed(
    frame: np.ndarray,
    bed_rect: tuple[int, int, int, int],
    cell_cols: int,
    cell_rows: int,
) -> tuple[list[tuple[int, int, int, int]] | None, dict[str, Any], bool]:
    bx1, by1, bx2, by2 = bed_rect
    bed_img = frame[by1:by2, bx1:bx2]
    bed_h, bed_w = bed_img.shape[:2]
    if bed_h < 2 or bed_w < 2:
        return None, {"used_detected": False}, False

    grid_top, detected_line_y, line_ratio = detect_bed_grid_top(bed_img)
    grid_roi = bed_img[grid_top:bed_h, :]
    detected = detect_grid_lines(grid_roi)
    fit_x = _fit_expected_lines(detected["x_lines"], cell_cols + 1, 0, max(bed_w - 1, 0))
    fit_y = _fit_expected_lines(detected["y_lines"], cell_rows + 1, 0, max(grid_roi.shape[0] - 1, 0))

    used_detected = fit_x is not None and fit_y is not None
    cells: list[tuple[int, int, int, int]] | None = None
    if used_detected:
        assert fit_x is not None
        assert fit_y is not None
        cells = []
        for row in range(cell_rows):
            for col in range(cell_cols):
                x1 = bx1 + fit_x[col]
                x2 = bx1 + fit_x[col + 1]
                y1 = by1 + grid_top + fit_y[row]
                y2 = by1 + grid_top + fit_y[row + 1]
                cells.append((x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)))

    debug = {
        "grid_top": grid_top,
        "header_bottom_line_y": (by1 + detected_line_y) if detected_line_y is not None else None,
        "line_ratio": line_ratio,
        "x_lines": [bx1 + x for x in detected["x_lines"]],
        "y_lines": [by1 + grid_top + y for y in detected["y_lines"]],
        "fit_x": [bx1 + x for x in fit_x] if fit_x is not None else None,
        "fit_y": [by1 + grid_top + y for y in fit_y] if fit_y is not None else None,
        "used_detected": used_detected,
    }
    return cells, debug, used_detected


def build_vital_rois(
    frame: np.ndarray,
    config: dict[str, Any],
    return_debug: bool = False,
) -> dict[str, dict[str, tuple[int, int, int, int]]] | tuple[
    dict[str, dict[str, tuple[int, int, int, int]]], dict[str, dict[str, Any]]
]:
    frame_h, frame_w = frame.shape[:2]
    header_crop_px = clamp(int(config.get("header_crop_px", 60)), 0, frame_h - 1)

    bed_grid = config.get("bed_grid", {})
    cell_grid = config.get("cell_grid", {})
    bed_cols, bed_rows = int(bed_grid.get("cols", 2)), int(bed_grid.get("rows", 3))
    cell_cols, cell_rows = int(cell_grid.get("cols", 4)), int(cell_grid.get("rows", 5))

    roi_cfg = config.get("value_roi", {})
    left_ratio = float(roi_cfg.get("left_ratio", 0.45))
    right_pad = float(roi_cfg.get("right_pad", 0.06))
    top_pad = float(roi_cfg.get("top_pad", 0.10))
    bottom_pad = float(roi_cfg.get("bottom_pad", 0.12))

    cell_inner_box_cfg = config.get("cell_inner_box") if isinstance(config.get("cell_inner_box"), dict) else {}
    cell_value_slice_cfg = config.get("cell_value_slice") if isinstance(config.get("cell_value_slice"), dict) else {}
    value_box_cfg = config.get("value_box") if isinstance(config.get("value_box"), dict) else None

    cell_pad_px = max(int(cell_inner_box_cfg.get("pad_px", 4)), 0)
    value_x1_ratio = float(cell_value_slice_cfg.get("x1_ratio", 0.60))
    value_x2_ratio = float(cell_value_slice_cfg.get("x2_ratio", 0.98))
    value_y1_ratio = float(cell_value_slice_cfg.get("y1_ratio", 0.12))
    value_y2_ratio = float(cell_value_slice_cfg.get("y2_ratio", 0.90))

    adjust_cfg = config.get("per_vital_roi_adjust", {})

    out: dict[str, dict[str, tuple[int, int, int, int]]] = {bed: {} for bed in BED_IDS}
    debug_meta: dict[str, dict[str, Any]] = {}

    bed_panels, panel_debug, panel_detected = detect_bed_panels(frame, header_crop_px, bed_cols, bed_rows)
    print(
        "[INFO] detect_bed_panels "
        f"used_detected={panel_detected} "
        f"x_lines={len(panel_debug.get('x_lines', []))} y_lines={len(panel_debug.get('y_lines', []))} "
        f"fit_x={panel_debug.get('fit_x')} fit_y={panel_debug.get('fit_y')}"
    )

    for bed in BED_IDS:
        bx1, by1, bx2, by2 = bed_panels[bed]

        bed_w = max(bx2 - bx1, 1)
        bed_h = max(by2 - by1, 1)

        cells, bed_line_debug, bed_detected = detect_cells_in_bed(frame, (bx1, by1, bx2, by2), cell_cols, cell_rows)
        print(
            f"[INFO] {bed} line_detect used_detected={bed_detected} "
            f"x_lines={len(bed_line_debug.get('x_lines', []))} y_lines={len(bed_line_debug.get('y_lines', []))} "
            f"fit_x={bed_line_debug.get('fit_x')} fit_y={bed_line_debug.get('fit_y')}"
        )

        grid_top = int(bed_line_debug.get("grid_top", 0))
        grid_y1 = by1 + grid_top
        grid_y2 = by2
        grid_h = max(grid_y2 - grid_y1, 1)

        debug_meta[bed] = {
            "bed_x1": bx1,
            "bed_y1": by1,
            "bed_x2": bx2,
            "bed_y2": by2,
            "grid_top": grid_top,
            "grid_y1": grid_y1,
            "grid_y2": grid_y2,
            "header_bottom_line_y": bed_line_debug.get("header_bottom_line_y"),
            "line_ratio": float(bed_line_debug.get("line_ratio", 0.0)),
            "header_line_detected": bed_line_debug.get("header_bottom_line_y") is not None,
            "used_detected_lines": bed_detected,
            "detected_x_lines": bed_line_debug.get("x_lines", []),
            "detected_y_lines": bed_line_debug.get("y_lines", []),
            "fit_x_lines": bed_line_debug.get("fit_x", []),
            "fit_y_lines": bed_line_debug.get("fit_y", []),
            "cell_inner_boxes": {},
            "cell_boxes": {},
            "value_rois": {},
        }

        for idx, vital in enumerate(VITAL_ORDER):
            c_row = idx // cell_cols
            c_col = idx % cell_cols

            if cells is not None:
                cx1, cy1, cx2, cy2 = cells[idx]
            else:
                cx1 = bx1 + int((c_col * bed_w) / cell_cols)
                cx2 = bx1 + int(((c_col + 1) * bed_w) / cell_cols)
                cy1 = grid_y1 + int((c_row * grid_h) / cell_rows)
                cy2 = grid_y1 + int(((c_row + 1) * grid_h) / cell_rows)

            cw, ch = max(cx2 - cx1, 1), max(cy2 - cy1, 1)

            if value_box_cfg is None and not cell_value_slice_cfg and not cell_inner_box_cfg:
                vx1 = clamp(cx1 + int(cw * left_ratio), cx1, cx2 - 1)
                vx2 = clamp(cx2 - int(cw * right_pad), vx1 + 1, cx2)
                vy1 = clamp(cy1 + int(ch * top_pad), cy1, cy2 - 1)
                vy2 = clamp(cy2 - int(ch * bottom_pad), vy1 + 1, cy2)
                cell_x1, cell_y1, cell_x2, cell_y2 = cx1, cy1, cx2, cy2
            else:
                cell_x1 = clamp(cx1 + cell_pad_px, cx1, cx2 - 1)
                cell_y1 = clamp(cy1 + cell_pad_px, cy1, cy2 - 1)
                cell_x2 = clamp(cx2 - cell_pad_px, cell_x1 + 1, cx2)
                cell_y2 = clamp(cy2 - cell_pad_px, cell_y1 + 1, cy2)

                inner_w = max(cell_x2 - cell_x1, 1)
                inner_h = max(cell_y2 - cell_y1, 1)

                slice_x1_ratio = min(value_x1_ratio, value_x2_ratio)
                slice_x2_ratio = max(value_x1_ratio, value_x2_ratio)
                slice_y1_ratio = min(value_y1_ratio, value_y2_ratio)
                slice_y2_ratio = max(value_y1_ratio, value_y2_ratio)

                vx1 = cell_x1 + int(inner_w * slice_x1_ratio)
                vx2 = cell_x1 + int(inner_w * slice_x2_ratio)
                vy1 = cell_y1 + int(inner_h * slice_y1_ratio)
                vy2 = cell_y1 + int(inner_h * slice_y2_ratio)

                vx1 = clamp(vx1, cell_x1, cell_x2 - 1)
                vy1 = clamp(vy1, cell_y1, cell_y2 - 1)
                vx2 = clamp(vx2, vx1 + 1, cell_x2)
                vy2 = clamp(vy2, vy1 + 1, cell_y2)

            vital_adjust = adjust_cfg.get(vital) if isinstance(adjust_cfg, dict) else None
            if isinstance(vital_adjust, dict):
                vx1 = clamp(vx1 + _offset_from_adjust(vital_adjust, "dx1", cw, ch), cx1, cx2 - 1)
                vy1 = clamp(vy1 + _offset_from_adjust(vital_adjust, "dy1", cw, ch), cy1, cy2 - 1)
                vx2 = clamp(vx2 + _offset_from_adjust(vital_adjust, "dx2", cw, ch), vx1 + 1, cx2)
                vy2 = clamp(vy2 + _offset_from_adjust(vital_adjust, "dy2", cw, ch), vy1 + 1, cy2)

            out[bed][vital] = (vx1, vy1, vx2, vy2)
            debug_meta[bed]["cell_boxes"][vital] = (cx1, cy1, cx2, cy2)
            debug_meta[bed]["cell_inner_boxes"][vital] = (cell_x1, cell_y1, cell_x2, cell_y2)
            debug_meta[bed]["value_rois"][vital] = (vx1, vy1, vx2, vy2)

    if return_debug:
        return out, debug_meta
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


def parse_cache_snapshot_timestamp(snapshot_path: Path) -> datetime:
    match = CAPTURE_CACHE_FILENAME_PATTERN.match(snapshot_path.name)
    if match:
        day_part, time_part, ms_part = match.groups()
        dt = datetime.strptime(day_part + time_part + ms_part, "%Y%m%d%H%M%S%f")
        return dt.astimezone()
    return datetime.fromtimestamp(snapshot_path.stat().st_mtime).astimezone()


def resolve_cache_snapshot_paths(
    cache_dir: Path,
    capture_dt: datetime,
    nearest_window_sec: float,
    use_nearest_past_fallback: bool,
) -> tuple[str | None, str | None, float | None]:
    if not cache_dir.exists():
        return None, None, None

    candidates: list[tuple[Path, datetime, float]] = []
    for path in cache_dir.glob("*_monitor_cache.json"):
        try:
            ts = parse_cache_snapshot_timestamp(path)
        except OSError:
            continue
        delta = (ts - capture_dt).total_seconds()
        candidates.append((path, ts, delta))

    if not candidates:
        return None, None, None

    nearest_overall = min(candidates, key=lambda item: abs(item[2]))
    selected = nearest_overall
    if abs(nearest_overall[2]) > nearest_window_sec and use_nearest_past_fallback:
        past_candidates = [item for item in candidates if item[1] <= capture_dt]
        if past_candidates:
            selected = max(past_candidates, key=lambda item: item[1])

    before_candidates = [item for item in candidates if item[1] <= capture_dt]
    selected_before = max(before_candidates, key=lambda item: item[1]) if before_candidates else None

    return (
        selected[0].as_posix() if selected else None,
        selected_before[0].as_posix() if selected_before else None,
        float(selected[2]) if selected else None,
    )


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
    target_display: str,
    windows_monitors: list[dict[str, int | bool]],
    display_index: int | None,
    mss_monitor_index: int | None,
    use_gpu: bool,
) -> None:
    import easyocr

    base = choose_capture_region(sct, target_display, windows_monitors, display_index, mss_monitor_index)
    frame = grab_frame(sct, base)
    base_img_h, base_img_w = frame.shape[:2]
    base_scale_x = base_img_w / float(max(base.width, 1))
    base_scale_y = base_img_h / float(max(base.height, 1))
    print(
        "[INFO] calibration base_img "
        f"size={base_img_w}x{base_img_h} monitor_rect={base.width}x{base.height} "
        f"scale_x={base_scale_x:.6f} scale_y={base_scale_y:.6f}"
    )

    selected = cv2.selectROI("Select monitor.py region", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select monitor.py region")
    x, y, w, h = [int(v) for v in selected]
    if w <= 0 or h <= 0:
        raise RuntimeError("Calibration cancelled: monitor_rect was not selected.")

    inv_scale_x = 1.0 / base_scale_x if base_scale_x > 0 else 1.0
    inv_scale_y = 1.0 / base_scale_y if base_scale_y > 0 else 1.0
    if abs(base_scale_x - 1.0) > 1e-6 or abs(base_scale_y - 1.0) > 1e-6:
        print("[WARN] calibration selection will be converted from physical px to logical px")

    logical_x = _scale_value(x, inv_scale_x)
    logical_y = _scale_value(y, inv_scale_y)
    logical_w = max(_scale_value(w, inv_scale_x), 1)
    logical_h = max(_scale_value(h, inv_scale_y), 1)

    config["target_display"] = target_display
    config["monitor_rect"] = {
        "left": base.left + logical_x,
        "top": base.top + logical_y,
        "width": logical_w,
        "height": logical_h,
    }
    capture_region = CaptureRegion(
        left=base.left + logical_x,
        top=base.top + logical_y,
        width=logical_w,
        height=logical_h,
    )
    cropped = normalize_capture_frame(frame, capture_region, base)
    reader = easyocr.Reader(["en"], gpu=use_gpu)

    win = "Calibration (s=save, q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def add_trackbar(name: str, value: int, max_value: int) -> None:
        cv2.createTrackbar(name, win, value, max_value, lambda _v: None)

    add_trackbar("header_crop_px", int(config.get("header_crop_px", 60)), max(1, cropped.shape[0] - 1))
    cell_inner = config.setdefault("cell_inner_box", dict(DEFAULT_CONFIG["cell_inner_box"]))
    cell_slice = config.setdefault("cell_value_slice", dict(DEFAULT_CONFIG["cell_value_slice"]))
    add_trackbar("cell_pad_px", int(cell_inner.get("pad_px", 4)), 40)
    add_trackbar("value_x1_ratio(%)", int(float(cell_slice.get("x1_ratio", 0.60)) * 100), 99)
    add_trackbar("value_x2_ratio(%)", int(float(cell_slice.get("x2_ratio", 0.98)) * 100), 100)
    add_trackbar("value_y1_ratio(%)", int(float(cell_slice.get("y1_ratio", 0.12)) * 100), 99)
    add_trackbar("value_y2_ratio(%)", int(float(cell_slice.get("y2_ratio", 0.90)) * 100), 100)

    while True:
        config["header_crop_px"] = cv2.getTrackbarPos("header_crop_px", win)
        value_x1 = cv2.getTrackbarPos("value_x1_ratio(%)", win)
        value_x2 = cv2.getTrackbarPos("value_x2_ratio(%)", win)
        value_y1 = cv2.getTrackbarPos("value_y1_ratio(%)", win)
        value_y2 = cv2.getTrackbarPos("value_y2_ratio(%)", win)
        config["cell_inner_box"]["pad_px"] = cv2.getTrackbarPos("cell_pad_px", win)
        config["cell_value_slice"]["x1_ratio"] = min(value_x1, value_x2) / 100.0
        config["cell_value_slice"]["x2_ratio"] = max(value_x1, value_x2) / 100.0
        config["cell_value_slice"]["y1_ratio"] = min(value_y1, value_y2) / 100.0
        config["cell_value_slice"]["y2_ratio"] = max(value_y1, value_y2) / 100.0

        rois, roi_debug = build_vital_rois(cropped, config, return_debug=True)
        x1, y1, x2, y2 = rois["BED01"]["RAP_M"]
        ci_x1, ci_y1, ci_x2, ci_y2 = roi_debug["BED01"]["cell_inner_boxes"]["RAP_M"]
        roi_img = cropped[y1:y2, x1:x2]
        roi_final = preprocess_roi(roi_img, "RAP_M", config, variant="normal") if roi_img.size > 0 else roi_img
        text, conf, _best_bbox, _ocr_results = ocr_one_roi(reader, roi_final) if roi_final.size > 0 else ("", 0.0, [], [])

        left = cropped.copy()
        cv2.rectangle(left, (ci_x1, ci_y1), (ci_x2, ci_y2), (255, 0, 0), 2)
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
    import easyocr
    import mss
    import torch

    parser = argparse.ArgumentParser(description="Capture monitor display and OCR all vitals")
    parser.add_argument("--cache", default="monitor_cache.json")
    parser.add_argument("--config", default="ocr_capture_config.json")
    parser.add_argument("--outdir", default="dataset")
    parser.add_argument("--capture-mode", choices=["window", "mss"], default="window")
    parser.add_argument("--roi-source", choices=["exported", "legacy"], default="exported")
    parser.add_argument("--window-title", default="HL7 Bed Monitor")
    parser.add_argument("--capture-client", type=parse_bool, default=True)
    parser.add_argument("--target-display", choices=["primary", "index", "mss"], default="primary")
    parser.add_argument("--display-index", type=int, default=None)
    parser.add_argument("--mss-monitor-index", type=int, default=None)
    parser.add_argument("--auto-detect-monitor", type=parse_bool, default=True)
    parser.add_argument("--auto-detect-debug", type=parse_bool, default=False)
    parser.add_argument("--interval-ms", type=int, default=10000)
    parser.add_argument("--save-images", type=parse_bool, default=True)
    parser.add_argument("--debug-roi", type=parse_bool, default=False)
    parser.add_argument("--debug-full-overlay", type=parse_bool, default=False)
    parser.add_argument("--debug-lines", type=parse_bool, default=False)
    parser.add_argument("--debug-window-rect", type=parse_bool, default=False)
    parser.add_argument("--save-fail-roi", type=parse_bool, default=None)
    parser.add_argument("--fail-roi-dir", default="day_dir/failed_roi")
    parser.add_argument("--fail-roi-fields", default="CVP_M,RAP_M")
    parser.add_argument("--fail-roi-max-per-tick", type=int, default=20)
    parser.add_argument("--tesseract-fallback", type=parse_bool, default=None)
    parser.add_argument("--tesseract-cmd", default="")
    parser.add_argument("--ocr-engine", choices=["easyocr", "tesseract", "both"], default="easyocr")
    parser.add_argument("--tess-fields", default="")
    parser.add_argument("--debug-save-failed-roi", action="store_true")
    parser.add_argument("--save-lowconf-roi-threshold", type=float, default=0.50)
    parser.add_argument("--gpu", type=parse_bool, default=True)
    parser.add_argument("--no-launch-monitor", type=parse_bool, default=True)  # MOD: 安全側デフォルト
    parser.add_argument("--run-validator", type=parse_bool, default=False)  # MOD
    parser.add_argument("--validator-last", type=int, default=50)  # MOD
    parser.add_argument("--validator-config", default="validator_config.json")  # MOD
    parser.add_argument("--cache-nearest-window-sec", type=float, default=5.0)
    parser.add_argument("--cache-nearest-past-fallback", type=parse_bool, default=True)
    parser.add_argument("--calibrate", action="store_true")
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        unsupported_args = [token for token in unknown_args if token != "^"]
        if unsupported_args:
            parser.error(f"unrecognized arguments: {' '.join(unsupported_args)}")
        print("[WARN] ignoring trailing '^' token in command line", file=sys.stderr)

    fail_roi_fields: set[str] | None = None
    fail_roi_fields_raw = str(args.fail_roi_fields or "").strip()
    if fail_roi_fields_raw:
        fail_roi_fields = {item.strip() for item in fail_roi_fields_raw.split(",") if item.strip()}

    tess_fields_raw = str(args.tess_fields or "").strip()
    tess_fields = {item.strip() for item in tess_fields_raw.split(",") if item.strip()} if tess_fields_raw else set()

    dpi_aware()
    log_windows_dpi_info()

    config_path = Path(args.config)
    config = ensure_config(config_path)
    config["window_title"] = args.window_title
    config["ocr_compare_mode"] = parse_bool(os.environ.get("OCR_COMPARE_MODE", "false"))
    print(f"[INFO] OCR_COMPARE_MODE={str(config['ocr_compare_mode']).lower()}")
    if args.tesseract_fallback is not None:
        config["enable_tesseract_fallback"] = bool(args.tesseract_fallback)
    if str(args.tesseract_cmd or "").strip():
        config["tesseract_cmd"] = str(args.tesseract_cmd).strip()
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

    windows_monitors = enum_windows_monitors()

    with mss.mss() as sct:
        log_monitors(sct)
        if args.calibrate:
            if args.capture_mode == "window":
                print("[WARN] calibration currently uses mss capture settings; --capture-mode window is ignored during calibration")
            run_calibration(
                sct,
                config_path,
                config,
                args.target_display,
                windows_monitors,
                args.display_index,
                args.mss_monitor_index,
                use_gpu,
            )
            return

        reader = easyocr.Reader(["en"], gpu=use_gpu)

        mss_monitor_base: CaptureRegion | None = None
        mss_capture_rect: CaptureRegion | None = None
        mss_fallback_base: CaptureRegion | None = None
        mss_fallback_rect: CaptureRegion | None = None
        if args.capture_mode == "mss":
            mss_monitor_base = choose_capture_region(
                sct,
                args.target_display,
                windows_monitors,
                args.display_index,
                args.mss_monitor_index,
            )
            mss_capture_rect = get_monitor_rect(config, mss_monitor_base)

        last_confirmed_values: dict[str, dict[str, float]] = {bed: {} for bed in BED_IDS}
        missing_by_field: dict[str, int] = {vital: 0 for vital in VITAL_ORDER}

        try:
            while True:
                tick = time.time()
                frame_start = time.perf_counter()
                print(f"[INFO] capture tick start ts={datetime.now().isoformat(timespec='seconds')}")
                try:
                    day = datetime.now().strftime("%Y%m%d")
                    day_dir = Path(args.outdir) / day
                    images_dir = day_dir / "images"
                    debug_dir = day_dir / "debug"
                    roi_crops_dir = day_dir / "roi_crops"
                    day_dir.mkdir(parents=True, exist_ok=True)
                    if args.save_images:
                        images_dir.mkdir(parents=True, exist_ok=True)

                    capture_rect: CaptureRegion | None = None
                    frame: np.ndarray | None = None

                    if args.capture_mode == "window":
                        capture_rect = get_window_capture_region(args.window_title, args.capture_client)
                        if capture_rect is None:
                            print(f"[INFO] window found=not found title='{args.window_title}'")
                            if mss_fallback_base is None:
                                selected_index: int | None = None
                                selected_score = 0.0
                                if args.auto_detect_monitor:
                                    selected_index, selected_score, _candidate_scores = auto_detect_monitor(
                                        sct,
                                        debug=bool(args.auto_detect_debug),
                                    )
                                if selected_index is not None:
                                    mon = sct.monitors[selected_index]
                                    mss_fallback_base = CaptureRegion(
                                        int(mon["left"]),
                                        int(mon["top"]),
                                        int(mon["width"]),
                                        int(mon["height"]),
                                    )
                                    print(
                                        "[INFO] auto_detect_monitor selected "
                                        f"mss-monitor-index={selected_index} score={selected_score:.4f}"
                                    )
                                else:
                                    mss_fallback_base = get_primary_mss_monitor_base(sct, windows_monitors)
                                    print(
                                        "[INFO] auto_detect_monitor selected "
                                        "mss-monitor-index=primary-fallback score=0.0000"
                                    )
                                mss_fallback_rect = get_monitor_rect(config, mss_fallback_base)
                            assert mss_fallback_base is not None
                            assert mss_fallback_rect is not None
                            capture_rect = mss_fallback_rect
                            base_frame = grab_frame(sct, mss_fallback_base)
                            frame = normalize_capture_frame(base_frame, capture_rect, mss_fallback_base)
                            print("[WARN] using mss fallback capture because Win32 client rect is unavailable", file=sys.stderr)
                        else:
                            print(f"[INFO] window found=found title='{args.window_title}'")
                            frame = grab_frame(sct, capture_rect)
                    else:
                        assert mss_monitor_base is not None
                        assert mss_capture_rect is not None
                        capture_rect = mss_capture_rect
                        print("[INFO] window found=not applicable capture-mode=mss")
                        base_frame = grab_frame(sct, mss_monitor_base)
                        frame = normalize_capture_frame(base_frame, capture_rect, mss_monitor_base)

                    assert capture_rect is not None
                    assert frame is not None
                    print(
                        "[INFO] capture rect "
                        f"left={capture_rect.left} top={capture_rect.top} width={capture_rect.width} height={capture_rect.height}"
                    )
                    print(f"[INFO] grab image size width={frame.shape[1]} height={frame.shape[0]}")

                    blue_rois, red_rois, roi_debug = build_rois_for_frame(
                        frame=frame,
                        config=config,
                        roi_source=args.roi_source,
                        capture_rect=capture_rect,
                    )
                    roi_inset_px = int(config.get("value_roi_inset_px", 2))
                    if args.roi_source == "exported":
                        # Exported ROIs from monitor.py already point to the value widget,
                        # so avoid shrinking and keep red boxes aligned with displayed numbers.
                        roi_inset_px = 0
                    rect_map = apply_inset_to_roi_map(red_rois, frame.shape, roi_inset_px)
                    capture_dt = datetime.now().astimezone()
                    capture_timestamp = capture_dt.isoformat(timespec="milliseconds")
                    stamp = capture_dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    roi_tick_dir: Path | None = None
                    roi_debug_input_dir: Path | None = None

                    image_path = None
                    if args.save_images:
                        image_path = images_dir / f"{stamp}.png"
                        cv2.imwrite(str(image_path), frame)
                    image_basename = image_path.stem if image_path is not None else stamp

                    cache_snapshot_before = copy_cache_snapshot(cache_path, day_dir, f"{stamp}_before")
                    if not cache_snapshot_before:
                        print(f"[WARN] pre-capture cache snapshot failed at {stamp}", file=sys.stderr)

                    if args.debug_window_rect:
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        write_debug_window_rect_image(debug_dir / f"{stamp}_window_rect.png", frame)

                    debug_img: np.ndarray | None = None
                    full_overlay_img: np.ndarray | None = None
                    if args.debug_roi or args.debug_lines:
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        debug_img = frame.copy()
                        if args.debug_full_overlay:
                            full_overlay_img = frame.copy()
                        for bed in BED_IDS:
                            meta = roi_debug.get(bed, {})
                            bx1 = int(meta.get("bed_x1", 0))
                            by1 = int(meta.get("bed_y1", 0))
                            bx2 = int(meta.get("bed_x2", 0))
                            by2 = int(meta.get("bed_y2", 0))
                            gy1 = int(meta.get("grid_y1", by1))
                            gy2 = int(meta.get("grid_y2", by2))
                            header_line = meta.get("header_bottom_line_y")

                            if args.debug_lines:
                                for line_x in meta.get("detected_x_lines", []):
                                    lx = int(line_x)
                                    cv2.line(debug_img, (lx, by1), (lx, by2 - 1), (0, 180, 0), 1)
                                for line_y in meta.get("detected_y_lines", []):
                                    ly = int(line_y)
                                    cv2.line(debug_img, (bx1, ly), (bx2 - 1, ly), (0, 180, 0), 1)

                            if args.debug_roi:
                                if "bed_x1" in meta:
                                    if header_line is not None:
                                        hy = int(header_line)
                                        cv2.line(debug_img, (bx1, hy), (bx2 - 1, hy), (0, 255, 0), 2)

                                    cv2.rectangle(debug_img, (bx1, gy1), (bx2 - 1, gy2 - 1), (255, 0, 0), 2)
                                    cv2.putText(
                                        debug_img,
                                        f"{bed} grid_top={int(meta.get('grid_top', 0))}",
                                        (bx1 + 3, max(gy1 - 6, 12)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.35,
                                        (255, 255, 0),
                                        1,
                                        cv2.LINE_AA,
                                    )

                                cell_boxes = meta.get("cell_boxes", {})
                                for vital in VITAL_ORDER:
                                    cell = cell_boxes.get(vital)
                                    if isinstance(cell, tuple) and len(cell) == 4:
                                        cx1, cy1, cx2, cy2 = cell
                                        cv2.rectangle(debug_img, (cx1, cy1), (cx2, cy2), (255, 0, 0), 1)

                                    if args.debug_lines:
                                        blue = blue_rois.get(bed, {}).get(vital)
                                        if blue:
                                            x1, y1, x2, y2 = blue
                                            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                                    red = rect_map.get(bed, {}).get(vital)
                                    if red:
                                        x1, y1, x2, y2 = red
                                        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        if args.debug_roi:
                            roi_tick_dir = roi_crops_dir / stamp
                            roi_tick_dir.mkdir(parents=True, exist_ok=True)
                            roi_debug_input_dir = Path(args.outdir) / "roi_debug" / stamp
                            roi_debug_input_dir.mkdir(parents=True, exist_ok=True)

                    save_fail_roi = bool(config.get("debug_save_failed_roi", True))
                    if args.save_fail_roi is not None:
                        save_fail_roi = save_fail_roi and bool(args.save_fail_roi)
                    if args.debug_save_failed_roi:
                        save_fail_roi = True

                    fail_roi_conf_threshold = safe_float(args.save_lowconf_roi_threshold)
                    if fail_roi_conf_threshold is None:
                        fail_roi_conf_threshold = 0.50
                    debug_popup_failed_roi = bool(config.get("debug_popup_failed_roi", False))
                    fail_roi_dir = day_dir / "failed_roi" if args.fail_roi_dir == "day_dir/failed_roi" else Path(args.fail_roi_dir)
                    if save_fail_roi:
                        fail_roi_dir.mkdir(parents=True, exist_ok=True)
                    fail_roi_saved_in_tick = 0
                    beds: dict[str, dict[str, Any]] = {bed: {} for bed in BED_IDS}
                    record_debug: dict[str, dict[str, Any]] = {}
                    for bed in BED_IDS:
                        bed_debug: dict[str, Any] = {}
                        for vital in VITAL_ORDER:
                            roi_box = rect_map.get(bed, {}).get(vital)
                            if roi_box is None:
                                missing_by_field[vital] = missing_by_field.get(vital, 0) + 1
                                beds[bed][vital] = {
                                    "text": "",
                                    "value": None,
                                    "confidence": 0.0,
                                    "ocr_method": "no_roi",
                                    "imputed": False,
                                    "ocr_pass": "none",
                                    "selected_engine": None,
                                    "easy_value": None,
                                    "easy_conf": None,
                                    "tess_value": None,
                                    "tess_conf": None,
                                    "selected_value": None,
                                    "easy_elapsed_ms": None,
                                    "tess_elapsed_ms": None,
                                }
                                continue

                            roi_crop_raw: np.ndarray | None = None
                            roi_coords: tuple[int, int, int, int] | None = None
                            roi_start = time.perf_counter()
                            try:
                                roi_crop_raw, roi_coords = crop_red_roi(frame, roi_box, bed, vital)
                                if roi_crop_raw is None or roi_coords is None:
                                    missing_by_field[vital] = missing_by_field.get(vital, 0) + 1
                                    beds[bed][vital] = {
                                        "text": "",
                                        "value": None,
                                        "confidence": 0.0,
                                        "ocr_method": "crop_failed",
                                        "imputed": False,
                                        "ocr_pass": "none",
                                        "selected_engine": None,
                                        "easy_value": None,
                                        "easy_conf": None,
                                        "tess_value": None,
                                        "tess_conf": None,
                                        "selected_value": None,
                                        "easy_elapsed_ms": None,
                                        "tess_elapsed_ms": None,
                                    }
                                    continue

                                if args.debug_roi and roi_debug_input_dir is not None:
                                    cv2.imwrite(str(roi_debug_input_dir / f"roi_{bed}_{vital}_raw.png"), roi_crop_raw)

                                if vital in {"CVP_M", "RAP_M"}:
                                    print(f"[INFO] fixed_roi_coords bed={bed} field={vital} coords={roi_coords}")

                                roi_for_ocr = roi_crop_raw
                                if vital == "ART_M":
                                    _, roi_width = roi_crop_raw.shape[:2]
                                    trim_x = int(roi_width * 0.05)
                                    if trim_x > 0 and trim_x < roi_width:
                                        roi_for_ocr = roi_crop_raw[:, trim_x:]
                                        print(
                                            f"[INFO] art_m_left_trim_applied bed={bed} field={vital} "
                                            f"trim_px={trim_x} original_w={roi_width}"
                                        )

                                prior_value = last_confirmed_values.get(bed, {}).get(vital)
                                run_tesseract_for_field = args.ocr_engine == "both" and (not tess_fields or vital in tess_fields)
                                ocr_result = ocr_numeric_roi(
                                    roi_for_ocr,
                                    reader,
                                    field_name=vital,
                                    prior_value=prior_value,
                                    config=config,
                                    ocr_engine=args.ocr_engine,
                                    run_tesseract_for_field=run_tesseract_for_field,
                                    low_conf_threshold=fail_roi_conf_threshold,
                                )
                                text = str(ocr_result.get("text") or "")
                                value = ocr_result.get("value")
                                conf_value = safe_float(ocr_result.get("conf"))
                                conf = conf_value if conf_value is not None else 0.0
                                method = str(ocr_result.get("method") or "none")
                                ocr_pass = str(ocr_result.get("ocr_pass") or "none")
                                imputed = bool(ocr_result.get("imputed", False))

                                roi_elapsed_ms = (time.perf_counter() - roi_start) * 1000.0
                                selected_engine = str(ocr_result.get("selected_engine") or "easyocr")
                                easy_result = ocr_result.get("easyocr") if isinstance(ocr_result.get("easyocr"), dict) else None
                                tess_result = ocr_result.get("tesseract") if isinstance(ocr_result.get("tesseract"), dict) else None
                                easy_value = safe_float(easy_result.get("value")) if easy_result else None
                                easy_conf = safe_float(easy_result.get("conf")) if easy_result else None
                                tess_value = safe_float(tess_result.get("value")) if tess_result else None
                                tess_conf = safe_float(tess_result.get("conf")) if tess_result else None
                                if imputed and method == "hold_last" and not bool(config.get("ocr_compare_mode", False)):
                                    easy_value = None
                                    tess_value = None
                                alt_ocr: dict[str, Any] = {}
                                if isinstance(ocr_result.get("tesseract"), dict):
                                    alt_ocr["tesseract"] = {
                                        "text": ocr_result["tesseract"].get("text", ""),
                                        "value": ocr_result["tesseract"].get("value"),
                                        "psm": ocr_result["tesseract"].get("psm"),
                                        "oem": ocr_result["tesseract"].get("oem"),
                                    }
                                beds[bed][vital] = {
                                    "text": text,
                                    "value": value,
                                    "confidence": conf,
                                    "ocr_method": method,
                                    "ocr_conf": conf_value,
                                    "imputed": imputed,
                                    "ocr_pass": ocr_pass,
                                    "selected_engine": selected_engine,
                                    "easy_value": easy_value,
                                    "easy_conf": easy_conf,
                                    "tess_value": tess_value,
                                    "tess_conf": tess_conf,
                                    "selected_value": value,
                                    "roi_elapsed_ms": roi_elapsed_ms,
                                    "easy_elapsed_ms": safe_float(ocr_result.get("easy_elapsed_ms")),
                                    "tess_elapsed_ms": safe_float(ocr_result.get("tess_elapsed_ms")),
                                }
                                if alt_ocr:
                                    beds[bed][vital]["alt_ocr"] = alt_ocr
                                print(f"[PERF] roi_elapsed_ms bed={bed} field={vital} ms={roi_elapsed_ms:.2f}")

                                if value is not None and not imputed:
                                    last_confirmed_values.setdefault(bed, {})[vital] = value

                                if not text:
                                    missing_by_field[vital] = missing_by_field.get(vital, 0) + 1
                                    bed_debug[vital] = ocr_result.get("debug", {})

                                save_target = fail_roi_fields is None or vital in fail_roi_fields
                                should_save, save_reason = should_save_failed_roi(
                                    ocr_result,
                                    text,
                                    value,
                                    confidence_threshold=fail_roi_conf_threshold,
                                )
                                if (
                                    save_fail_roi
                                    and save_target
                                    and should_save
                                    and fail_roi_saved_in_tick < max(int(args.fail_roi_max_per_tick), 0)
                                ):
                                    saved_files = save_failed_roi_artifacts(
                                        fail_roi_dir,
                                        timestamp=stamp,
                                        image_basename=image_basename,
                                        image_path=image_path.as_posix() if image_path else None,
                                        bed=bed,
                                        field=vital,
                                        reason=save_reason,
                                        roi_coords=roi_coords,
                                        ocr_result=ocr_result,
                                        debug_popup_failed_roi=debug_popup_failed_roi,
                                        elapsed_ms=roi_elapsed_ms,
                                    )
                                    fail_roi_saved_in_tick += 1
                                    print(
                                        f"[WARN] ocr_fail_saved bed={bed} field={vital} reason={save_reason} out={fail_roi_dir}"
                                    )

                                if args.debug_roi and roi_tick_dir is not None:
                                    cv2.imwrite(str(roi_tick_dir / f"roi_{bed}_{vital}.png"), roi_crop_raw)
                                    print(
                                        f"[INFO] debug sample bed={bed} vital={vital} roi_raw_shape={roi_crop_raw.shape} "
                                        f"OCR text='{text}' method={method} pass={ocr_pass}"
                                    )
                            except Exception as field_exc:  # noqa: BLE001
                                missing_by_field[vital] = missing_by_field.get(vital, 0) + 1
                                err_msg = str(field_exc)
                                beds[bed][vital] = {
                                    "text": "",
                                    "value": None,
                                    "confidence": 0.0,
                                    "ocr_method": "field_error",
                                    "ocr_conf": None,
                                    "imputed": False,
                                    "ocr_pass": "none",
                                    "selected_engine": None,
                                    "easy_value": None,
                                    "easy_conf": None,
                                    "tess_value": None,
                                    "tess_conf": None,
                                    "selected_value": None,
                                    "roi_elapsed_ms": (time.perf_counter() - roi_start) * 1000.0,
                                    "easy_elapsed_ms": None,
                                    "tess_elapsed_ms": None,
                                }
                                bed_debug[vital] = {"error": err_msg}

                                save_target = fail_roi_fields is None or vital in fail_roi_fields
                                if (
                                    save_fail_roi
                                    and save_target
                                    and fail_roi_saved_in_tick < max(int(args.fail_roi_max_per_tick), 0)
                                ):
                                    ocr_result_err = {
                                        "text": "",
                                        "value": None,
                                        "conf": None,
                                        "method": "field_error",
                                        "ocr_pass": "none",
                                        "debug": {
                                            "preprocess_passes_tried": [],
                                            "chosen_pass": "none",
                                            "raw_easyocr": {},
                                            "exception": err_msg,
                                            "traceback": traceback.format_exc(),
                                        },
                                        "debug_images": {
                                            "raw": roi_crop_raw,
                                            "pre": roi_crop_raw,
                                            "passes": {"error": roi_crop_raw},
                                        },
                                    }
                                    saved_files = save_failed_roi_artifacts(
                                        fail_roi_dir,
                                        timestamp=stamp,
                                        image_basename=image_basename,
                                        image_path=image_path.as_posix() if image_path else None,
                                        bed=bed,
                                        field=vital,
                                        reason="parse_fail",
                                        roi_coords=roi_coords if roi_coords is not None else (0, 0, 0, 0),
                                        ocr_result=ocr_result_err,
                                        debug_popup_failed_roi=debug_popup_failed_roi,
                                        elapsed_ms=(time.perf_counter() - roi_start) * 1000.0,
                                    )
                                    fail_roi_saved_in_tick += 1
                                    print(
                                        f"[WARN] ocr_fail_saved bed={bed} field={vital} reason=exception out={fail_roi_dir}"
                                    )

                                print(f"[WARN] field ocr failed bed={bed} field={vital} err={err_msg}", file=sys.stderr)
                        if bed_debug:
                            record_debug[bed] = bed_debug

                    if debug_img is not None:
                        cv2.imwrite(str(debug_dir / f"{stamp}_debug_rois.png"), debug_img)
                    if args.debug_roi and full_overlay_img is not None:
                        cv2.imwrite(str(debug_dir / f"{stamp}_full_overlay.png"), full_overlay_img)

                    copied_cache_snapshot = copy_cache_snapshot(cache_path, day_dir, stamp)
                    if not copied_cache_snapshot:
                        print(f"[WARN] cache snapshot failed at {stamp}", file=sys.stderr)

                    resolved_cache_snapshot, resolved_cache_snapshot_before, snapshot_delta_seconds = resolve_cache_snapshot_paths(
                        day_dir / "cache",
                        capture_dt,
                        nearest_window_sec=max(float(args.cache_nearest_window_sec), 0.0),
                        use_nearest_past_fallback=bool(args.cache_nearest_past_fallback),
                    )
                    if resolved_cache_snapshot is None and copied_cache_snapshot is not None:
                        resolved_cache_snapshot = copied_cache_snapshot
                    if resolved_cache_snapshot_before is None and cache_snapshot_before is not None:
                        resolved_cache_snapshot_before = cache_snapshot_before

                    frame_elapsed_ms = (time.perf_counter() - frame_start) * 1000.0
                    print(f"[PERF] frame_elapsed_ms={frame_elapsed_ms:.2f}")
                    ocr_timestamp = datetime.now().astimezone().isoformat(timespec="milliseconds")
                    record = {
                        "timestamp": capture_timestamp,
                        "capture_timestamp": capture_timestamp,
                        "ocr_timestamp": ocr_timestamp,
                        "image_path": image_path.as_posix() if image_path else None,
                        "cache_snapshot_path_before": resolved_cache_snapshot_before,
                        "cache_snapshot_path": resolved_cache_snapshot,
                        "cache_snapshot_delta_seconds": snapshot_delta_seconds,
                        "frame_elapsed_ms": frame_elapsed_ms,
                        "beds": beds,
                    }
                    if record_debug:
                        record["debug"] = record_debug
                    jsonl = day_dir / "ocr_results.jsonl"
                    with jsonl.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fh.flush()

                    if args.run_validator:
                        run_validator_once(jsonl, cache_path, validator_config, max(args.validator_last, 1))

                    print(f"[INFO] missing_by_field={json.dumps(missing_by_field, ensure_ascii=False)}")

                except Exception as exc:  # noqa: BLE001
                    print(f"[WARN] frame skipped due to error: {exc}", file=sys.stderr)

                sleep_ms = max(args.interval_ms - (time.time() - tick) * 1000, 0)
                print(f"[INFO] sleeping {sleep_ms:.1f} ms")
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
