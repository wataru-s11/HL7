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
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from ctypes import wintypes

import cv2
import easyocr
import mss
import numpy as np
import torch

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
    # MOD: vital別ROI微調整設定を追加（px/ratioの両対応）
    "per_vital_roi_adjust": {},
    "preprocess": {
        "resize": 2.5,
        "trim_px": 3,
        "threshold_mode": "adaptive",
        "adaptive_block_size": 31,
        "adaptive_c": 2,
        "morph_kernel": 2,
    },
    "vital_ranges": DEFAULT_VITAL_RANGES,
}


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


def get_window_capture_region(title: str, capture_client: bool) -> CaptureRegion | None:
    if not sys.platform.startswith("win"):
        return None

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    hwnd = find_monitor_window_hwnd(title)
    if not hwnd:
        return None

    if capture_client:
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
    else:
        window_rect = RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(window_rect)):
            return None
        left = int(window_rect.left)
        top = int(window_rect.top)
        width = int(window_rect.right - window_rect.left)
        height = int(window_rect.bottom - window_rect.top)

    if width <= 0 or height <= 0:
        return None

    return CaptureRegion(left=left, top=top, width=width, height=height)


def dpi_aware() -> None:
    if not sys.platform.startswith("win"):
        return
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
    if "resize" not in pp_cfg and "scale" in pp_cfg:
        pp_cfg["resize"] = pp_cfg.get("scale")
    if "threshold" in pp_cfg and "threshold_mode" not in pp_cfg:
        pp_cfg["threshold_mode"] = "otsu" if pp_cfg.get("threshold") else "adaptive"
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

    resize = float(pp.get("resize", 2.5))
    if vital_name in TEMPERATURE_VITALS:
        resize = max(resize, 3.0)
    if resize > 1.0:
        out = cv2.resize(out, None, fx=resize, fy=resize, interpolation=cv2.INTER_CUBIC)

    if out.ndim == 3:
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    trim_px = int(pp.get("trim_px", 3))
    if vital_name in TEMPERATURE_VITALS:
        trim_px = max(trim_px, 2)
    max_trim = max(min(out.shape[0], out.shape[1]) // 4, 0)
    trim_px = clamp(trim_px, 0, max_trim)
    if trim_px > 0 and out.shape[0] > trim_px * 2 and out.shape[1] > trim_px * 2:
        out = out[trim_px:-trim_px, trim_px:-trim_px]

    threshold_mode = str(pp.get("threshold_mode", "adaptive")).lower()
    if variant == "otsu" or threshold_mode == "otsu":
        _, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        block_size = int(pp.get("adaptive_block_size", 31))
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(block_size, 3)
        c_value = int(pp.get("adaptive_c", 2))
        out = cv2.adaptiveThreshold(out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_value)

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


def run_ocr(reader: easyocr.Reader, roi_image: np.ndarray, vital_name: str, config: dict[str, Any]) -> tuple[str, float, list[tuple[str, np.ndarray]]]:
    variants = ["normal", "invert", "otsu"] if vital_name in TEMPERATURE_VITALS else ["normal", "otsu"]
    best_text, best_conf = "", -1.0
    debug_images: list[tuple[str, np.ndarray]] = []

    for variant in variants:
        processed = preprocess_roi(roi_image, vital_name, config, variant=variant)
        debug_images.append((variant, processed))
        results = reader.readtext(processed, detail=1, paragraph=False, allowlist=ALLOWLIST)
        for _, text, conf in results:
            filtered = _sanitize_numeric_text(text)
            if filtered and conf > best_conf:
                best_text, best_conf = filtered, float(conf)

    if best_conf < 0:
        return "", 0.0, debug_images
    return best_text, best_conf, debug_images


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
        text, conf, _ = run_ocr(reader, roi_img, "RAP_M", config)

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
    parser.add_argument("--interval-ms", type=int, default=10000)
    parser.add_argument("--save-images", type=parse_bool, default=True)
    parser.add_argument("--debug-roi", type=parse_bool, default=False)
    parser.add_argument("--debug-lines", type=parse_bool, default=False)
    parser.add_argument("--debug-window-rect", type=parse_bool, default=False)
    parser.add_argument("--gpu", type=parse_bool, default=True)
    parser.add_argument("--no-launch-monitor", type=parse_bool, default=True)  # MOD: 安全側デフォルト
    parser.add_argument("--run-validator", type=parse_bool, default=False)  # MOD
    parser.add_argument("--validator-last", type=int, default=50)  # MOD
    parser.add_argument("--validator-config", default="validator_config.json")  # MOD
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()

    dpi_aware()
    log_windows_dpi_info()

    config_path = Path(args.config)
    config = ensure_config(config_path)
    config["window_title"] = args.window_title
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
        if args.capture_mode == "mss":
            mss_monitor_base = choose_capture_region(
                sct,
                args.target_display,
                windows_monitors,
                args.display_index,
                args.mss_monitor_index,
            )
            mss_capture_rect = get_monitor_rect(config, mss_monitor_base)

        try:
            while True:
                tick = time.time()
                print(f"[INFO] capture tick start ts={datetime.now().isoformat(timespec='seconds')}")
                try:
                    day = datetime.now().strftime("%Y%m%d")
                    day_dir = Path(args.outdir) / day
                    images_dir = day_dir / "images"
                    debug_dir = day_dir / "debug"
                    day_dir.mkdir(parents=True, exist_ok=True)
                    if args.save_images:
                        images_dir.mkdir(parents=True, exist_ok=True)

                    capture_rect: CaptureRegion | None = None
                    frame: np.ndarray | None = None

                    if args.capture_mode == "window":
                        capture_rect = get_window_capture_region(args.window_title, args.capture_client)
                        if capture_rect is None:
                            print(f"[INFO] window found=not found title='{args.window_title}'")
                            raise RuntimeError("monitor window not found")

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

                    blue_rois: dict[str, dict[str, tuple[int, int, int, int]]]
                    red_rois: dict[str, dict[str, tuple[int, int, int, int]]]
                    roi_debug: dict[str, dict[str, Any]]
                    if args.roi_source == "exported":
                        roi_map = load_exported_roi_map(ROI_MAP_PATH)
                        if roi_map is None:
                            print("[WARN] ROI map not found, fallback to legacy", file=sys.stderr)
                            red_rois, roi_debug = build_vital_rois(frame, config, return_debug=True)
                            blue_rois = red_rois
                        else:
                            value_slice_cfg = config.get("cell_value_slice") if isinstance(config.get("cell_value_slice"), dict) else {}
                            slice_x1 = float(value_slice_cfg.get("x1_ratio", 0.60))
                            slice_x2 = float(value_slice_cfg.get("x2_ratio", 0.98))
                            slice_y1 = float(value_slice_cfg.get("y1_ratio", 0.10))
                            slice_y2 = float(value_slice_cfg.get("y2_ratio", 0.92))
                            blue_rois, red_rois, roi_debug = build_exported_rois(
                                frame=frame,
                                capture_rect=capture_rect,
                                roi_map=roi_map,
                                x1_ratio=slice_x1,
                                x2_ratio=slice_x2,
                                y1_ratio=slice_y1,
                                y2_ratio=slice_y2,
                            )
                    else:
                        red_rois, roi_debug = build_vital_rois(frame, config, return_debug=True)
                        blue_rois = red_rois
                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

                    image_path = None
                    if args.save_images:
                        image_path = images_dir / f"{stamp}.png"
                        cv2.imwrite(str(image_path), frame)

                    if args.debug_window_rect:
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        write_debug_window_rect_image(debug_dir / f"{stamp}_window_rect.png", frame)

                    if args.debug_roi or args.debug_lines:
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        debug_img = frame.copy()
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

                                    blue = blue_rois.get(bed, {}).get(vital)
                                    if blue:
                                        x1, y1, x2, y2 = blue
                                        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                                    red = red_rois.get(bed, {}).get(vital)
                                    if red:
                                        x1, y1, x2, y2 = red
                                        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        cv2.imwrite(str(debug_dir / f"{stamp}_rois.png"), debug_img)

                    beds: dict[str, dict[str, Any]] = {bed: {} for bed in BED_IDS}
                    for bed in BED_IDS:
                        for vital in VITAL_ORDER:
                            roi_box = red_rois.get(bed, {}).get(vital)
                            if roi_box is None:
                                beds[bed][vital] = {"text": "", "value": None, "confidence": 0.0}
                                continue
                            x1, y1, x2, y2 = roi_box
                            roi = frame[y1:y2, x1:x2]
                            text, conf, debug_variants = run_ocr(reader, roi, vital, config)
                            value = parse_numeric(text, vital, config)
                            beds[bed][vital] = {"text": text, "value": value, "confidence": conf}

                            if args.debug_roi:
                                for variant_name, variant_img in debug_variants:
                                    roi_debug_path = debug_dir / f"{stamp}_roi_{bed}_{vital}_{variant_name}.png"
                                    cv2.imwrite(str(roi_debug_path), variant_img)

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
