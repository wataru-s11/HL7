#!/usr/bin/env python3
"""
固定レイアウトのモニタGUI。
hl7_receiver.py が出力する monitor_cache.json を監視表示する。
OCR前提: 白背景・黒文字・桁固定(等幅フォント)。
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import font


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

DEC1_VITALS = {"TSKIN", "TRECT", "TEMP"}
MIN_VALUE_FONT_SIZE = 18
MISSING_PLACEHOLDER = "...."
CELL_BORDER_MARGIN = 1
LABEL_SAFE_PAD_X = 4
LABEL_SAFE_PAD_Y = 6
VALUE_RESERVED_BOTTOM = 3


def parse_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def format_value(vital_key: str, value) -> str:
    if value is None:
        return MISSING_PLACEHOLDER
    try:
        num = float(value)
    except (TypeError, ValueError):
        return MISSING_PLACEHOLDER

    if vital_key in ("TSKIN", "TRECT"):
        return f"{num:.1f}"
    return f"{int(round(num))}"


def format_value_candidates(vital: str, value) -> list[str]:
    if value is None:
        return [MISSING_PLACEHOLDER]
    return [format_value(vital, value)]


def shorten_text_for_fit(text: str, allow_decimal_drop: bool = True) -> list[str]:
    """fit-to-box 用の表示短縮候補を優先順で返す。"""
    candidates: list[str] = [text]
    try:
        num = float(text)
    except (TypeError, ValueError):
        num = None

    # 要件: 小数は 1桁 -> 0桁 へ短縮。
    if allow_decimal_drop and num is not None and "." in str(text):
        candidates.append(str(int(round(num))))

    # 要件: それでも入らない場合は整数化し、3〜4文字程度に短縮。
    if allow_decimal_drop and num is not None:
        rounded = int(round(num))
        reduced = str(rounded)
        if len(reduced) > 4:
            reduced = reduced[:4]
        candidates.append(reduced)

    uniq: list[str] = []
    for c in candidates:
        if c not in uniq:
            uniq.append(c)
    return uniq


class MonitorApp:
    def __init__(self, cache_path: Path, refresh_ms: int = 1000, stale_seconds: int = 30):
        self.cache_path = cache_path
        self.refresh_ms = refresh_ms
        self.stale_seconds = stale_seconds
        self.last_payload: dict = {}
        self.last_bed_seen_at: dict[str, float] = {bed: 0.0 for bed in BED_IDS}
        self.last_bed_stamp: dict[str, str] = {bed: "" for bed in BED_IDS}

        self.root = tk.Tk()
        self.root.title("HL7 Bed Monitor")
        self.root.configure(bg="white")
        self.root.geometry("1920x1080")
        self.root.minsize(1280, 760)

        self.title_font = font.Font(family="Consolas", size=20, weight="bold")
        self.label_font = font.Font(family="Consolas", size=16, weight="bold")
        self.value_font = font.Font(family="Consolas", size=40, weight="normal")
        self.value_measure_font = font.Font(family="Consolas", size=40, weight="normal")
        self.time_font = font.Font(family="Consolas", size=12)

        self.safe_top = 8
        self.safe_bottom = 16
        self.outer_margin = 10
        self.header_h = 40
        self.header_gap = 4
        self.default_col_gap = 8
        self.default_row_gap = 6
        self.value_pad_x = 2
        self.value_pad_y = 0
        self.value_pad_left = self.value_pad_x
        self.value_pad_right = self.value_pad_x
        self.value_pad_top = self.value_pad_y
        self.value_pad_bottom = self.value_pad_y
        self.max_value_font_size = 40
        self._redrawing = False

        self.header = tk.Label(
            self.root,
            text="HL7 MONITOR (Fixed Layout 4x5 / Bed)",
            bg="white",
            fg="black",
            font=self.title_font,
            anchor="w",
        )
        self.header.place(x=12, y=4, width=1896, height=40)

        self.canvas = tk.Canvas(self.root, bg="white", highlightthickness=0, bd=0)
        self.canvas.place(x=0, y=0, width=1, height=1)
        self.canvas_w = 1
        self.canvas_h = 1

        self.value_labels: dict[tuple[str, str], tk.Label] = {}
        self.updated_labels: dict[str, tk.Label] = {}
        self.bed_frames: dict[str, tk.Frame] = {}
        self.cell_frames: list[tk.Frame] = []
        self.vital_name_labels: list[tk.Label] = []

        for i, bed in enumerate(BED_IDS):
            bed_frame = tk.Frame(self.canvas, bg="white", bd=2, relief="solid")
            self.bed_frames[bed] = bed_frame

            tk.Label(
                bed_frame,
                text=bed,
                bg="white",
                fg="black",
                font=self.title_font,
                anchor="w",
            ).grid(row=0, column=0, columnspan=4, sticky="w", padx=8, pady=(2, 0))

            updated_lbl = tk.Label(
                bed_frame,
                text="last: --:--:--",
                bg="white",
                fg="black",
                font=self.time_font,
                anchor="w",
            )
            updated_lbl.grid(row=1, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 2))
            self.updated_labels[bed] = updated_lbl

            for c in range(4):
                bed_frame.grid_columnconfigure(c, weight=1)
            for r in range(2, 7):
                bed_frame.grid_rowconfigure(r, weight=1)

            for i, vital in enumerate(VITAL_ORDER):
                row = 2 + (i // 4)
                col = i % 4
                cell = tk.Frame(bed_frame, bg="white", bd=1, relief="solid")
                cell.grid(row=row, column=col, sticky="nsew", padx=CELL_BORDER_MARGIN, pady=CELL_BORDER_MARGIN)
                self.cell_frames.append(cell)

                cell.grid_rowconfigure(1, weight=1)
                cell.grid_columnconfigure(0, weight=1)

                name_lbl = tk.Label(cell, text=vital, bg="white", fg="black", font=self.label_font, anchor="w")
                name_lbl.grid(row=0, column=0, sticky="nw", padx=5, pady=(0, 0))
                self.vital_name_labels.append(name_lbl)

                value_label = tk.Label(
                    cell,
                    text="NA",
                    bg="white",
                    fg="black",
                    font=self.value_font,
                    anchor="e",
                    justify="right",
                )
                value_label.grid(row=1, column=0, sticky="nsew", padx=2, pady=(0, 3))
                self.value_labels[(bed, vital)] = value_label

        self.root.update_idletasks()
        self.relayout()
        self.root.bind("<Configure>", self.on_configure)
        self.canvas.bind("<Configure>", self.on_resize)
        self.root.after(200, self.redraw_all)

    def on_configure(self, event):
        if event.widget is self.root:
            self.relayout()

    def on_resize(self, event):
        if event.widget is self.canvas:
            self.canvas_w = max(event.width, 1)
            self.canvas_h = max(event.height, 1)
            self.redraw_all()

    def redraw_all(self):
        if self._redrawing:
            return
        self._redrawing = True
        try:
            self.relayout()
            self.render_from_payload()
        finally:
            self._redrawing = False

    def relayout(self):
        self.root.update_idletasks()
        client_w = max(self.root.winfo_width(), 1)
        client_h = max(self.root.winfo_height(), 1)

        header_y = self.safe_top
        safe_left = self.outer_margin
        safe_top = self.outer_margin
        safe_right = max(client_w - self.outer_margin, safe_left + 1)
        safe_bottom = max(client_h - self.outer_margin, safe_top + 1)
        safe_w = max(safe_right - safe_left, 1)
        safe_h = max(safe_bottom - safe_top, 1)

        header_x = safe_left + 4
        header_w = max(safe_w - 8, 1)
        self.header.place(x=header_x, y=header_y, width=header_w, height=self.header_h)

        body_x = safe_left
        body_y = safe_top + self.header_h + self.header_gap
        body_w = safe_w
        body_h = max(safe_h - self.header_h - self.header_gap, 1)
        self.canvas.place(x=body_x, y=body_y, width=body_w, height=body_h)

        self.canvas.update_idletasks()
        W = max(self.canvas.winfo_width(), 1)
        H = max(self.canvas.winfo_height(), 1)
        self.canvas_w = W
        self.canvas_h = H

        col_gap = self.default_col_gap if W >= 1800 else 4
        row_gap = self.default_row_gap if H >= 1000 else 3
        cell_gap = 2 if min(W, H) >= 900 else 1

        bed_w = max((W - col_gap) // 2, 1)
        bed_h = max((H - (2 * row_gap)) // 3, 1)
        cell_h = max(bed_h / 5.0, 1.0)

        safe_cell_h = max(int(cell_h * 0.95), 1)
        value_available_h = max(safe_cell_h - (self.value_pad_y * 2), 1)
        # 値フォントは従来より 2〜4px 程度大きくしつつ、実際の描画時は fit-to-box で必ず収める。
        value_font_size = max(int(cell_h * 0.75) + 3, MIN_VALUE_FONT_SIZE)
        while value_font_size > 6:
            self.value_font.configure(size=value_font_size)
            metrics = self.value_font.metrics()
            line_h = int(metrics.get("ascent", 0)) + int(metrics.get("descent", 0))
            if line_h <= value_available_h:
                break
            value_font_size -= 1
        self.max_value_font_size = value_font_size

        label_font_size = max(min(int(cell_h * 0.24), max(value_font_size // 2, 8)), 8)
        time_font_size = max(min(int(cell_h * 0.18), 14), 8)
        title_font_size = max(min(int(cell_h * 0.26), 22), 12)

        self.label_font.configure(size=label_font_size)
        self.time_font.configure(size=time_font_size)
        self.title_font.configure(size=title_font_size)

        for name_lbl in self.vital_name_labels:
            name_lbl.configure(pady=0)

        for i, bed in enumerate(BED_IDS):
            row_idx = i // 2
            col_idx = i % 2
            x = col_idx * (bed_w + col_gap)
            y = row_idx * (bed_h + row_gap)
            self.bed_frames[bed].place(x=x, y=y, width=bed_w, height=bed_h)

        for cell in self.cell_frames:
            cell.grid_configure(padx=cell_gap, pady=cell_gap)

    def _fit_single_text(self, text: str, avail_w: int, avail_h: int) -> tuple[bool, int, int, int]:
        start_size = int(min(avail_h * 0.95 + 2, 64))
        font_size = max(start_size, MIN_VALUE_FONT_SIZE)

        while font_size >= MIN_VALUE_FONT_SIZE:
            self.value_measure_font.configure(size=font_size)
            width = int(self.value_measure_font.measure(text))
            height = int(self.value_measure_font.metrics("ascent")) + int(self.value_measure_font.metrics("descent"))
            if width <= avail_w and height <= avail_h:
                return True, font_size, width, height
            font_size -= 2

        self.value_measure_font.configure(size=MIN_VALUE_FONT_SIZE)
        width = int(self.value_measure_font.measure(text))
        height = int(self.value_measure_font.metrics("ascent")) + int(self.value_measure_font.metrics("descent"))
        return (width <= avail_w and height <= avail_h), MIN_VALUE_FONT_SIZE, width, height

    def fit_text_to_box(
        self,
        text_candidates: list[str],
        avail_w: int,
        avail_h: int,
        allow_decimal_drop: bool = True,
    ) -> tuple[str, int, int, int]:
        avail_w = max(avail_w, 1)
        avail_h = max(avail_h, 1)

        fit_attempts: list[str] = []
        for text in text_candidates:
            for candidate in shorten_text_for_fit(text, allow_decimal_drop=allow_decimal_drop):
                if candidate not in fit_attempts:
                    fit_attempts.append(candidate)

        for candidate in fit_attempts:
            fits, size, width, height = self._fit_single_text(candidate, avail_w, avail_h)
            if fits:
                return candidate, size, width, height

        # 最終手段: 最後の候補を最小フォントで強制表示（値がある限り数値表示を維持）。
        fallback = fit_attempts[-1] if fit_attempts else "0"
        self.value_measure_font.configure(size=MIN_VALUE_FONT_SIZE)
        width = int(self.value_measure_font.measure(fallback))
        height = int(self.value_measure_font.metrics("ascent")) + int(self.value_measure_font.metrics("descent"))
        return fallback, MIN_VALUE_FONT_SIZE, width, height

    def render_value(self, bed: str, vital: str, candidates: list[str]):
        value_label = self.value_labels[(bed, vital)]

        value_label.update_idletasks()
        label_w = max(value_label.winfo_width(), 1)
        label_h = max(value_label.winfo_height(), 1)

        left_inset = self.value_pad_left + CELL_BORDER_MARGIN
        right_inset = self.value_pad_right + CELL_BORDER_MARGIN
        top_inset = self.value_pad_top
        bottom_inset = self.value_pad_bottom

        avail_w = max(label_w - (left_inset + right_inset), 1)
        avail_h = max(label_h - (top_inset + bottom_inset), 1)
        avail_w = max(avail_w - LABEL_SAFE_PAD_X, 1)
        avail_h = max(avail_h - LABEL_SAFE_PAD_Y, 1)
        avail_h = max(avail_h - VALUE_RESERVED_BOTTOM, 1)

        is_temp_vital = vital in DEC1_VITALS
        text, size, _, _ = self.fit_text_to_box(
            candidates,
            avail_w,
            avail_h,
            allow_decimal_drop=not is_temp_vital,
        )

        value_label.configure(text=text, font=("Consolas", size, "normal"), anchor="e", justify="right")

    def load_cache(self) -> dict | None:
        if not self.cache_path.exists():
            return None
        try:
            with self.cache_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

    def render_from_payload(self):
        beds = self.last_payload.get("beds", {}) if isinstance(self.last_payload, dict) else {}
        now = time.monotonic()

        for bed in BED_IDS:
            row = beds.get(bed, {})
            vitals = row.get("vitals", {}) if isinstance(row, dict) else {}
            bed_stamp = str(row.get("message_datetime", "")) if isinstance(row, dict) else ""

            if bed_stamp and bed_stamp != self.last_bed_stamp[bed]:
                self.last_bed_stamp[bed] = bed_stamp
                self.last_bed_seen_at[bed] = now

            dt = parse_timestamp(self.last_bed_stamp[bed])
            ts_label = dt.strftime("%H:%M:%S") if dt else "--:--:--"
            self.updated_labels[bed].configure(text=f"last: {ts_label}")

            for vital in VITAL_ORDER:
                vital_obj = vitals.get(vital) if isinstance(vitals.get(vital), dict) else None
                value = vital_obj.get("value") if vital_obj else None
                candidates = format_value_candidates(vital, value)
                self.render_value(bed, vital, candidates)

    def refresh(self):
        payload = self.load_cache()
        if isinstance(payload, dict):
            self.last_payload = payload

        self.render_from_payload()
        self.root.after(self.refresh_ms, self.refresh)

    def run(self):
        self.refresh()
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="HL7 monitor GUI")
    parser.add_argument("--cache", default="monitor_cache.json")
    parser.add_argument("--refresh-ms", type=int, default=1000, help="JSON再読込周期(ms)")
    parser.add_argument("--stale-sec", "--stale-seconds", dest="stale_sec", type=int, default=30, help="更新停止時にNA化する秒数")
    args = parser.parse_args()

    app = MonitorApp(Path(args.cache), refresh_ms=args.refresh_ms, stale_seconds=args.stale_sec)
    app.run()


if __name__ == "__main__":
    main()
