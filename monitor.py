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

INT_VITALS = {
    "HR", "ART_S", "ART_D", "ART_M", "CVP_M", "RAP_M", "SpO2", "rRESP", "EtCO2", "RR",
    "VTe", "VTi", "Ppeak", "PEEP", "O2conc", "NO", "BSR1", "BSR2",
}
DEC1_VITALS = {"TSKIN", "TRECT", "TEMP"}
MIN_VALUE_FONT_SIZE = 24


def parse_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def format_value_candidates(vital: str, value, stale: bool) -> list[str]:
    if stale or value is None:
        return ["NA"]
    try:
        num = float(value)
    except (TypeError, ValueError):
        return ["NA"]

    if vital in DEC1_VITALS:
        # 温度は 34.5 表記を維持（ゼロ埋めなし）
        return [f"{num:.1f}", f"{int(round(num))}"]
    if vital in INT_VITALS:
        return [str(int(round(num)))]
    return [f"{num:.1f}", f"{int(round(num))}"]


def shorten_text_candidates(text: str) -> list[str]:
    candidates = [text]
    if "." in text:
        candidates.append(text.split(".")[0])

    sign = ""
    body = text
    if body.startswith(("+", "-")):
        sign, body = body[0], body[1:]
    for keep in range(len(body) - 1, 0, -1):
        candidates.append(f"{sign}{body[-keep:]}")
    candidates.append(body[-1:] or "0")

    uniq: list[str] = []
    for c in candidates:
        if c and c not in uniq:
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
        self.value_font = font.Font(family="Consolas", size=40, weight="bold")
        self.value_measure_font = font.Font(family="Consolas", size=40, weight="bold")
        self.time_font = font.Font(family="Consolas", size=12)

        self.safe_top = 8
        self.safe_bottom = 16
        self.outer_x = 8
        self.header_h = 40
        self.header_gap = 4
        self.default_col_gap = 8
        self.default_row_gap = 6
        self.value_pad_x = 6
        self.value_pad_y = 4
        self.max_value_font_size = 40

        self.header = tk.Label(
            self.root,
            text="HL7 MONITOR (Fixed Layout 4x5 / Bed)",
            bg="white",
            fg="black",
            font=self.title_font,
            anchor="w",
        )
        self.header.place(x=12, y=4, width=1896, height=40)

        self.body = tk.Frame(self.root, bg="white")
        self.body.place(x=0, y=0, width=1, height=1)

        self.value_canvases: dict[tuple[str, str], tk.Canvas] = {}
        self.value_items: dict[tuple[str, str], int] = {}
        self.updated_labels: dict[str, tk.Label] = {}
        self.bed_frames: dict[str, tk.Frame] = {}
        self.cell_frames: list[tk.Frame] = []
        self.vital_name_labels: list[tk.Label] = []

        for i, bed in enumerate(BED_IDS):
            bed_frame = tk.Frame(self.body, bg="white", bd=2, relief="solid")
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
                cell.grid(row=row, column=col, sticky="nsew", padx=2, pady=2)
                self.cell_frames.append(cell)

                name_lbl = tk.Label(cell, text=vital, bg="white", fg="black", font=self.label_font, anchor="w")
                name_lbl.pack(anchor="nw", padx=5, pady=(0, 0))
                self.vital_name_labels.append(name_lbl)

                value_canvas = tk.Canvas(cell, bg="white", highlightthickness=0, bd=0)
                value_canvas.pack(fill="both", expand=True, padx=2, pady=(0, 4))
                item_id = value_canvas.create_text(2, 2, text="NA", anchor="se", fill="black", font=self.value_font)
                self.value_canvases[(bed, vital)] = value_canvas
                self.value_items[(bed, vital)] = item_id

        self.root.update_idletasks()
        self.relayout()
        self.root.bind("<Configure>", self.on_configure)

    def on_configure(self, event):
        if event.widget is self.root:
            self.relayout()

    def relayout(self):
        self.root.update_idletasks()
        client_w = max(self.root.winfo_width(), 1)
        client_h = max(self.root.winfo_height(), 1)

        header_y = self.safe_top
        header_x = self.outer_x + 4
        header_w = max(client_w - (self.outer_x * 2) - 8, 1)
        self.header.place(x=header_x, y=header_y, width=header_w, height=self.header_h)

        body_x = self.outer_x
        body_y = header_y + self.header_h + self.header_gap
        body_w = max(client_w - (self.outer_x * 2), 1)
        body_h = max(client_h - body_y - self.safe_bottom, 1)
        self.body.place(x=body_x, y=body_y, width=body_w, height=body_h)

        col_gap = self.default_col_gap if body_w >= 1800 else 4
        row_gap = self.default_row_gap if body_h >= 1000 else 3
        cell_gap = 2 if min(body_w, body_h) >= 900 else 1

        bed_w = max((body_w - col_gap) // 2, 1)
        bed_h = max((body_h - (2 * row_gap)) // 3, 1)
        cell_h = max(bed_h / 5.0, 1.0)

        safe_cell_h = max(int(cell_h * 0.95), 1)
        value_available_h = max(safe_cell_h - (self.value_pad_y * 2), 1)
        value_font_size = max(int(cell_h * 0.75), MIN_VALUE_FONT_SIZE)
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

        self.render_from_payload()

    def fit_text_to_box(self, text_candidates: list[str], box_w: int, box_h: int) -> tuple[str, int]:
        box_w = max(box_w, 1)
        box_h = max(box_h, 1)

        expanded: list[str] = []
        for text in text_candidates:
            for candidate in shorten_text_candidates(text):
                if candidate not in expanded:
                    expanded.append(candidate)

        for size in range(self.max_value_font_size, MIN_VALUE_FONT_SIZE - 1, -1):
            self.value_measure_font.configure(size=size)
            if self.value_measure_font.metrics("linespace") > box_h:
                continue
            for text in expanded:
                if self.value_measure_font.measure(text) <= box_w:
                    return text, size

        self.value_measure_font.configure(size=MIN_VALUE_FONT_SIZE)
        for text in expanded:
            if self.value_measure_font.measure(text) <= box_w:
                return text, MIN_VALUE_FONT_SIZE
        return (expanded[-1] if expanded else "NA"), MIN_VALUE_FONT_SIZE

    def render_value(self, bed: str, vital: str, candidates: list[str]):
        canvas = self.value_canvases[(bed, vital)]
        item_id = self.value_items[(bed, vital)]

        canvas.update_idletasks()
        c_w = max(canvas.winfo_width(), 1)
        c_h = max(canvas.winfo_height(), 1)
        avail_w = max(c_w - (self.value_pad_x * 2), 1)
        avail_h = max(c_h - (self.value_pad_y * 2), 1)

        text, size = self.fit_text_to_box(candidates, avail_w, avail_h)

        # 右寄せ時にxがセル外にならないようクランプ。
        x = max(c_w - self.value_pad_x, self.value_pad_x)
        y = max(c_h - self.value_pad_y, self.value_pad_y)
        canvas.coords(item_id, x, y)
        canvas.itemconfigure(item_id, text=text, font=("Consolas", size, "bold"), anchor="se")

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

            stale = (now - self.last_bed_seen_at[bed]) > self.stale_seconds if self.last_bed_seen_at[bed] else True

            dt = parse_timestamp(self.last_bed_stamp[bed])
            ts_label = dt.strftime("%H:%M:%S") if dt else "--:--:--"
            self.updated_labels[bed].configure(text=f"last: {ts_label}")

            for vital in VITAL_ORDER:
                vital_obj = vitals.get(vital) if isinstance(vitals.get(vital), dict) else None
                value = vital_obj.get("value") if vital_obj else None
                candidates = format_value_candidates(vital, value, stale)
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
