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
DEC1_VITALS = {"TSKIN", "TRECT"}


def parse_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d%H%M%S"):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def format_value(vital: str, value, stale: bool) -> str:
    if stale or value is None:
        return "  NA  " if vital in DEC1_VITALS else "  NA "
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "  NA  " if vital in DEC1_VITALS else "  NA "

    if vital in DEC1_VITALS:
        return f"{num:05.1f}"
    if vital in INT_VITALS:
        return f"{int(round(num)):4d}"
    return f"{num:5.1f}"


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
        self.root.resizable(False, False)

        title_font = font.Font(family="Consolas", size=20, weight="bold")
        label_font = font.Font(family="Consolas", size=16, weight="bold")
        value_font = font.Font(family="Consolas", size=40, weight="bold")
        time_font = font.Font(family="Consolas", size=12)

        header = tk.Label(
            self.root,
            text="HL7 MONITOR (Fixed Layout 4x5 / Bed)",
            bg="white",
            fg="black",
            font=title_font,
            anchor="w",
        )
        header.pack(fill="x", padx=12, pady=(4, 4))

        body = tk.Frame(self.root, bg="white", width=1920, height=1040)
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        body.pack_propagate(False)

        for r in range(3):
            body.grid_rowconfigure(r, minsize=340, weight=0)
        for c in range(2):
            body.grid_columnconfigure(c, minsize=952, weight=0)

        self.cells: dict[tuple[str, str], tk.Label] = {}
        self.updated_labels: dict[str, tk.Label] = {}

        for i, bed in enumerate(BED_IDS):
            row_idx = i // 2
            col_idx = i % 2

            bed_frame = tk.Frame(body, bg="white", bd=2, relief="solid", width=944, height=334)
            bed_frame.grid(row=row_idx, column=col_idx, padx=4, pady=3)
            bed_frame.grid_propagate(False)

            tk.Label(
                bed_frame,
                text=bed,
                bg="white",
                fg="black",
                font=title_font,
                anchor="w",
            ).grid(row=0, column=0, columnspan=4, sticky="w", padx=8, pady=(2, 0))

            updated_lbl = tk.Label(
                bed_frame,
                text="last: --:--:--",
                bg="white",
                fg="black",
                font=time_font,
                anchor="w",
            )
            updated_lbl.grid(row=1, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 2))
            self.updated_labels[bed] = updated_lbl

            for c in range(4):
                bed_frame.grid_columnconfigure(c, minsize=231, weight=0)

            for i, vital in enumerate(VITAL_ORDER):
                row = 2 + (i // 4)
                col = i % 4
                cell = tk.Frame(bed_frame, bg="white", bd=1, relief="solid", width=223, height=50)
                cell.grid(row=row, column=col, padx=2, pady=2)
                cell.grid_propagate(False)

                tk.Label(cell, text=vital, bg="white", fg="black", font=label_font, anchor="w").place(x=5, y=0)
                value_lbl = tk.Label(cell, text=" NA ", bg="white", fg="black", font=value_font, anchor="e")
                value_lbl.place(x=2, y=8, width=216, height=40)
                self.cells[(bed, vital)] = value_lbl

    def load_cache(self) -> dict | None:
        if not self.cache_path.exists():
            return None
        try:
            with self.cache_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

    def refresh(self):
        payload = self.load_cache()
        if isinstance(payload, dict):
            self.last_payload = payload

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
                self.cells[(bed, vital)].configure(text=format_value(vital, value, stale))

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
