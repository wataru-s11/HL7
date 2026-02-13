#!/usr/bin/env python3
"""
固定レイアウトのモニタGUI。
hl7_receiver.py が出力する monitor_cache.json を監視表示する。
OCR前提: 白背景・黒文字・桁固定(等幅フォント)。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter import font


BED_IDS = [f"BED0{i}" for i in range(1, 7)]
VITAL_ORDER = ["HR", "SpO2", "NIBP_SYS", "NIBP_DIA", "RR", "TEMP"]


def format_value(value) -> str:
    if value is None:
        return "----"
    if isinstance(value, float):
        return f"{value:>4.1f}"
    return f"{int(value):>4d}" if isinstance(value, int) else f"{str(value):>4}"


class MonitorApp:
    def __init__(self, cache_path: Path, refresh_ms: int = 1000):
        self.cache_path = cache_path
        self.refresh_ms = refresh_ms

        self.root = tk.Tk()
        self.root.title("HL7 Bed Monitor")
        self.root.configure(bg="white")

        mono = font.Font(family="Consolas", size=14)
        title_font = font.Font(family="Consolas", size=16, weight="bold")

        header = tk.Label(
            self.root,
            text="HL7 MONITOR (Fixed Layout)",
            bg="white",
            fg="black",
            font=title_font,
        )
        header.grid(row=0, column=0, columnspan=7, sticky="w", padx=12, pady=8)

        tk.Label(self.root, text="BED", bg="white", fg="black", font=mono).grid(row=1, column=0, padx=8)
        for idx, vital in enumerate(VITAL_ORDER, start=1):
            tk.Label(self.root, text=f"{vital:>8}", bg="white", fg="black", font=mono).grid(
                row=1, column=idx, padx=8
            )

        self.cells: dict[tuple[str, str], tk.Label] = {}
        for row_idx, bed in enumerate(BED_IDS, start=2):
            tk.Label(self.root, text=f"{bed:>6}", bg="white", fg="black", font=mono).grid(
                row=row_idx, column=0, padx=8, pady=4
            )
            for col_idx, vital in enumerate(VITAL_ORDER, start=1):
                lbl = tk.Label(
                    self.root,
                    text="----    ",
                    width=8,
                    anchor="e",
                    bg="white",
                    fg="black",
                    font=mono,
                )
                lbl.grid(row=row_idx, column=col_idx, padx=8, pady=4)
                self.cells[(bed, vital)] = lbl

    def load_cache(self) -> dict:
        if not self.cache_path.exists():
            return {}
        try:
            with self.cache_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            return {}

    def refresh(self):
        payload = self.load_cache()
        beds = payload.get("beds", {}) if isinstance(payload, dict) else {}

        for bed in BED_IDS:
            row = beds.get(bed, {})
            vitals = row.get("vitals", {}) if isinstance(row, dict) else {}
            for vital in VITAL_ORDER:
                value = vitals.get(vital, {}).get("value") if isinstance(vitals.get(vital), dict) else None
                self.cells[(bed, vital)].configure(text=f"{format_value(value):>8}")

        self.root.after(self.refresh_ms, self.refresh)

    def run(self):
        self.refresh()
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="HL7 monitor GUI")
    parser.add_argument("--cache", default="monitor_cache.json")
    parser.add_argument("--refresh-ms", type=int, default=1000)
    args = parser.parse_args()

    app = MonitorApp(Path(args.cache), refresh_ms=args.refresh_ms)
    app.run()


if __name__ == "__main__":
    main()
