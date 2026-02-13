#!/usr/bin/env python3
"""固定レイアウトの6ベッドモニタGUI。"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication, QGridLayout, QLabel, QMainWindow, QWidget

VITAL_ORDER = [
    "HR", "ART_S", "ART_D", "ART_M", "CVP_M", "RAP_M", "SpO2", "TSKIN", "TRECT", "rRESP",
    "EtCO2", "RR", "VTe", "VTi", "Ppeak", "PEEP", "O2conc", "NO", "BSR1", "BSR2",
]

FORMATTERS = {
    "HR": "{:03.0f}", "ART_S": "{:03.0f}", "ART_D": "{:03.0f}", "ART_M": "{:03.0f}",
    "CVP_M": "{:02.0f}", "RAP_M": "{:02.0f}", "SpO2": "{:03.0f}", "TSKIN": "{:04.1f}",
    "TRECT": "{:04.1f}", "rRESP": "{:03.0f}", "EtCO2": "{:03.0f}", "RR": "{:03.0f}",
    "VTe": "{:03.0f}", "VTi": "{:03.0f}", "Ppeak": "{:02.0f}", "PEEP": "{:02.0f}",
    "O2conc": "{:03.0f}", "NO": "{:02.0f}", "BSR1": "{:03.0f}", "BSR2": "{:03.0f}",
}


class MonitorWindow(QMainWindow):
    def __init__(self, latest_file: str = "latest.json"):
        super().__init__()
        self.latest_file = Path(latest_file)
        self.beds = [f"BED{i:02d}" for i in range(1, 7)]
        self.value_labels = {}

        self.setWindowTitle("ICU Monitor")
        self.resize(1600, 700)

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("white"))
        palette.setColor(QPalette.WindowText, QColor("black"))
        self.setPalette(palette)

        central = QWidget()
        grid = QGridLayout(central)
        mono = QFont("Courier New", 11)
        mono.setStyleHint(QFont.Monospace)

        header = QLabel("ITEM")
        header.setFont(mono)
        grid.addWidget(header, 0, 0)

        for col, bed in enumerate(self.beds, start=1):
            label = QLabel(bed)
            label.setFont(mono)
            grid.addWidget(label, 0, col)

        for row, vital in enumerate(VITAL_ORDER, start=1):
            item = QLabel(vital)
            item.setFont(mono)
            grid.addWidget(item, row, 0)
            for col, bed in enumerate(self.beds, start=1):
                value_label = QLabel("NA")
                value_label.setFont(mono)
                value_label.setMinimumWidth(90)
                grid.addWidget(value_label, row, col)
                self.value_labels[(bed, vital)] = value_label

        self.setCentralWidget(central)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(1000)
        self.refresh()

    def refresh(self):
        now = datetime.now(timezone.utc)
        data = {}
        if self.latest_file.exists():
            try:
                data = json.loads(self.latest_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}

        for bed in self.beds:
            bed_data = data.get(bed, {})
            for vital in VITAL_ORDER:
                label = self.value_labels[(bed, vital)]
                entry = bed_data.get(vital)
                if not entry:
                    label.setText("NA")
                    label.setStyleSheet("color: gray;")
                    continue

                ts = entry.get("time")
                stale = True
                if ts:
                    try:
                        dt = datetime.fromisoformat(ts)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        stale = (now - dt).total_seconds() > 20
                    except ValueError:
                        stale = True

                value = entry.get("value")
                if value is None:
                    txt = "NA"
                else:
                    fmt = FORMATTERS.get(vital, "{}")
                    txt = fmt.format(value)
                label.setText(txt)
                label.setStyleSheet("color: gray;" if stale else "color: black;")


def main():
    latest_file = sys.argv[1] if len(sys.argv) > 1 else "latest.json"
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MonitorWindow(latest_file=latest_file)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
