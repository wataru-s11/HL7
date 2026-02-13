#!/usr/bin/env python3
"""OCR最優先の固定レイアウトモニタGUI (PySide6)。"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QLabel,
    QMainWindow,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
BED_IDS = [f"BED{i:02d}" for i in range(1, 7)]
DATA_PATH = Path("latest.json")
POLL_INTERVAL_MS = 500
STALE_TIMEOUT_SEC = 15


@dataclass(frozen=True)
class VitalSpec:
    key: str
    label: str
    int_digits: int
    decimals: int = 0

    @property
    def total_len(self) -> int:
        return self.int_digits + (1 if self.decimals > 0 else 0) + self.decimals

    def format_value(self, raw: object) -> str:
        if raw is None:
            return self.format_na()

        text = str(raw).strip()
        if text == "":
            return self.format_na()

        if text.upper() == "NA":
            return self.format_na()

        try:
            num = float(text)
        except (TypeError, ValueError):
            return self.format_na()

        max_int = (10 ** self.int_digits) - 1
        min_int = -(10 ** (self.int_digits - 1)) if self.int_digits > 1 else 0
        if self.decimals == 0:
            ival = int(round(num))
            ival = min(max(ival, min_int), max_int)
            return f"{ival:0{self.int_digits}d}" if ival >= 0 else f"-{abs(ival):0{self.int_digits - 1}d}"

        max_abs = max_int + (1 - (10 ** -self.decimals))
        num = max(min(num, max_abs), -max_abs)
        width = self.total_len
        return f"{num:0{width}.{self.decimals}f}"

    def format_na(self) -> str:
        return "NA".rjust(self.total_len)


VITAL_SPECS = [
    VitalSpec("HR", "Heart Rate", 3),
    VitalSpec("SpO2", "SpO2", 3),
    VitalSpec("RR", "Resp Rate", 2),
    VitalSpec("TEMP", "Temperature", 2, 1),
    VitalSpec("NIBP_SYS", "NIBP SYS", 3),
    VitalSpec("NIBP_DIA", "NIBP DIA", 3),
    VitalSpec("NIBP_MEAN", "NIBP MEAN", 3),
    VitalSpec("PR", "Pulse Rate", 3),
    VitalSpec("EtCO2", "EtCO2", 3),
    VitalSpec("CVP", "CVP", 3),
    VitalSpec("ABP_SYS", "ABP SYS", 3),
    VitalSpec("ABP_DIA", "ABP DIA", 3),
    VitalSpec("ABP_MEAN", "ABP MEAN", 3),
    VitalSpec("ICP", "ICP", 3),
    VitalSpec("BT", "Body Temp", 2, 1),
    VitalSpec("CO", "Cardiac Out", 2, 1),
    VitalSpec("CI", "Cardiac Index", 2, 1),
    VitalSpec("SV", "Stroke Vol", 3),
    VitalSpec("SVR", "SVR", 4),
    VitalSpec("FiO2", "FiO2", 3),
]


class VitalCell(QFrame):
    def __init__(self, spec: VitalSpec, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.spec = spec

        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(2)
        self.setFixedSize(440, 78)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(6)

        label = QLabel(spec.label)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        label.setFont(QFont("Noto Sans Mono", 16, QFont.Weight.DemiBold))
        label.setFixedHeight(20)

        self.value_label = QLabel(spec.format_na())
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.value_label.setFont(QFont("DejaVu Sans Mono", 44, QFont.Weight.Bold))
        self.value_label.setFixedHeight(46)

        layout.addWidget(label)
        layout.addWidget(self.value_label)

    def set_value(self, raw: object):
        self.value_label.setText(self.spec.format_value(raw))

    def set_na(self):
        self.value_label.setText(self.spec.format_na())


class BedPanel(QFrame):
    def __init__(self, bed_id: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.bed_id = bed_id
        self.cells: Dict[str, VitalCell] = {}

        self.setFrameShape(QFrame.Shape.Box)
        self.setLineWidth(3)
        self.setFixedSize(1860, 448)

        wrapper = QVBoxLayout(self)
        wrapper.setContentsMargins(12, 12, 12, 12)
        wrapper.setSpacing(10)

        title = QLabel(bed_id)
        title.setFont(QFont("Roboto Mono", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        title.setFixedHeight(34)
        wrapper.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        grid.setContentsMargins(0, 0, 0, 0)

        for idx, spec in enumerate(VITAL_SPECS):
            row = idx // 4
            col = idx % 4
            cell = VitalCell(spec)
            self.cells[spec.key] = cell
            grid.addWidget(cell, row, col)

        wrapper.addLayout(grid)

    def update_values(self, values: Dict[str, object], stale: bool):
        for key, cell in self.cells.items():
            if stale:
                cell.set_na()
            else:
                cell.set_value(values.get(key))


class OCRMonitorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OCR Priority Monitor")
        self.setFixedSize(SCREEN_WIDTH, SCREEN_HEIGHT)

        self.last_mtime = 0.0
        self.last_global_update = 0.0
        self.latest_data: Dict[str, Dict[str, object]] = {}

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(16)

        self.beds: Dict[str, BedPanel] = {}
        for bed_id in BED_IDS:
            panel = BedPanel(bed_id)
            self.beds[bed_id] = panel
            content_layout.addWidget(panel, alignment=Qt.AlignmentFlag.AlignHCenter)

        content_layout.addStretch(1)
        scroll.setWidget(content)
        root_layout.addWidget(scroll)
        self.setCentralWidget(root)

        self.setStyleSheet(
            """
            QMainWindow, QWidget, QScrollArea {
                background-color: #FFFFFF;
                color: #000000;
            }
            QLabel { color: #000000; }
            QFrame {
                background-color: #FFFFFF;
                border-color: #000000;
            }
            """
        )

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(POLL_INTERVAL_MS)

        self.refresh()

    def refresh(self):
        now = time.time()
        self._read_json_if_updated(now)

        stale_global = (now - self.last_global_update) > STALE_TIMEOUT_SEC
        for bed_id, panel in self.beds.items():
            bed_payload = self.latest_data.get(bed_id, {})
            bed_values = bed_payload if isinstance(bed_payload, dict) else {}

            bed_timestamp = bed_payload.get("timestamp") if isinstance(bed_payload, dict) else None
            bed_stale = stale_global
            if isinstance(bed_timestamp, (int, float)):
                bed_stale = (now - float(bed_timestamp)) > STALE_TIMEOUT_SEC
            panel.update_values(bed_values, stale=bed_stale)

    def _read_json_if_updated(self, now: float):
        try:
            mtime = os.path.getmtime(DATA_PATH)
        except OSError:
            return

        if mtime <= self.last_mtime:
            return

        try:
            with DATA_PATH.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        if isinstance(payload, dict):
            self.latest_data = payload
            self.last_mtime = mtime
            self.last_global_update = now


def main() -> int:
    app = QApplication(sys.argv)
    window = OCRMonitorWindow()
    window.showFullScreen()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
