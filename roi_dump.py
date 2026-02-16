#!/usr/bin/env python3
"""Dump all ROI crops for a single image using the same ROI logic as ocr_capture_app.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from ocr_capture_app import BED_IDS, DEFAULT_CONFIG, VITAL_ORDER, build_vital_rois, ensure_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROI dumper for one image")
    parser.add_argument("--image", required=True, help="Input image path (e.g. dataset/...png)")
    parser.add_argument("--outdir", required=True, help="Output directory (e.g. dataset/.../roi_dump)")
    parser.add_argument("--config", default=None, help="Optional config json path")
    return parser.parse_args()


def load_roi_config(config_path: str | None) -> dict:
    if config_path:
        return ensure_config(Path(config_path))
    return dict(DEFAULT_CONFIG)


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    outdir = Path(args.outdir)

    frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    outdir.mkdir(parents=True, exist_ok=True)

    config = load_roi_config(args.config)
    rois, _debug = build_vital_rois(frame, config, return_debug=True)

    debug_img = frame.copy()

    for bed in BED_IDS:
        bed_rois = rois.get(bed, {})
        for key in VITAL_ORDER:
            bbox = bed_rois.get(key)
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            crop = frame[y1:y2, x1:x2]
            out_path = outdir / f"roi_{bed}_{key}.png"
            cv2.imwrite(str(out_path), crop)

            cv2.rectangle(debug_img, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 1)

    debug_path = outdir / "debug_rois.png"
    cv2.imwrite(str(debug_path), debug_img)

    print(f"[INFO] image: {image_path}")
    print(f"[INFO] outdir: {outdir}")
    print(f"[INFO] debug image: {debug_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
