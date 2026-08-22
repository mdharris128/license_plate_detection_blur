#!/usr/bin/env python3
"""
detect_and_blur.py

Detect license plates with a trained YOLOv8 model and anonymize them by
blurring the detected regions. Works on a single image, a directory of
images, or a video file.

Usage
-----
    # Single image
    python detect_and_blur.py --weights best.pt --input car.jpg --output out.jpg

    # Directory of images
    python detect_and_blur.py --weights best.pt --input ./images --output ./blurred

    # Video
    python detect_and_blur.py --weights best.pt --input drive.mp4 --output drive_blurred.mp4

    # Use Gaussian blur instead of the default box blur, and report how
    # unreadable the blurred regions are (Laplacian variance)
    python detect_and_blur.py --weights best.pt --input car.jpg --output out.jpg \\
        --blur-type gaussian --report

Notes
-----
Box blur is the default because, on this project's validation set, it left
license plates consistently unreadable while Gaussian blur (same kernel
size) sometimes did not — see the project README for the comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect license plates and blur them for anonymization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights", "-w", required=True, type=Path,
        help="Path to trained YOLOv8 weights (e.g. best.pt).",
    )
    parser.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Input image file, directory of images, or video file.",
    )
    parser.add_argument(
        "--output", "-o", required=True, type=Path,
        help="Output file (image/video) or directory (batch of images).",
    )
    parser.add_argument(
        "--blur-type", choices=["box", "gaussian"], default="box",
        help="Blur algorithm applied to each detected plate region.",
    )
    parser.add_argument(
        "--kernel-size", type=int, default=27,
        help="Blur kernel size in pixels. Must be odd for Gaussian blur.",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--iou", type=float, default=0.7,
        help="IoU threshold used for non-max suppression.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Inference device, e.g. 'cpu', 'cuda:0'. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--pad", type=float, default=0.0,
        help=(
            "Extra margin added around each detected box before blurring, "
            "as a fraction of the box size (e.g. 0.1 = 10%% larger on each "
            "side). Useful if plates are sometimes cut off by tight boxes."
        ),
    )
    parser.add_argument(
        "--report", action="store_true",
        help=(
            "Print Laplacian variance stats for the blurred regions "
            "(lower = less readable = more effective anonymization)."
        ),
    )
    return parser.parse_args()


def load_model(weights: Path, device: str | None):
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "ultralytics is not installed. Run: pip install ultralytics"
        )
    if not weights.exists():
        sys.exit(f"Weights file not found: {weights}")
    model = YOLO(str(weights))
    if device:
        model.to(device)
    return model


def blur_regions(
    image: np.ndarray,
    boxes: np.ndarray,
    blur_type: str,
    kernel_size: int,
    pad: float,
) -> tuple[np.ndarray, list[float]]:
    """Blur each detected box region in `image`. Returns the blurred image
    and the Laplacian variance of every blurred region (for reporting)."""
    h, w = image.shape[:2]
    out = image.copy()
    lap_variances: list[float] = []

    ksize = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    for x1, y1, x2, y2 in boxes:
        box_w, box_h = x2 - x1, y2 - y1
        x1 -= box_w * pad
        x2 += box_w * pad
        y1 -= box_h * pad
        y2 += box_h * pad

        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue

        roi = out[y1:y2, x1:x2]
        if blur_type == "gaussian":
            blurred_roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)
        else:
            blurred_roi = cv2.blur(roi, (ksize, ksize))
        out[y1:y2, x1:x2] = blurred_roi

        gray = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2GRAY)
        lap_variances.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

    return out, lap_variances


def process_image(model, image_path: Path, args: argparse.Namespace) -> list[float]:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  [skip] could not read {image_path}")
        return []

    results = model.predict(
        image, conf=args.conf, iou=args.iou, device=args.device, verbose=False
    )
    boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) else np.empty((0, 4))

    blurred, lap_vars = blur_regions(
        image, boxes, args.blur_type, args.kernel_size, args.pad
    )
    return blurred, lap_vars, len(boxes)


def run_on_image(model, in_path: Path, out_path: Path, args: argparse.Namespace) -> list[float]:
    blurred, lap_vars, n = process_image(model, in_path, args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), blurred)
    print(f"  {in_path.name}: {n} plate(s) blurred -> {out_path}")
    return lap_vars


def run_on_directory(model, in_dir: Path, out_dir: Path, args: argparse.Namespace) -> list[float]:
    image_paths = sorted(
        p for p in in_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        sys.exit(f"No images found in {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    all_lap_vars: list[float] = []
    for p in image_paths:
        all_lap_vars.extend(run_on_image(model, p, out_dir / p.name, args))
    return all_lap_vars


def run_on_video(model, in_path: Path, out_path: Path, args: argparse.Namespace) -> list[float]:
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {in_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    all_lap_vars: list[float] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        results = model.predict(
            frame, conf=args.conf, iou=args.iou, device=args.device, verbose=False
        )
        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) else np.empty((0, 4))
        blurred, lap_vars = blur_regions(
            frame, boxes, args.blur_type, args.kernel_size, args.pad
        )
        all_lap_vars.extend(lap_vars)
        writer.write(blurred)

        if frame_idx % 30 == 0 or frame_idx == total_frames:
            print(f"  frame {frame_idx}/{total_frames or '?'}", end="\r")

    cap.release()
    writer.release()
    print(f"\n  video written -> {out_path}")
    return all_lap_vars


def print_report(lap_vars: list[float]) -> None:
    if not lap_vars:
        print("\nNo plate regions were blurred — nothing to report.")
        return
    arr = np.array(lap_vars)
    print("\n--- Blur effectiveness (Laplacian variance) ---")
    print(f"  regions blurred : {len(arr)}")
    print(f"  mean            : {arr.mean():.2f}")
    print(f"  min / max       : {arr.min():.2f} / {arr.max():.2f}")
    print("  (lower = less detail retained = more effective anonymization)")


def main() -> None:
    args = parse_args()
    model = load_model(args.weights, args.device)

    suffix = args.input.suffix.lower()
    print(f"Loading model from {args.weights} ...")

    if args.input.is_dir():
        print(f"Processing images in {args.input} ...")
        lap_vars = run_on_directory(model, args.input, args.output, args)
    elif suffix in VIDEO_EXTENSIONS:
        print(f"Processing video {args.input} ...")
        lap_vars = run_on_video(model, args.input, args.output, args)
    elif suffix in IMAGE_EXTENSIONS:
        print(f"Processing image {args.input} ...")
        lap_vars = run_on_image(model, args.input, args.output, args)
    else:
        sys.exit(f"Unrecognized input type: {args.input}")

    if args.report:
        print_report(lap_vars)


if __name__ == "__main__":
    main()
