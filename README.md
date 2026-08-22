# License Plate Detection & Blurring with YOLOv8

![Python](https://img.shields.io/badge/python-3.12.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![YOLOv8](https://img.shields.io/badge/built%20with-YOLOv8-purple)

Automated pipeline that detects vehicle license plates in images and anonymizes them with blurring — built to help teams handling automotive image/video data comply with privacy regulations (GDPR, CCPA) without losing the utility of the surrounding visual data.

## Overview

License plates are personally identifiable information (PII). Before automotive imagery can be used for public datasets, model training, or insurance/claims processing, plates need to be reliably located and obscured — at scale, and fast enough for real-time or near-real-time pipelines.

This project fine-tunes a **YOLOv8m** object detector on ~27,000 annotated images to localize license plates, then applies and quantitatively compares two anonymization strategies (Gaussian vs. box blur) to determine which one actually makes plates unreadable.

## Results

| Metric | Value |
|---|---|
| Model | YOLOv8m (single class: `license_plate`) |
| Training images | 25,470 |
| Validation images | 1,000 |
| Test images | 400 |
| Precision | 0.916 |
| Recall | 0.898 |
| mAP@50 | 0.919 |
| mAP@50-95 | 0.672 |
| Inference latency | ~8 ms/image (~125 FPS) on an RTX 3090 |

**Blur method comparison** (lower Laplacian variance = less readable = more effective anonymization):

| Blur type | Mean Laplacian variance | Min | Max |
|---|---|---|---|
| Gaussian | 4.03 | 2.24 | 6.63 |
| Box | 3.86 | 1.92 | 9.34 |

Visual inspection showed plates were still partially legible after Gaussian blur, so the pipeline defaults to **box blur**, which produced consistently unreadable plates despite a similar mean variance — illustrating why a single averaged metric isn't enough and visual QA mattered here.

## How it works

1. **Data audit** – inspect class balance and visually spot-check YOLO-format bounding box annotations across train/val/test splits before trusting them for training.
2. **Train** – fine-tune YOLOv8m (chosen for its accuracy/speed balance on a large, visually diverse dataset — plate formats, lighting, camera angles) on the labeled plate dataset.
3. **Evaluate** – compute precision, recall, and mAP@50-95 on the held-out test split.
4. **Detect + blur** – run inference on new images and apply blur to each detected bounding box region.
5. **Measure anonymization quality** – score each blurred region with Laplacian variance (a standard blur/sharpness metric) to verify plates are actually unreadable, not just detected.

## Tech stack

- **Model**: YOLOv8 (Ultralytics)
- **Core libraries**: PyTorch, OpenCV, NumPy, Pandas, Matplotlib/Seaborn
- **Compute**: Rented NVIDIA RTX 3090 GPU via [vast.ai](https://vast.ai) (cloud GPU marketplace)

## Project structure

```
├── README.md
├── LICENSE
├── requirements.txt
├── detect_and_blur.py                                   # CLI: run detection + blurring on images/video
├── data.yaml                                             # YOLO dataset config
├── notebooks/
│   └── License_Plate_Detection_Blurring_MH_vastai.ipynb  # end-to-end training/eval notebook
├── weights/
│   └── best.pt                                           # trained YOLOv8m weights
└── sample_images/
    ├── before/                                            # a few example input images
    └── after/                                              # matching blurred outputs
```

## Setup

This notebook was run on a rented GPU instance (vast.ai) rather than locally or on Colab.

```bash
pip install -r requirements.txt
```

1. Provision a GPU instance (e.g. via vast.ai) with CUDA available.
2. Place your dataset under `./data/` following the structure above, with a `data.yaml` pointing to the train/val/test splits.
3. Run the notebook cells in order: data audit → training → evaluation → inference → blurring.

> Note: paths in the notebook (e.g. `/workspace/runs/...`) reflect the specific vast.ai instance layout used during development — update these to match your own environment.

## Usage

`detect_and_blur.py` runs the detect-and-blur pipeline directly from the command line — on a single image, a folder of images, or a video — without needing to open the notebook.

```bash
# Single image
python detect_and_blur.py --weights ./weights/best.pt --input car.jpg --output out.jpg

# Directory of images
python detect_and_blur.py --weights ./weights/best.pt --input ./sample_images/before --output ./sample_images/after

# Video
python detect_and_blur.py --weights ./weights/best.pt --input drive.mp4 --output drive_blurred.mp4

# Gaussian blur instead of the default box blur, plus a readability report
python detect_and_blur.py --weights ./weights/best.pt --input car.jpg --output out.jpg \
    --blur-type gaussian --report
```

**Key options:**

| Flag | Default | Description |
|---|---|---|
| `--weights`, `-w` | *required* | Path to trained YOLOv8 weights |
| `--input`, `-i` | *required* | Image file, directory of images, or video file |
| `--output`, `-o` | *required* | Output file or directory |
| `--blur-type` | `box` | `box` or `gaussian` — box blur was more consistently effective in testing (see [Results](#results)) |
| `--kernel-size` | `27` | Blur kernel size in pixels |
| `--conf` | `0.25` | Detection confidence threshold |
| `--iou` | `0.7` | IoU threshold for non-max suppression |
| `--pad` | `0.0` | Extra margin around each box before blurring, as a fraction of box size |
| `--device` | auto | Inference device, e.g. `cpu`, `cuda:0` |
| `--report` | off | Print Laplacian variance stats for the blurred regions |

## Dataset

The training set used here is a large (~27k image) license-plate detection dataset in YOLO annotation format. **The full dataset is not redistributed in this repository for licensing reasons.** A small sample of images is included under `sample_images/` for demonstration — see [Usage](#usage) above to try the pipeline on your own images.

## Future improvements

- Wrap `detect_and_blur.py` in a Streamlit/Gradio demo for an interactive, no-install way to try the pipeline
- Add tracking-assisted video blurring (e.g. ByteTrack) so plates stay blurred smoothly across frames instead of being detected independently each frame
- Quantize/export to ONNX or TensorRT for edge deployment
- Expand to multi-class detection (e.g. faces) for full PII redaction pipelines
- Add unit tests and a small CI pipeline for `detect_and_blur.py`

## License

[MIT](LICENSE)
