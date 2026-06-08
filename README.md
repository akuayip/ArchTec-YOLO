# ArchDetec YOLO Training

Project ini dipakai untuk training dan membandingkan model YOLO pada dataset arsitektur bangunan tradisional Lampung. Script utama mendukung object detection dan instance segmentation menggunakan Ultralytics YOLO.

## Struktur Project

```text
.
├── train.py          # Runner training detect/segment
├── compare.py        # Visualisasi perbandingan hasil training
├── dataset/          # Dataset YOLO/Roboflow, tidak masuk Git
├── Output_dir/       # Output training dan grafik compare, tidak masuk Git
├── yolov9t.pt        # Weight lokal YOLOv9 tiny, tidak masuk Git
├── pyproject.toml
├── uv.lock
└── .gitignore
```

## Setup

Gunakan `uv run` dari root folder project. Tidak perlu mengaktifkan virtual environment manual.

```bash
uv run python train.py --help
```

Kalau tetap ingin memakai environment manual:

```bash
source .venv/bin/activate
python train.py --help
```

## Dataset

Script memakai file:

```text
dataset/data.yaml
```

Struktur dataset yang diharapkan:

```text
dataset/
├── train/images
├── train/labels
├── valid/images
├── valid/labels
├── test/images
└── test/labels
```

Dataset tidak dimasukkan ke Git karena ukurannya besar dan biasanya dikelola terpisah.

## Training Object Detection

Model detection yang tersedia:

```text
yolo8n
yolo9t
yolo10n
yolo11n
```

Contoh menjalankan satu model:

```bash
uv run python train.py --task detect --model yolo9t --epochs 100 --imgsz 640 --batch 16 --device mps
```

Contoh menjalankan semua model detection berurutan:

```bash
uv run python train.py --task detect --model yolo8n && \
uv run python train.py --task detect --model yolo9t && \
uv run python train.py --task detect --model yolo10n && \
uv run python train.py --task detect --model yolo11n
```

Output default detection:

```text
Output_dir/YOLOv8-Nano-Detect/
Output_dir/YOLOv9-Tiny-Detect/
Output_dir/YOLOv10-Nano-Detect/
Output_dir/YOLO11-Nano-Detect/
```

## Training Segmentation

Model segmentation yang tersedia di setup Ultralytics project ini:

```text
yolo8n
yolo9c
yolo11n
```

Contoh menjalankan satu model:

```bash
uv run python train.py --task segment --model yolo8n --epochs 100 --imgsz 640 --batch 8 --device mps
```

Contoh menjalankan semua model segmentation berurutan:

```bash
uv run python train.py --task segment --model yolo8n && \
uv run python train.py --task segment --model yolo9c && \
uv run python train.py --task segment --model yolo11n
```

Output default segmentation:

```text
Output_dir/YOLOv8-Nano-Segment/
Output_dir/YOLOv9-Compact-Segment/
Output_dir/YOLOv11-Nano-Segment/
```

Catatan: YOLOv10 segmentation tidak tersedia di package Ultralytics yang dipakai project ini. YOLOv9 segmentation juga tidak tersedia untuk `yolo9t`; yang tersedia adalah `yolo9c`.

## Device

Pilih device dengan argumen `--device`.

```bash
# Apple Silicon GPU
uv run python train.py --task detect --model yolo9t --device mps

# NVIDIA CUDA GPU pertama
uv run python train.py --task detect --model yolo9t --device 0

# Multi-GPU CUDA
uv run python train.py --task detect --model yolo9t --device 0,1

# CPU
uv run python train.py --task detect --model yolo9t --device cpu
```

Untuk Mac/MPS, jika training segmentasi berat, coba turunkan batch:

```bash
uv run python train.py --task segment --model yolo11n --batch 4 --workers 0
```

## Compare Hasil Training

Setelah training selesai, setiap folder run akan memiliki:

```text
results.csv
```

Script `compare.py` membaca file tersebut dan membuat grafik perbandingan.

### Compare Detection

```bash
uv run python compare.py --task detect
```

Output:

```text
Output_dir/compare_detection/
├── map50.png
├── map50_95.png
├── box_loss.png
├── class_loss.png
├── precision.png
├── recall.png
├── summary_best_metrics.png
└── summary_best_metrics.csv
```

### Compare Segmentation

```bash
uv run python compare.py --task segment
```

Output:

```text
Output_dir/compare_segmentation/
├── mask_map50.png
├── mask_map50_95.png
├── mask_precision.png
├── mask_recall.png
├── box_loss.png
├── segment_loss.png
├── class_loss.png
├── summary_best_metrics.png
└── summary_best_metrics.csv
```

## File yang Tidak Masuk Git

File dan folder berikut di-ignore:

```text
.venv/
dataset/
Output_dir/
runs/
weights/
*.pt
*.pth
*.onnx
*.engine
*.cache
```

Ini sengaja dilakukan supaya repository GitHub hanya berisi kode, konfigurasi, dan lockfile, bukan dataset, model weight, environment, atau output training.

## Workflow Singkat

1. Siapkan dataset di `dataset/`.
2. Jalankan training detect atau segment dengan `train.py`.
3. Setelah semua model selesai, jalankan `compare.py`.
4. Buka grafik di `Output_dir/compare_detection/` atau `Output_dir/compare_segmentation/`.
