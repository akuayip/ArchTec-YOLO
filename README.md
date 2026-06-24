# ArchDetec YOLO Experiments

Project ini berisi runner training dan visualisasi perbandingan model YOLO untuk dataset arsitektur bangunan tradisional Lampung. Task yang didukung:

- Object detection
- Instance segmentation

Dataset, model weight, virtual environment, dan output training sengaja tidak dimasukkan ke Git.

## Struktur

```text
.
├── train.py                    # Runner utama untuk detect/segment
├── compare.py                  # Visualisasi perbandingan results.csv
├── train_baseline_detects.py   # Run detect yolo8n, yolo9t, yolo10n, yolo11n
├── train_yolo26_detects.py     # Run detect yolo26n/s/m/l/x
├── train_yolo26_segments.py    # Run segment yolo26n/s/m/l/x
├── dataset/                    # Dataset YOLO, ignored
├── Output_dir/                 # Hasil training dan compare, ignored
├── pyproject.toml
├── uv.lock
└── .gitignore
```

## Setup

Gunakan `uv run` dari root project. Tidak perlu mengaktifkan `.venv` manual.

```bash
uv run python train.py --help
```

Project dikunci untuk Python `>=3.12,<3.14`. Jika environment di cloud terlanjur memakai Python 3.14, buat ulang:

```bash
rm -rf .venv
uv python install 3.12
uv sync --python 3.12
```

Cek CUDA:

```bash
uv run python -c "import sys, torch; print(sys.version); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0))"
```

## Dataset

File konfigurasi dataset:

```text
dataset/data.yaml
```

Struktur yang diharapkan:

```text
dataset/
├── train/images
├── train/labels
├── valid/images
├── valid/labels
├── test/images
└── test/labels
```

## Training Tunggal

Format umum:

```bash
uv run python train.py --task <detect|segment> --model <model_name> --epochs 100 --imgsz 640 --batch 16 --device 0
```

Device:

```text
auto  # otomatis
0     # CUDA GPU pertama
0,1   # multi-GPU CUDA
mps   # Apple Silicon
cpu   # CPU
```

Contoh:

```bash
uv run python train.py --task detect --model yolo9t --epochs 100 --imgsz 640 --batch 16 --device 0
uv run python train.py --task segment --model yolo11n --epochs 100 --imgsz 640 --batch 16 --device 0
```

## Model Detection

Baseline detection:

```text
yolo8n
yolo9t
yolo10n
yolo11n
```

YOLO26 detection:

```text
yolo26n
yolo26s
yolo26m
yolo26l
yolo26x
```

Run semua baseline detection dalam satu command:

```bash
uv run python train_baseline_detects.py --epochs 100 --imgsz 640 --batch 16 --device 0 --workers 8
```

Resume baseline detection dari model tertentu:

```bash
uv run python train_baseline_detects.py --start-from yolo10n --epochs 100 --imgsz 640 --batch 16 --device 0 --workers 8
```

Run semua YOLO26 detection:

```bash
uv run python train_yolo26_detects.py --epochs 100 --imgsz 640 --batch 16 --device 0 --workers 8
```

## Model Segmentation

Baseline segmentation:

```text
yolo8n
yolo9c
yolo11n
```

Catatan: YOLOv10 segmentation tidak tersedia di setup ini. YOLOv9 segmentation yang tersedia adalah `yolo9c`, bukan `yolo9t`.

YOLO26 segmentation:

```text
yolo26n
yolo26s
yolo26m
yolo26l
yolo26x
```

Run semua YOLO26 segmentation:

```bash
uv run python train_yolo26_segments.py --epochs 100 --imgsz 640 --batch 16 --device 0 --workers 8
```

Resume YOLO26 segmentation dari model tertentu:

```bash
uv run python train_yolo26_segments.py --start-from yolo26m --epochs 100 --imgsz 640 --batch 16 --device 0 --workers 8
```

Jika model besar seperti `yolo26x` kehabisan VRAM, turunkan batch:

```bash
uv run python train.py --task segment --model yolo26x --epochs 100 --imgsz 640 --batch 2 --workers 4 --device 0 --no-deterministic
```

## Output Folder

Semua hasil training masuk ke:

```text
Output_dir/
```

Contoh output detection:

```text
Output_dir/YOLOv8-Nano-Detect/
Output_dir/YOLOv9-Tiny-Detect/
Output_dir/YOLOv10-Nano-Detect/
Output_dir/YOLOv11-Nano-Detect/
Output_dir/YOLOv26-Nano-Detect/
Output_dir/YOLOv26-Small-Detect/
Output_dir/YOLOv26-Medium-Detect/
Output_dir/YOLOv26-Large-Detect/
Output_dir/YOLOv26-XL-Detect/
```

Contoh output segmentation:

```text
Output_dir/YOLOv8-Nano-Segment/
Output_dir/YOLOv9-Compact-Segment/
Output_dir/YOLOv11-Nano-Segment/
Output_dir/YOLOv26-Nano-Segment/
Output_dir/YOLOv26-Small-Segment/
Output_dir/YOLOv26-Medium-Segment/
Output_dir/YOLOv26-Large-Segment/
Output_dir/YOLOv26-XL-Segment/
```

Isi umum tiap folder:

```text
results.csv
weights/best.pt
weights/last.pt
results.png
confusion_matrix.png
train_batch*.jpg
val_batch*_pred.jpg
```

## Compare

Compare membaca `results.csv` dari folder training.

Baseline + YOLO26 detection:

```bash
uv run python compare.py --task detect
```

Khusus YOLO26 detection:

```bash
uv run python compare.py --task detect26
```

Baseline + YOLO26 segmentation:

```bash
uv run python compare.py --task segment
```

Khusus YOLO26 segmentation:

```bash
uv run python compare.py --task segment26
```

Output compare:

```text
Output_dir/compare_detection/
Output_dir/compare_yolo26_detection/
Output_dir/compare_segmentation/
Output_dir/compare_yolo26_segmentation/
```

## Troubleshooting

Jika muncul cuDNN/CUDA error:

```text
CUDNN_STATUS_INTERNAL_ERROR
CUBLAS_STATUS_ALLOC_FAILED
```

Coba:

```bash
nvidia-smi
uv run python train.py --task segment --model yolo26x --epochs 1 --imgsz 640 --batch 2 --workers 4 --device 0 --no-deterministic
```

Jika masih error, turunkan batch dan matikan AMP:

```bash
uv run python train.py --task segment --model yolo26x --epochs 1 --imgsz 512 --batch 1 --workers 2 --device 0 --no-amp --no-deterministic
```

Cek apakah proses dibunuh sistem:

```bash
dmesg -T | grep -i -E "killed process|out of memory|oom"
```

## Git Ignore

Yang sengaja tidak masuk Git:

```text
.venv/
dataset/
Output_dir/
runs/
weights/
wandb/
mlruns/
*.pt
*.pth
*.onnx
*.engine
*.cache
```

Repository sebaiknya hanya berisi source code, konfigurasi, README, dan lockfile.
