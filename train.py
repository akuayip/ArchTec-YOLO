import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parent
DATA_YAML = ROOT_DIR / "dataset" / "data.yaml"
OUTPUT_DIR = ROOT_DIR / "Output_dir"

MODEL_WEIGHTS = {
    "detect": {
        "yolo8n": "yolov8n.pt",
        "yolo9t": "yolov9t.pt",
        "yolo10n": "yolov10n.pt",
        "yolo11n": "yolo11n.pt",
    },
    "segment": {
        "yolo8n": "yolov8n-seg.pt",
        "yolo9c": "yolov9c-seg.pt",
        "yolo11n": "yolo11n-seg.pt",
    },
}

RUN_NAMES = {
    "detect": {
        "yolo8n": "YOLOv8-Nano-Detect",
        "yolo9t": "YOLOv9-Tiny-Detect",
        "yolo10n": "YOLOv10-Nano-Detect",
        "yolo11n": "YOLO11-Nano-Detect",
    },
    "segment": {
        "yolo8n": "YOLOv8-Nano-Segment",
        "yolo9c": "YOLOv9-Compact-Segment",
        "yolo11n": "YOLO11-Nano-Segment",
    },
}


def resolve_weight(task: str, model_key: str) -> str:
    """Use local weights when present, otherwise let Ultralytics download them."""
    weight_name = MODEL_WEIGHTS[task][model_key]
    local_weight = ROOT_DIR / weight_name
    return str(local_weight if local_weight.exists() else weight_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO detection or segmentation models on the ArchDetec dataset."
    )
    parser.add_argument(
        "--task",
        choices=MODEL_WEIGHTS.keys(),
        default="detect",
        help="Training task.",
    )
    parser.add_argument(
        "--model",
        default="yolo9t",
        help=(
            "YOLO model variant. Detect: yolo8n, yolo9t, yolo10n, yolo11n. "
            "Segment: yolo8n, yolo9c, yolo11n."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--device",
        default="mps",
        help="Training device, for example mps, cpu, or 0 for CUDA.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    parser.add_argument(
        "--name",
        default=None,
        help="Output run name. Defaults to the selected model name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model not in MODEL_WEIGHTS[args.task]:
        available = ", ".join(MODEL_WEIGHTS[args.task])
        raise ValueError(
            f"Model '{args.model}' is not available for task '{args.task}'. "
            f"Available models: {available}"
        )

    weight = resolve_weight(args.task, args.model)
    run_name = args.name or RUN_NAMES[args.task][args.model]

    model = YOLO(weight)
    model.train(
        data=str(DATA_YAML),
        task=args.task,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(OUTPUT_DIR),
        name=run_name,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
