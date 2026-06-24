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
        "yolo26n": "yolo26n.pt",
        "yolo26s": "yolo26s.pt",
        "yolo26m": "yolo26m.pt",
        "yolo26l": "yolo26l.pt",
        "yolo26x": "yolo26x.pt",
    },
    "segment": {
        "yolo8n": "yolov8n-seg.pt",
        "yolo9c": "yolov9c-seg.pt",
        "yolo11n": "yolo11n-seg.pt",
        "yolo26n": "yolo26n-seg.pt",
        "yolo26s": "yolo26s-seg.pt",
        "yolo26m": "yolo26m-seg.pt",
        "yolo26l": "yolo26l-seg.pt",
        "yolo26x": "yolo26x-seg.pt",
    },
}

RUN_NAMES = {
    "detect": {
        "yolo8n": "YOLOv8-Nano-Detect",
        "yolo9t": "YOLOv9-Tiny-Detect",
        "yolo10n": "YOLOv10-Nano-Detect",
        "yolo11n": "YOLOv11-Nano-Detect",
        "yolo26n": "YOLOv26-Nano-Detect",
        "yolo26s": "YOLOv26-Small-Detect",
        "yolo26m": "YOLOv26-Medium-Detect",
        "yolo26l": "YOLOv26-Large-Detect",
        "yolo26x": "YOLOv26-XL-Detect",
    },
    "segment": {
        "yolo8n": "YOLOv8-Nano-Segment",
        "yolo9c": "YOLOv9-Compact-Segment",
        "yolo11n": "YOLOv11-Nano-Segment",
        "yolo26n": "YOLOv26-Nano-Segment",
        "yolo26s": "YOLOv26-Small-Segment",
        "yolo26m": "YOLOv26-Medium-Segment",
        "yolo26l": "YOLOv26-Large-Segment",
        "yolo26x": "YOLOv26-XL-Segment",
    },
}


def default_batch(task: str, model_key: str) -> int:
    if task == "segment" and model_key == "yolo26x":
        return 2
    if task == "segment" and model_key == "yolo9c":
        return 2
    if task == "segment" and model_key in {"yolo26m", "yolo26l"}:
        return 4
    if task == "segment":
        return 8
    return 16


def default_workers(task: str, model_key: str) -> int:
    if task == "segment" and model_key in {"yolo9c", "yolo26m", "yolo26l", "yolo26x"}:
        return 4
    return 8


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
            "YOLO model variant. Detect: yolo8n, yolo9t, yolo10n, yolo11n, "
            "yolo26n, yolo26s, yolo26m, yolo26l, yolo26x. "
            "Segment: yolo8n, yolo9c, yolo11n, yolo26n, yolo26s, "
            "yolo26m, yolo26l, yolo26x."
        ),
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Batch size. Defaults depend on task and model.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device: auto, mps, cpu, or 0 for CUDA.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Dataloader workers. Defaults depend on task and model.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use automatic mixed precision.",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache dataset images. Leave off if RAM is limited.",
    )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save training plots and batch previews.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic training algorithms when possible.",
    )
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
    device = None if args.device == "auto" else args.device
    batch = args.batch if args.batch is not None else default_batch(args.task, args.model)
    workers = args.workers if args.workers is not None else default_workers(args.task, args.model)

    model = YOLO(weight)
    model.train(
        data=str(DATA_YAML),
        task=args.task,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        project=str(OUTPUT_DIR),
        name=run_name,
        workers=workers,
        amp=args.amp,
        cache=args.cache,
        plots=args.plots,
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()
