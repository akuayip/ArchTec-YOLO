import argparse
import subprocess
import sys


BASELINE_DETECT_MODELS = ["yolo8n", "yolo9t", "yolo10n", "yolo11n"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8/v9/v10/v11 object detection experiments sequentially."
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--start-from",
        choices=BASELINE_DETECT_MODELS,
        default="yolo8n",
        help="Resume the sequence from this model.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable training plots.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue to the next model if one experiment fails.",
    )
    return parser.parse_args()


def build_command(model: str, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "train.py",
        "--task",
        "detect",
        "--model",
        model,
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--device",
        args.device,
        "--workers",
        str(args.workers),
    ]

    if args.no_amp:
        command.append("--no-amp")
    if args.no_plots:
        command.append("--no-plots")

    return command


def main() -> None:
    args = parse_args()
    start_index = BASELINE_DETECT_MODELS.index(args.start_from)
    models = BASELINE_DETECT_MODELS[start_index:]

    for model in models:
        command = build_command(model, args)
        print(f"\n=== Running {model} object detection ===", flush=True)
        print(" ".join(command), flush=True)

        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"Experiment {model} failed with exit code {result.returncode}.", flush=True)
            if not args.keep_going:
                sys.exit(result.returncode)

    print("\nAll requested baseline object detection experiments finished.", flush=True)


if __name__ == "__main__":
    main()
