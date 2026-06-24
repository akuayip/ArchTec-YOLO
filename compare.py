import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "Output_dir"

TASK_CONFIG = {
    "detect": {
        "title": "Detection",
        "default_output": OUTPUT_DIR / "compare_detection",
        "runs": {
            "YOLOv8n": "YOLOv8-Nano-Detect",
            "YOLOv9t": "YOLOv9-Tiny-Detect",
            "YOLOv10n": "YOLOv10-Nano-Detect",
            "YOLOv11n": "YOLO11-Nano-Detect",
        },
        "metrics": {
            "mAP50": "metrics/mAP50(B)",
            "mAP50-95": "metrics/mAP50-95(B)",
            "Box Loss": "train/box_loss",
            "Class Loss": "train/cls_loss",
            "Precision": "metrics/precision(B)",
            "Recall": "metrics/recall(B)",
        },
    },
    "segment": {
        "title": "Segmentation",
        "default_output": OUTPUT_DIR / "compare_segmentation",
        "runs": {
            "YOLOv8n-seg": "YOLOv8-Nano-Segment",
            "YOLOv9c-seg": "YOLOv9-Compact-Segment",
            "YOLOv11n-seg": "YOLOv11-Nano-Segment",
        },
        "metrics": {
            "Mask mAP50": "metrics/mAP50(M)",
            "Mask mAP50-95": "metrics/mAP50-95(M)",
            "Mask Precision": "metrics/precision(M)",
            "Mask Recall": "metrics/recall(M)",
            "Box Loss": "train/box_loss",
            "Segment Loss": "train/seg_loss",
            "Class Loss": "train/cls_loss",
        },
    },
}


def read_results(csv_path: Path) -> list[dict[str, float]]:
    with csv_path.open("r", newline="") as file:
        reader = csv.DictReader(file)
        rows = []
        for row in reader:
            cleaned = {}
            for key, value in row.items():
                if key is None:
                    continue
                normalized_key = key.strip()
                try:
                    cleaned[normalized_key] = float(value)
                except (TypeError, ValueError):
                    continue
            rows.append(cleaned)
    return rows


def load_runs(run_dirs: dict[str, str]) -> tuple[dict[str, list[dict[str, float]]], list[Path]]:
    runs = {}
    missing = []
    for label, folder_name in run_dirs.items():
        csv_path = OUTPUT_DIR / folder_name / "results.csv"
        if not csv_path.exists():
            missing.append(csv_path)
            continue
        rows = read_results(csv_path)
        if rows:
            runs[label] = rows
    return runs, missing


def get_series(rows: list[dict[str, float]], metric_key: str) -> list[float]:
    return [row[metric_key] for row in rows if metric_key in row]


def get_epochs(rows: list[dict[str, float]], series_length: int) -> list[int]:
    if rows and "epoch" in rows[0]:
        return [int(row["epoch"]) + 1 for row in rows[:series_length]]
    return list(range(1, series_length + 1))


def plot_metric_lines(
    runs: dict[str, list[dict[str, float]]],
    metric_name: str,
    metric_key: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))
    plotted = False
    for label, rows in runs.items():
        series = get_series(rows, metric_key)
        if not series:
            continue
        epochs = get_epochs(rows, len(series))
        plt.plot(epochs, series, markersize=3, label=label)
        plotted = True

    if not plotted:
        plt.close()
        return

    plt.title(f"{metric_name} Comparison")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_summary_bars(
    runs: dict[str, list[dict[str, float]]],
    metrics: dict[str, str],
    output_path: Path,
    title: str,
) -> None:
    available_metrics = []
    summary_values = {}

    for metric_name, metric_key in metrics.items():
        values = []
        labels = []
        for label, rows in runs.items():
            series = get_series(rows, metric_key)
            if not series:
                continue
            value = min(series) if "Loss" in metric_name else max(series)
            labels.append(label)
            values.append(value)

        if values:
            available_metrics.append(metric_name)
            summary_values[metric_name] = (labels, values)

    if not available_metrics:
        return

    columns = 3
    rows = 3 if len(available_metrics) > 6 else 2
    fig, axes = plt.subplots(rows, columns, figsize=(16, 4.5 * rows))
    axes = axes.flatten()

    for index, metric_name in enumerate(available_metrics[: len(axes)]):
        ax = axes[index]
        labels, values = summary_values[metric_name]
        ax.bar(labels, values, color=["#2563eb", "#16a34a", "#dc2626", "#7c3aed"][: len(labels)])
        ax.set_title(metric_name)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)

    for index in range(len(available_metrics), len(axes)):
        axes[index].axis("off")

    fig.suptitle(f"Best {title} Metrics by Model", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_summary_csv(
    runs: dict[str, list[dict[str, float]]],
    metrics: dict[str, str],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", *metrics.keys()])
        for label, rows in runs.items():
            row = [label]
            for metric_name, metric_key in metrics.items():
                series = get_series(rows, metric_key)
                if not series:
                    row.append("")
                    continue
                row.append(min(series) if "Loss" in metric_name else max(series))
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare YOLO object detection or segmentation training results."
    )
    parser.add_argument(
        "--task",
        choices=TASK_CONFIG.keys(),
        default="detect",
        help="Which training task to compare.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Folder to save comparison plots and summary CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TASK_CONFIG[args.task]
    output_dir = args.output or config["default_output"]
    output_dir.mkdir(parents=True, exist_ok=True)

    runs, missing = load_runs(config["runs"])
    if missing:
        print("Missing results.csv files:")
        for path in missing:
            print(f"- {path}")

    if not runs:
        print(f"No {args.task} results found. Train at least one {args.task} model first.")
        return

    for metric_name, metric_key in config["metrics"].items():
        filename = metric_name.lower().replace(" ", "_").replace("-", "_")
        plot_metric_lines(runs, metric_name, metric_key, output_dir / f"{filename}.png")

    plot_summary_bars(
        runs,
        config["metrics"],
        output_dir / "summary_best_metrics.png",
        config["title"],
    )
    write_summary_csv(runs, config["metrics"], output_dir / "summary_best_metrics.csv")

    print(f"Comparison saved to: {output_dir}")


if __name__ == "__main__":
    main()
