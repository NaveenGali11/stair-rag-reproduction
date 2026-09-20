from pathlib import Path
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/whole-child-results.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/figures"),
    )
    return parser.parse_args()


def label_bars(axis, bars, digits=3):
    for bar in bars:
        height = bar.get_height()
        axis.annotate(
            f"{height:.{digits}f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def main():
    args = parse_args()

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise SystemExit(
            "Install reporting dependencies with "
            "`python -m pip install matplotlib`."
        ) from error

    payload = json.loads(args.results.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    paired = payload["paired_dsi_vs_stair"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    names = ["BM25", "NV-Embed-v2", "DSI", "STAIR"]
    keys = ["bm25", "nvembed", "dsi", "stair"]
    colors = ["#31688e", "#35b779", "#fde725", "#7b3294"]

    figure, axis = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(keys))
    width = 0.24

    for offset, (metric, label) in enumerate(
        (
            ("recall_at_1", "Recall@1"),
            ("recall_at_3", "Recall@3"),
            ("recall_at_5", "Recall@5"),
        )
    ):
        values = [metrics[key][metric] for key in keys]
        bars = axis.bar(
            x + (offset - 1) * width,
            values,
            width,
            label=label,
            alpha=0.88,
        )
        label_bars(axis, bars)

    axis.set_title("Whole Child retrieval on the frozen test split")
    axis.set_ylabel("Recall")
    axis.set_ylim(0, 1.06)
    axis.set_xticks(x, names)
    axis.legend(frameon=False, ncol=3, loc="upper right")
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "retriever-recall.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    depth = paired["by_hierarchy_depth"]
    depth_names = ["Depth 2", "Depth 3"]
    dsi_values = [depth["2"]["dsi_recall_at_1"], depth["3"]["dsi_recall_at_1"]]
    stair_values = [
        depth["2"]["stair_recall_at_1"],
        depth["3"]["stair_recall_at_1"],
    ]

    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    x = np.arange(2)
    width = 0.34
    dsi_bars = axis.bar(
        x - width / 2,
        dsi_values,
        width,
        label="DSI",
        color=colors[2],
    )
    stair_bars = axis.bar(
        x + width / 2,
        stair_values,
        width,
        label="STAIR",
        color=colors[3],
    )
    label_bars(axis, dsi_bars)
    label_bars(axis, stair_bars)
    axis.set_title("Hierarchy depth reverses the DSI–STAIR result")
    axis.set_ylabel("Recall@1")
    axis.set_ylim(0, 0.66)
    axis.set_xticks(x, depth_names)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "dsi-stair-by-depth.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    coverage = paired["by_development_leaf_coverage"]
    coverage_keys = ["seen_in_dev", "absent_from_dev"]
    coverage_names = ["Leaf seen in dev", "Leaf absent from dev"]
    dsi_values = [coverage[key]["dsi_recall_at_1"] for key in coverage_keys]
    stair_values = [coverage[key]["stair_recall_at_1"] for key in coverage_keys]

    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    x = np.arange(2)
    dsi_bars = axis.bar(
        x - width / 2,
        dsi_values,
        width,
        label="DSI",
        color=colors[2],
    )
    stair_bars = axis.bar(
        x + width / 2,
        stair_values,
        width,
        label="STAIR",
        color=colors[3],
    )
    label_bars(axis, dsi_bars)
    label_bars(axis, stair_bars)
    axis.set_title("Development-label coverage explains most of the reversal")
    axis.set_ylabel("Recall@1")
    axis.set_ylim(0, 0.52)
    axis.set_xticks(x, coverage_names)
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(
        args.output_dir / "dsi-stair-by-dev-coverage.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    latency = [metrics[key]["milliseconds_per_query"] for key in keys]
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    bars = axis.bar(names, latency, color=colors)
    axis.set_yscale("log")
    axis.set_title("Measured query latency (log scale)")
    axis.set_ylabel("Milliseconds per query")
    axis.spines[["top", "right"]].set_visible(False)

    for bar, value in zip(bars, latency):
        axis.annotate(
            f"{value:.2f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    figure.tight_layout()
    figure.savefig(
        args.output_dir / "retriever-latency.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
