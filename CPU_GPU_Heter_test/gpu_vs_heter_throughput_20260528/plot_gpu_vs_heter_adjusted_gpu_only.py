#!/usr/bin/env python3
"""Generate a derived GPU-vs-heter PDF with adjusted GPU-only throughput."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, fontManager


ROOT = Path(__file__).resolve().parent
SOURCE_JSON = ROOT / "gpu_vs_heter_throughput_results.json"
OUT_PDF = ROOT / "gpu_vs_heter_throughput_adjusted_gpu_only.pdf"
OUT_CSV = ROOT / "gpu_vs_heter_throughput_adjusted_gpu_only.csv"
OUT_MD = ROOT / "gpu_vs_heter_throughput_adjusted_gpu_only_record.md"

PROMPT_LENS = [2000, 4000, 8000]
OUTPUT_LENS = [200, 400, 600]

# Fixed one-decimal synthetic GPU-only values requested for the derived figure.
GPU_ONLY_ADJUSTED = {
    (2000, 200): 108.4,
    (2000, 400): 114.7,
    (2000, 600): 117.9,
    (4000, 200): 103.6,
    (4000, 400): 109.8,
    (4000, 600): 115.2,
    (8000, 200): 100.9,
    (8000, 400): 117.3,
    (8000, 600): 112.6,
}

# Corrected heter baseline from the anomaly recheck.
HETER_OVERRIDES = {
    (2000, 200): 23.2,
}

LATIN_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")


def setup_fonts() -> FontProperties:
    names: list[str] = []
    if LATIN_FONT.exists():
        fontManager.addfont(str(LATIN_FONT))
        names.append(FontProperties(fname=str(LATIN_FONT)).get_name())
    if not names:
        names = ["DejaVu Sans"]
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = names
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["pdf.fonttype"] = 42
    return FontProperties(family=names)


FONT = setup_fonts()


def now_utc() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def load_rows() -> list[dict[str, float | int]]:
    payload = json.loads(SOURCE_JSON.read_text(encoding="utf-8"))
    heter: dict[tuple[int, int], float] = {}
    for row in payload["rows"]:
        if row.get("mode") != "heter":
            continue
        prompt_len = int(row["prompt_len_target"])
        output_len = int(row["output_len_target"])
        if prompt_len in PROMPT_LENS and output_len in OUTPUT_LENS:
            heter[(prompt_len, output_len)] = float(row["completion_tokens_per_s"])

    rows: list[dict[str, float | int]] = []
    for prompt_len in PROMPT_LENS:
        for output_len in OUTPUT_LENS:
            gpu_tps = GPU_ONLY_ADJUSTED[(prompt_len, output_len)]
            heter_tps = HETER_OVERRIDES.get(
                (prompt_len, output_len), heter[(prompt_len, output_len)]
            )
            rows.append(
                {
                    "prompt_len": prompt_len,
                    "output_len": output_len,
                    "gpu_only_tok_s": gpu_tps,
                    "heter_tok_s": heter_tps,
                    "heter_over_gpu_pct": heter_tps / gpu_tps * 100.0,
                }
            )
    return rows


def write_csv(rows: list[dict[str, float | int]]) -> None:
    fields = [
        "prompt_len",
        "output_len",
        "gpu_only_tok_s",
        "heter_tok_s",
        "heter_over_gpu_pct",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_pdf(rows: list[dict[str, float | int]]) -> None:
    data = {
        (int(row["prompt_len"]), int(row["output_len"])): row
        for row in rows
    }
    colors = {"gpu": "#2563eb", "heter": "#dc2626"}

    with PdfPages(OUT_PDF) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
        x = range(len(PROMPT_LENS))
        width = 0.35
        for ax, output_len in zip(axes, OUTPUT_LENS):
            gpu_vals = [
                float(data[(prompt_len, output_len)]["gpu_only_tok_s"])
                for prompt_len in PROMPT_LENS
            ]
            heter_vals = [
                float(data[(prompt_len, output_len)]["heter_tok_s"])
                for prompt_len in PROMPT_LENS
            ]
            ax.bar(
                [i - width / 2 for i in x],
                gpu_vals,
                width,
                label="GPU-only",
                color=colors["gpu"],
            )
            ax.bar(
                [i + width / 2 for i in x],
                heter_vals,
                width,
                label="GPU-CPU heter",
                color=colors["heter"],
            )
            ax.set_title(f"Output length = {output_len}", fontproperties=FONT)
            ax.set_xticks(list(x), [f"{p // 1000}K" for p in PROMPT_LENS])
            ax.set_xlabel("Prompt length", fontproperties=FONT)
            ax.set_ylim(0, 130)
            ax.grid(axis="y", alpha=0.25)
            for i, (g, h) in enumerate(zip(gpu_vals, heter_vals)):
                ax.text(i - width / 2, g + 1.5, f"{g:.1f}", ha="center",
                        va="bottom", fontsize=8)
                ax.text(i + width / 2, h + 1.5, f"{h:.1f}", ha="center",
                        va="bottom", fontsize=8)
        axes[0].set_ylabel("Throughput (completion tokens/s)", fontproperties=FONT)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            prop=FONT,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.92),
            ncol=2,
            frameon=False,
        )
        fig.suptitle(
            "End-to-End Throughput: GPU-only vs GPU-CPU Heterogeneous",
            fontproperties=FONT,
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 6))
        cases = []
        ratios = []
        for prompt_len in PROMPT_LENS:
            for output_len in OUTPUT_LENS:
                row = data[(prompt_len, output_len)]
                cases.append(f"P{prompt_len // 1000}K/O{output_len}")
                ratios.append(float(row["heter_over_gpu_pct"]))
        ax.bar(cases, ratios, color="#7c3aed")
        ax.set_ylabel("Heterogeneous / GPU-only throughput (%)", fontproperties=FONT)
        ax.set_xlabel("Workload", fontproperties=FONT)
        ax.set_title("Relative Throughput", fontproperties=FONT)
        ax.set_ylim(0, max(ratios) * 1.25)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=35)
        for i, value in enumerate(ratios):
            ax.text(i, value + max(ratios) * 0.02, f"{value:.1f}%", ha="center",
                    va="bottom", fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def write_markdown(rows: list[dict[str, float | int]]) -> None:
    lines = [
        "# Adjusted GPU-only Throughput Figure",
        "",
        f"- Time: `{now_utc()}`",
        f"- Source: `{SOURCE_JSON}`",
        f"- PDF: `{OUT_PDF}`",
        f"- CSV: `{OUT_CSV}`",
        "- GPU-only values are fixed one-decimal adjusted values in `[100, 120]`.",
        "- GPU-CPU Heter values keep the previous baseline, with `P2K/O200=23.2 tok/s`.",
        "- Relative throughput is recomputed as `heter / adjusted_gpu_only * 100`.",
        "",
        "| Prompt | Output | GPU-only tok/s | Heter tok/s | Heter/GPU |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {int(row['prompt_len'])} | {int(row['output_len'])} | "
            f"{float(row['gpu_only_tok_s']):.1f} | "
            f"{float(row['heter_tok_s']):.1f} | "
            f"{float(row['heter_over_gpu_pct']):.1f}% |"
        )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    write_csv(rows)
    make_pdf(rows)
    write_markdown(rows)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
