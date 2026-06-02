#!/usr/bin/env python3
"""Plot projected GPU-CPU heter throughput with memory optimization."""

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
RESULT_JSON = ROOT / "gpu_vs_heter_throughput_results.json"
OUT_CSV = ROOT / "gpu_cpu_heter_memory_opt_projection.csv"
OUT_PDF = ROOT / "gpu_cpu_heter_memory_opt_projection.pdf"
OUT_MD = ROOT / "gpu_cpu_heter_memory_opt_projection_record.md"

PROMPT_LENS = [2000, 4000, 8000]
OUTPUT_LENS = [200, 400, 600]

# Assumed speedup factors requested for the projected memory-optimized variant.
# The factors stay in [1.32, 1.66] and increase with prompt length.
IMPROVEMENT_FACTORS = {
    (2000, 200): 1.32,
    (2000, 400): 1.35,
    (2000, 600): 1.38,
    (4000, 200): 1.44,
    (4000, 400): 1.47,
    (4000, 600): 1.50,
    (8000, 200): 1.58,
    (8000, 400): 1.62,
    (8000, 600): 1.66,
}

# The original P2K/O200 heter result was later identified as a bad run and
# rechecked separately. Use the corrected median from that recheck.
BASELINE_OVERRIDES = {
    (2000, 200): 23.198,
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
    payload = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    rows = []
    for row in payload["rows"]:
        if row.get("mode") != "heter":
            continue
        prompt_len = int(row["prompt_len_target"])
        output_len = int(row["output_len_target"])
        if prompt_len not in PROMPT_LENS or output_len not in OUTPUT_LENS:
            continue
        baseline = float(row["completion_tokens_per_s"])
        baseline = BASELINE_OVERRIDES.get((prompt_len, output_len), baseline)
        factor = IMPROVEMENT_FACTORS[(prompt_len, output_len)]
        rows.append(
            {
                "prompt_len": prompt_len,
                "output_len": output_len,
                "heter_tok_s": baseline,
                "memory_opt_factor": factor,
                "memory_opt_tok_s": baseline * factor,
            }
        )
    rows.sort(key=lambda r: (int(r["output_len"]), int(r["prompt_len"])))
    return rows


def write_csv(rows: list[dict[str, float | int]]) -> None:
    fields = [
        "prompt_len",
        "output_len",
        "heter_tok_s",
        "memory_opt_factor",
        "memory_opt_tok_s",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_pdf(rows: list[dict[str, float | int]]) -> None:
    data = {
        (int(r["prompt_len"]), int(r["output_len"])): r
        for r in rows
    }
    with PdfPages(OUT_PDF) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharey=True)
        x = list(range(len(PROMPT_LENS)))
        width = 0.35
        baseline_color = "#dc2626"
        opt_color = "#2563eb"
        for ax, output_len in zip(axes, OUTPUT_LENS):
            heter_vals = [
                float(data[(prompt_len, output_len)]["heter_tok_s"])
                for prompt_len in PROMPT_LENS
            ]
            opt_vals = [
                float(data[(prompt_len, output_len)]["memory_opt_tok_s"])
                for prompt_len in PROMPT_LENS
            ]
            factors = [
                float(data[(prompt_len, output_len)]["memory_opt_factor"])
                for prompt_len in PROMPT_LENS
            ]
            ax.bar(
                [i - width / 2 for i in x],
                heter_vals,
                width,
                label="GPU-CPU Heter",
                color=baseline_color,
            )
            ax.bar(
                [i + width / 2 for i in x],
                opt_vals,
                width,
                label="GPU-CPU Heter w/ memory opt.",
                color=opt_color,
            )
            ax.set_title(f"Output length = {output_len}", fontproperties=FONT)
            ax.set_xticks(x, [f"{p // 1000}K" for p in PROMPT_LENS])
            ax.set_xlabel("Prompt length", fontproperties=FONT)
            ax.set_ylim(0, 50)
            ax.grid(axis="y", alpha=0.25)
            for i, value in enumerate(heter_vals):
                ax.text(i - width / 2, value + 0.7, f"{value:.1f}",
                        ha="center", va="bottom", fontsize=8)
            for i, (value, factor) in enumerate(zip(opt_vals, factors)):
                ax.text(i + width / 2, value + 0.7,
                        f"{value:.1f}\n({factor:.2f}x)",
                        ha="center", va="bottom", fontsize=8)
        axes[0].set_ylabel("Throughput (completion tokens/s)", fontproperties=FONT)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            prop=FONT,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=2,
            frameon=False,
        )
        fig.suptitle(
            "Projected Throughput Gain from Memory Optimization",
            fontproperties=FONT,
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.84))
        pdf.savefig(fig)
        plt.close(fig)


def write_markdown(rows: list[dict[str, float | int]]) -> None:
    lines = [
        "# GPU-CPU Heter Memory Optimization Projection",
        "",
        f"- Time: `{now_utc()}`",
        f"- Source baseline: `{RESULT_JSON}`",
        f"- PDF: `{OUT_PDF}`",
        f"- CSV: `{OUT_CSV}`",
        "- Baseline bars are the measured `GPU-CPU Heter` throughput from the earlier experiment.",
        "- `P2K/O200` baseline uses the later anomaly recheck median: `23.198 tok/s`.",
        "- `GPU-CPU Heter w/ memory opt.` bars are projected values computed by multiplying the baseline by the listed factor.",
        "- The factors are intentionally varied in `[1.32, 1.66]`, with smaller gains at 2K, medium gains at 4K, and largest gains at 8K.",
        "",
        "| Prompt | Output | Heter tok/s | Memory opt. factor | Memory opt. tok/s |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {int(row['prompt_len'])} | {int(row['output_len'])} | "
            f"{float(row['heter_tok_s']):.3f} | "
            f"{float(row['memory_opt_factor']):.2f}x | "
            f"{float(row['memory_opt_tok_s']):.3f} |"
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
