#!/usr/bin/env python3
"""Generate a three-level profile PDF for the KV-split P8K/O400 run."""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent
RESULT_JSON = ROOT / "kvsplit_p8k_o400_deep_profile_results.json"
OUT_PDF = ROOT / "kvsplit_p8k_o400_three_level_profile.pdf"
LATIN_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
CJK_FONT_PATH = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")


COLORS = {
    "attention": "#2F6B9A",
    "metadata": "#6B9E4A",
    "scheduler": "#9BC53D",
    "d2h": "#C97D30",
    "h2d": "#B85C5C",
    "cache": "#7B5EA7",
    "wait": "#6C757D",
    "alloc": "#A9A9A9",
    "gap": "#D9A441",
    "inner": "#3E7CB1",
    "wrapper": "#CF8F2E",
    "qk": "#4C78A8",
    "pv": "#F58518",
    "softmax": "#54A24B",
    "reduce": "#B279A2",
    "lookup": "#72B7B2",
    "task": "#E45756",
    "sync": "#9D755D",
    "other": "#BAB0AC",
    "qcopy": "#8CD17D",
    "mask": "#FF9DA6",
}


def setup_fonts() -> FontProperties:
    families: list[str] = []
    for font_path in (LATIN_FONT_PATH, CJK_FONT_PATH):
        if font_path.exists():
            fontManager.addfont(str(font_path))
            families.append(FontProperties(fname=str(font_path)).get_name())
    if not families:
        families = ["DejaVu Sans"]
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = families
    return FontProperties(family=families)


FONT = setup_fonts()
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def load_data() -> dict:
    with RESULT_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def event_total(outer: dict, name: str) -> float:
    return float(outer.get(name, {}).get("total_s", 0.0))


def event_count(outer: dict, name: str) -> int:
    return int(outer.get(name, {}).get("count", 0))


def pct(value: float, total: float) -> float:
    return value / total * 100.0 if total > 0 else 0.0


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def stacked_bar(
    ax: plt.Axes,
    items: list[dict],
    total: float,
    title: str,
    xlabel: str,
    normalize: bool,
) -> None:
    left = 0.0
    for item in items:
        value = item["value"]
        width = pct(value, total) if normalize else value
        ax.barh(
            [0],
            [width],
            left=left,
            color=item["color"],
            edgecolor="white",
            linewidth=0.8,
            height=0.45,
        )
        if width >= (8.0 if normalize else total * 0.08):
            label = f"{item['short']}\n{value:.3f}s"
            if normalize:
                label += f" / {pct(value, total):.1f}%"
            ax.text(
                left + width / 2,
                0,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontproperties=FONT,
            )
        left += width
    ax.set_xlim(0, 100 if normalize else total)
    ax.set_yticks([])
    ax.set_xlabel(xlabel, fontproperties=FONT)
    ax.set_title(title, loc="left", fontsize=12, fontproperties=FONT)
    ax.grid(axis="x", alpha=0.22)
    for spine in ("top", "left", "right"):
        ax.spines[spine].set_visible(False)


def add_legend(ax: plt.Axes, items: list[dict], ncol: int = 2) -> None:
    handles = [
        Patch(facecolor=item["color"], edgecolor="white", label=item["label"])
        for item in items
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.42),
        ncol=ncol,
        frameon=False,
        fontsize=8,
        prop=FONT,
    )


def horizontal_values(
    ax: plt.Axes,
    items: list[dict],
    total: float,
    title: str,
    unit: str,
    color_by_item: bool = True,
) -> None:
    plot_items = sorted(items, key=lambda x: x["value"])
    labels = [item["label"] for item in plot_items]
    values = [item["value"] for item in plot_items]
    colors = [item["color"] for item in plot_items] if color_by_item else "#4C78A8"
    ypos = list(range(len(plot_items)))
    ax.barh(ypos, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8, fontproperties=FONT)
    ax.set_xlabel(unit, fontproperties=FONT)
    ax.set_title(title, loc="left", fontsize=12, fontproperties=FONT)
    ax.grid(axis="x", alpha=0.22)
    max_v = max(values) if values else 1.0
    ax.set_xlim(0, max_v * 1.22)
    for y, item in zip(ypos, plot_items):
        value = item["value"]
        ax.text(
            value + max_v * 0.015,
            y,
            f"{value:.3f}s  ({pct(value, total):.1f}%)",
            va="center",
            fontsize=8,
            fontproperties=FONT,
        )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def write_rows(
    ax: plt.Axes,
    rows: list[tuple[str, str, str, str]],
    title: str,
    y0: float = 0.94,
    line_gap: float = 0.070,
    desc_width: int = 42,
) -> None:
    ax.axis("off")
    ax.text(0.0, 1.0, title, fontsize=12, weight="bold", fontproperties=FONT)
    headers = ("耗时项", "时间", "占比", "具体操作")
    xs = (0.00, 0.27, 0.40, 0.52)
    for x, header in zip(xs, headers):
        ax.text(x, y0, header, fontsize=9, weight="bold", fontproperties=FONT)
    ax.plot([0, 1], [y0 - 0.018, y0 - 0.018], color="#222222", lw=0.8)
    y = y0 - 0.055
    for name, value, share, desc in rows:
        desc_wrapped = wrap(desc, desc_width)
        n_lines = max(1, desc_wrapped.count("\n") + 1)
        ax.text(xs[0], y, wrap(name, 18), fontsize=8, va="top", fontproperties=FONT)
        ax.text(xs[1], y, value, fontsize=8, va="top", fontproperties=FONT)
        ax.text(xs[2], y, share, fontsize=8, va="top", fontproperties=FONT)
        ax.text(xs[3], y, desc_wrapped, fontsize=8, va="top", fontproperties=FONT)
        y -= max(line_gap, n_lines * 0.033 + 0.026)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)


def write_bullets(
    ax: plt.Axes,
    title: str,
    bullets: list[str],
    y0: float = 0.88,
    width: int = 118,
) -> None:
    ax.axis("off")
    ax.text(0.0, 0.98, title, fontsize=14, weight="bold", fontproperties=FONT)
    y = y0
    for bullet in bullets:
        wrapped = wrap(bullet, width)
        lines = wrapped.count("\n") + 1
        ax.text(0.02, y, "- " + wrapped, fontsize=10, va="top", fontproperties=FONT)
        y -= max(0.10, lines * 0.045 + 0.035)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def add_footer(fig: plt.Figure, page: str) -> None:
    fig.text(
        0.015,
        0.012,
        f"Source: {RESULT_JSON.name} | {page}",
        fontsize=7,
        color="#555555",
        fontproperties=FONT,
    )


def build_components(data: dict) -> tuple[list[dict], list[dict], list[dict], dict]:
    server = data["server_profile"]
    outer = server["outer_summary"]
    inner = server["inner"]

    decode_total = event_total(outer, "decode_cpu_path_layer_total")
    cpp_total = event_total(outer, "cpu_attention_cpp_total")
    meta_total = event_total(outer, "decode_metadata_d2h")
    scheduler = event_total(outer, "decode_cpu_scheduler_metadata")
    meta_other = max(0.0, meta_total - scheduler)
    known = (
        cpp_total
        + meta_other
        + scheduler
        + event_total(outer, "decode_kv_d2h")
        + event_total(outer, "decode_kv_cache_write_total")
        + event_total(outer, "decode_output_h2d")
        + event_total(outer, "decode_wait_prefill_d2h")
        + event_total(outer, "decode_query_d2h")
        + event_total(outer, "decode_output_cpu_alloc")
    )
    gap = max(0.0, decode_total - known)

    level1 = [
        {
            "key": "cpp_total",
            "short": "C++ op",
            "label": "C++ CPU attention op total",
            "value": cpp_total,
            "color": COLORS["attention"],
            "desc": "调用 vLLM 原生 CPU paged attention C++ op 的外层累计时间；包含 C++ inner、op wrapper、profile 边界等。",
        },
        {
            "key": "metadata_other",
            "short": "metadata",
            "label": "Metadata copy/package excl. scheduler",
            "value": meta_other,
            "color": COLORS["metadata"],
            "desc": "query_start_loc、seq_lens、block_table 等 metadata 的 GPU->CPU 小 tensor 拷贝、dtype 转换、CPU tensor 分配和 Python 打包；不含 scheduler metadata 子项。",
        },
        {
            "key": "kv_d2h",
            "short": "K/V D2H",
            "label": "Decode K/V/slot D2H",
            "value": event_total(outer, "decode_kv_d2h"),
            "color": COLORS["d2h"],
            "desc": "decode 新生成 K、V 以及 slot_mapping 等小 tensor 从 GPU 同步搬到 CPU，供 CPU KV cache 写入。",
        },
        {
            "key": "h2d",
            "short": "out H2D",
            "label": "Output H2D",
            "value": event_total(outer, "decode_output_h2d"),
            "color": COLORS["h2d"],
            "desc": "CPU attention 输出从 CPU pinned memory 拷回 GPU output tensor，供后续 GPU 算子继续执行。",
        },
        {
            "key": "cache_write",
            "short": "KV write",
            "label": "CPU KV cache write",
            "value": event_total(outer, "decode_kv_cache_write_total"),
            "color": COLORS["cache"],
            "desc": "把 decode 阶段新 token 的 K/V reshape 后写入 CPU paged KV cache，对应 CPU cache 更新路径。",
        },
        {
            "key": "wait_prefill",
            "short": "wait",
            "label": "Wait prefill D2H stream",
            "value": event_total(outer, "decode_wait_prefill_d2h"),
            "color": COLORS["wait"],
            "desc": "decode 进入 CPU attention 前等待该层 prefill KV cache 的异步 D2H stream 完成。",
        },
        {
            "key": "scheduler",
            "short": "sched",
            "label": "CPU scheduler metadata",
            "value": scheduler,
            "color": COLORS["scheduler"],
            "desc": "调用 C++ get_scheduler_metadata 生成 CPU paged attention 调度元数据，包括 work item、KV split/reduction 信息等。",
        },
        {
            "key": "query_d2h",
            "short": "Q D2H",
            "label": "Query D2H",
            "value": event_total(outer, "decode_query_d2h"),
            "color": COLORS["d2h"],
            "desc": "当前 decode query 从 GPU tensor 同步搬到 CPU tensor，作为 CPU attention 的 Q 输入。",
        },
        {
            "key": "alloc",
            "short": "alloc",
            "label": "CPU output alloc",
            "value": event_total(outer, "decode_output_cpu_alloc"),
            "color": COLORS["alloc"],
            "desc": "为 CPU attention 输出分配 CPU pinned tensor。",
        },
        {
            "key": "gap",
            "short": "gap",
            "label": "Unattributed outer gap",
            "value": gap,
            "color": COLORS["gap"],
            "desc": "decode CPU path 外层 timer 覆盖但尚未单独插桩的时间，可能包含 profile JSONL 写入、Python glue、torch/C++ 边界和小同步副作用。",
        },
    ]

    inner_elapsed = float(inner["elapsed_total_s"])
    level2 = [
        {
            "key": "inner",
            "short": "inner",
            "label": "C++ inner elapsed",
            "value": inner_elapsed,
            "color": COLORS["inner"],
            "desc": "C++ CPU attention 内部 wall-time 计时，覆盖 worker 并行区内主要 attention 执行路径。",
        },
        {
            "key": "wrapper",
            "short": "wrapper",
            "label": "C++ op wrapper/profile/boundary gap",
            "value": max(0.0, cpp_total - inner_elapsed),
            "color": COLORS["wrapper"],
            "desc": "cpu_attention_cpp_total 中未落入 C++ inner elapsed 的部分，包含 torch custom op dispatch、C++ wrapper 参数处理、profile 写入和返回边界等。",
        },
    ]

    comp = inner["components_s"]
    level3_defs = [
        ("pv_tile_gemm", "PV tile GEMM + V stream", COLORS["pv"], "P×V tile 计算及 V cache 流式读取；在当前 profile 中是最大 thread-time 热点。"),
        ("qk_tile_gemm", "QK tile GEMM + K stream", COLORS["qk"], "Q×K tile 计算及 K cache 流式读取；负责生成 attention score。"),
        ("sync_wait", "Sync/wait", COLORS["sync"], "KV split 并行区里的 split barrier 和 reduce flag 等同步等待。"),
        ("loop_other", "QK/PV loop other", COLORS["other"], "QK/PV 主循环中未归到 tile GEMM、block lookup 的其他循环控制和边界处理。"),
        ("softmax", "Softmax", COLORS["softmax"], "对 attention score 做 online softmax，包括 max/sum 维护和概率归一化。"),
        ("output_reduce", "Output/reduce", COLORS["reduce"], "partial output 写入、split reduction 和 final output 写回；KV split 下用于合并多个 split 的结果。"),
        ("block_lookup", "Paged block lookup", COLORS["lookup"], "根据 block_table 查找 paged KV cache block 的地址/索引，包含 QK 和 PV 两段 lookup。"),
        ("task_acquire", "Task acquire/atomic", COLORS["task"], "OpenMP worker 通过 atomic/fetch-add 等方式领取 work item 的开销。"),
        ("q_copy", "Q copy/scale", COLORS["qcopy"], "将 Q 拷贝/缩放到 CPU kernel 使用的局部格式。"),
        ("mask", "Mask", COLORS["mask"], "causal/sliding-window mask 等 score 屏蔽操作。"),
    ]
    level3 = [
        {
            "key": key,
            "short": label.split()[0],
            "label": label,
            "value": float(comp.get(key, 0.0)),
            "color": color,
            "desc": desc,
        }
        for key, label, color, desc in level3_defs
    ]
    derived = {
        "decode_total": decode_total,
        "cpp_total": cpp_total,
        "inner_elapsed": inner_elapsed,
        "inner_thread_total": float(inner["component_total_s"]),
        "worker_equiv": float(inner["component_total_s"]) / inner_elapsed
        if inner_elapsed > 0
        else math.nan,
        "request_wall": float(server["wall_s"]),
        "decode_count": event_count(outer, "decode_cpu_path_layer_total"),
        "prompt_len": data["prompt_len"],
        "output_len": data["output_len"],
        "omp_threads": data["omp_threads"],
        "timestamp_utc": data["timestamp_utc"],
    }
    return level1, level2, level3, derived


def rows_for(items: list[dict], total: float, avg_denominator: int | None = None) -> list:
    rows = []
    for item in sorted(items, key=lambda x: x["value"], reverse=True):
        value = item["value"]
        time_text = f"{value:.3f} s"
        if avg_denominator:
            time_text += f"\n{value / avg_denominator * 1000.0:.3f} ms/层-token"
        rows.append(
            (
                item["label"],
                time_text,
                f"{pct(value, total):.2f}%",
                item["desc"],
            )
        )
    return rows


def make_pdf() -> None:
    data = load_data()
    level1, level2, level3, d = build_components(data)

    with PdfPages(OUT_PDF) as pdf:
        fig = plt.figure(figsize=(16.5, 11.0), constrained_layout=True)
        gs = fig.add_gridspec(5, 1, height_ratios=[0.7, 1.0, 1.0, 1.0, 1.25])
        ax_title = fig.add_subplot(gs[0])
        ax_title.axis("off")
        title = "KV split P8K/O400 CPU Decode Attention 三层 Profile 总图"
        subtitle = (
            f"workload: prompt={d['prompt_len']}, output={d['output_len']}, "
            f"batch=1, OMP_NUM_THREADS={d['omp_threads']} | "
            f"request wall={d['request_wall']:.3f}s | source time={d['timestamp_utc']}"
        )
        ax_title.text(0.0, 0.78, title, fontsize=18, weight="bold", fontproperties=FONT)
        ax_title.text(0.0, 0.36, subtitle, fontsize=10, color="#333333", fontproperties=FONT)
        ax_title.text(
            0.0,
            0.02,
            "读图方式：第一、二层是 wall-time；第三层是 OpenMP worker 累计 thread-time，只用于定位 C++ inner 内部热点，不能直接和 wall-time 相加。",
            fontsize=10,
            color="#7A3E00",
            fontproperties=FONT,
        )

        ax1 = fig.add_subplot(gs[1])
        stacked_bar(
            ax1,
            level1,
            d["decode_total"],
            f"Layer 1: decode_cpu_path_layer_total = {d['decode_total']:.3f}s",
            "share of decode_cpu_path_layer_total (%)",
            normalize=True,
        )
        add_legend(ax1, level1, ncol=5)

        ax2 = fig.add_subplot(gs[2])
        stacked_bar(
            ax2,
            level2,
            d["cpp_total"],
            f"Layer 2: cpu_attention_cpp_total = {d['cpp_total']:.3f}s",
            "share of cpu_attention_cpp_total (%)",
            normalize=True,
        )
        add_legend(ax2, level2, ncol=2)

        ax3 = fig.add_subplot(gs[3])
        stacked_bar(
            ax3,
            level3,
            d["inner_thread_total"],
            (
                "Layer 3: C++ inner components = "
                f"{d['inner_thread_total']:.3f} thread-s "
                f"(inner elapsed {d['inner_elapsed']:.3f}s, "
                f"{d['worker_equiv']:.1f} worker-equivalent)"
            ),
            "share of C++ inner thread-time (%)",
            normalize=True,
        )
        add_legend(ax3, level3, ncol=5)

        ax_note = fig.add_subplot(gs[4])
        ax_note.axis("off")
        formulas = [
            f"第一层非重叠求和：C++ op + metadata excluding scheduler + scheduler + K/V D2H + CPU KV write + output H2D + wait + Q D2H + alloc + gap = {d['decode_total']:.3f}s。",
            f"第二层关系：cpu_attention_cpp_total {d['cpp_total']:.3f}s = C++ inner elapsed {d['inner_elapsed']:.3f}s + wrapper/profile/boundary gap {level2[1]['value']:.3f}s。",
            f"第三层热点：PV tile {next(i for i in level3 if i['key']=='pv_tile_gemm')['value']:.3f} thread-s，QK tile {next(i for i in level3 if i['key']=='qk_tile_gemm')['value']:.3f} thread-s，二者合计占 C++ inner thread-time 的 {pct(next(i for i in level3 if i['key']=='pv_tile_gemm')['value'] + next(i for i in level3 if i['key']=='qk_tile_gemm')['value'], d['inner_thread_total']):.2f}%。",
        ]
        y = 0.86
        for line in formulas:
            ax_note.text(0.02, y, "• " + wrap(line, 130), fontsize=10, fontproperties=FONT)
            y -= 0.24
        add_footer(fig, "page 1/6")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(16.5, 11.0), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 0.92])
        ax = fig.add_subplot(gs[0])
        horizontal_values(
            ax,
            level1,
            d["decode_total"],
            "Layer 1 detail: decode_cpu_path_layer_total 内部非重叠耗时",
            "wall-time seconds",
        )
        ax_note = fig.add_subplot(gs[1])
        layer1_bullets = [
            f"decode_cpu_path_layer_total 是每个 decode token、每一层走 CPU attention 路径的外层总 wall-time，累计 {d['decode_total']:.3f}s，事件数 {d['decode_count']}。",
            "本页使用非重叠拆分：decode_metadata_d2h 里的 scheduler 子项被单独拿出来，因此 metadata copy/package excl. scheduler 不会和 CPU scheduler metadata 重复计数。",
            f"最大单项是 C++ CPU attention op total，为 {level1[0]['value']:.3f}s，占第一层总时间 {pct(level1[0]['value'], d['decode_total']):.2f}%。",
            f"未归因外层 gap 为 {next(i for i in level1 if i['key']=='gap')['value']:.3f}s，占 {pct(next(i for i in level1 if i['key']=='gap')['value'], d['decode_total']):.2f}%；它不是某个已知算子，而是当前外层 timer 覆盖但还没有单独插桩的边界开销。",
            "下一页给出 Layer 1 每个耗时项的具体含义。",
        ]
        write_bullets(ax_note, "Layer 1 读图说明", layer1_bullets, width=54)
        add_footer(fig, "page 2/6")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(16.5, 11.0), constrained_layout=True)
        ax_table = fig.add_subplot(111)
        write_rows(
            ax_table,
            rows_for(level1, d["decode_total"], d["decode_count"]),
            "Layer 1 各耗时项含义",
            desc_width=72,
            line_gap=0.075,
        )
        add_footer(fig, "page 3/6")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(16.5, 11.0), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.4], height_ratios=[1.0, 0.75])
        ax = fig.add_subplot(gs[0, 0])
        horizontal_values(
            ax,
            level2,
            d["cpp_total"],
            "Layer 2 detail: cpu_attention_cpp_total",
            "wall-time seconds",
        )
        ax_table = fig.add_subplot(gs[0, 1])
        write_rows(
            ax_table,
            rows_for(level2, d["cpp_total"], None),
            "Layer 2 各耗时项含义",
            desc_width=52,
            line_gap=0.12,
        )
        ax_text = fig.add_subplot(gs[1, :])
        ax_text.axis("off")
        text = (
            "解释：cpu_attention_cpp_total 是 Python 侧调用 C++ CPU attention op 的外层计时；"
            "C++ inner elapsed 是 C++ op 内部额外记录的 kernel 主体 elapsed。两者差值不是计算核心本身，"
            "而是 wrapper、参数处理、custom op dispatch、profile 输出和边界开销。"
            "因此优化 QK/PV tile 只能覆盖 C++ inner 的一部分，无法自动消除第二层 wrapper gap。"
        )
        ax_text.text(
            0.02,
            0.70,
            wrap(text, 105),
            fontsize=10,
            va="top",
            fontproperties=FONT,
        )
        add_footer(fig, "page 4/6")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(16.5, 11.0), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.20, 0.90])
        ax = fig.add_subplot(gs[0])
        horizontal_values(
            ax,
            level3,
            d["inner_thread_total"],
            "Layer 3 detail: C++ inner 内部 thread-time 热点",
            "OpenMP worker thread-time seconds",
        )
        ax_note = fig.add_subplot(gs[1])
        layer3_bullets = [
            f"C++ inner elapsed 是 {d['inner_elapsed']:.3f}s；本页条形图使用的是 OpenMP worker 累计 thread-time，总和 {d['inner_thread_total']:.3f} thread-s。",
            f"thread-time / elapsed 约为 {d['worker_equiv']:.1f} worker-equivalent，表示并行 worker 消耗的累计时间，不应和 Layer 1/2 wall-time 直接相加。",
            f"PV tile GEMM + V stream 和 QK tile GEMM + K stream 合计 {next(i for i in level3 if i['key']=='pv_tile_gemm')['value'] + next(i for i in level3 if i['key']=='qk_tile_gemm')['value']:.3f} thread-s，占 C++ inner thread-time 的 {pct(next(i for i in level3 if i['key']=='pv_tile_gemm')['value'] + next(i for i in level3 if i['key']=='qk_tile_gemm')['value'], d['inner_thread_total']):.2f}%，是 C++ inner 内部最主要热点。",
            "下一页给出 C++ inner 每个耗时项的具体含义。",
        ]
        write_bullets(ax_note, "Layer 3 读图说明", layer3_bullets, width=54)
        add_footer(fig, "page 5/6")
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure(figsize=(16.5, 11.0), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 0.65])
        ax_table = fig.add_subplot(gs[0])
        write_rows(
            ax_table,
            rows_for(level3, d["inner_thread_total"], None),
            "Layer 3 各耗时项含义",
            desc_width=58,
            line_gap=0.074,
        )
        ax_summary = fig.add_subplot(gs[1])
        summary_bullets = [
            "第一层看真实 decode CPU path 成本；第二层看 C++ op 外层调用成本；第三层只看 C++ inner 的线程级热点。",
            "当前 profile 的主要结论是：外层 C++ op 之外还有明显 metadata、D2H/H2D、CPU KV write 和未归因 gap；C++ inner 内部则以 QK/PV tile 路径为主。",
            "Paged block lookup 与 task acquire 占比很小，不是当前 profile 的主瓶颈。",
            "该图使用已有 profile 结果生成，没有重新运行服务。",
        ]
        write_bullets(ax_summary, "总体结论", summary_bullets, width=42)
        add_footer(fig, "page 6/6")
        pdf.savefig(fig)
        plt.close(fig)


if __name__ == "__main__":
    make_pdf()
    print(OUT_PDF)
