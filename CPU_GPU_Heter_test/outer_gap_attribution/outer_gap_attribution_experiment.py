#!/usr/bin/env python3
"""Run profile-mode A/B experiments to attribute decode outer gap."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, fontManager
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "CPU_GPU_Heter_test" / "outer_gap_attribution"
PYTHON = Path("/home/zb/tool/miniconda3/envs/vLLM_GPU/bin/python")
VLLM = Path("/home/zb/tool/miniconda3/envs/vLLM_GPU/bin/vllm")
MODEL = Path("/home/zb/data/model_data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")
CPU_ATTN_LIB = ROOT / "vllm" / "libs" / "libvllm_heter_cpu_attn.so"
PORT = 8036
PROMPT_LEN = 8000
OUTPUT_LEN = 400
OMP_THREADS = 120
AFFINITY = (
    "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,"
    "20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,"
    "40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,"
    "60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,"
    "80,81,82,83,84,85,86,87,88,89,90,91,96,97,98,99,100,101,102,"
    "103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,"
    "118,119,120,121,122,123"
)

RESULT_JSON = OUT_DIR / "outer_gap_attribution_results.json"
RESULT_PDF = OUT_DIR / "outer_gap_attribution.pdf"
RESULT_MD = OUT_DIR / "outer_gap_attribution_记录.md"

LATIN_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
CJK_FONT = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")

OUTER_EVENTS = [
    "cpu_attention_cpp_total",
    "decode_metadata_d2h",
    "decode_cpu_scheduler_metadata",
    "decode_kv_d2h",
    "decode_kv_cache_write_total",
    "decode_output_h2d",
    "decode_wait_prefill_d2h",
    "decode_query_d2h",
    "decode_output_cpu_alloc",
]

NON_OVERLAP_BASE = [
    "cpu_attention_cpp_total",
    "metadata_excl_scheduler",
    "decode_cpu_scheduler_metadata",
    "decode_kv_d2h",
    "decode_kv_cache_write_total",
    "decode_output_h2d",
    "decode_wait_prefill_d2h",
    "decode_query_d2h",
    "decode_output_cpu_alloc",
]

FINE_OUTER_GAP_EVENTS = [
    "profile_record_self_time",
    "decode_gap_after_wait_before_kv",
    "decode_gap_after_kv_before_query",
    "decode_gap_after_query_before_alloc",
    "decode_gap_after_alloc_before_metadata",
    "decode_gap_after_metadata_before_cpp",
    "decode_gap_after_cpp_before_h2d",
    "decode_gap_after_h2d_before_total",
]

FINE_METADATA_EVENTS = [
    "metadata_query_start_loc_cpu_to_i32",
    "metadata_seq_lens_cpu_to_i32",
    "metadata_block_table_cpu_to_i32",
    "metadata_base_dict_package",
    "metadata_alibi_package",
    "metadata_ops_lookup",
    "metadata_get_scheduler_attr",
    "metadata_scheduler_arg_prepare",
]

FINE_CPP_WRAPPER_EVENTS = [
    "cpu_attention_marker_file",
    "cpu_attention_kv_cache_unbind",
    "cpu_attention_metadata_get",
    "cpu_attention_ops_lookup",
    "cpu_attention_arg_prepare",
]

GROUPS = [
    {
        "name": "A_current_jsonl",
        "mode": "jsonl",
        "profile": True,
        "inner": False,
        "desc": "旧行为：每个 profile event 逐条写 JSONL。",
    },
    {
        "name": "B_memory_aggregate",
        "mode": "memory",
        "profile": True,
        "inner": False,
        "desc": "仍记录所有外层事件，但只在内存聚合并周期性写 summary。",
    },
    {
        "name": "C_outer_only",
        "mode": "outer_only",
        "profile": True,
        "inner": False,
        "desc": "只记录 decode_cpu_path_layer_total，用于估计子项计时/记录扰动。",
    },
    {
        "name": "D_profile_off",
        "mode": "off",
        "profile": False,
        "inner": False,
        "desc": "关闭 heter profile，只测 request wall。",
    },
    {
        "name": "E_fine_memory",
        "mode": "fine",
        "profile": True,
        "inner": False,
        "desc": "内存聚合，并增加 outer gap/metadata/C++ wrapper 细粒度计时。",
    },
]


def setup_fonts() -> FontProperties:
    names: list[str] = []
    for path in (LATIN_FONT, CJK_FONT):
        if path.exists():
            fontManager.addfont(str(path))
            names.append(FontProperties(fname=str(path)).get_name())
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


def make_prompt(token_count: int) -> str:
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    seed = (
        "This deterministic profiling prompt contains repeated factual clauses. "
        "The model should continue with numbered observations and avoid summaries. "
    )
    text = seed
    while len(tokenizer.encode(text, add_special_tokens=False)) < token_count + 128:
        text += seed
    token_ids = tokenizer.encode(text, add_special_tokens=False)[:token_count]
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def http_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_ready(port: int, timeout_s: float, log_path: Path) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    deadline = time.time() + timeout_s
    last_error = None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.time() < deadline:
        try:
            with opener.open(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2)
    tail = ""
    if log_path.exists():
        tail = "\n".join(log_path.read_text(errors="ignore").splitlines()[-80:])
    raise RuntimeError(f"vLLM server not ready: {last_error}\n{tail}")


def terminate_process(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=45)
        return
    except Exception:
        pass
    try:
        proc.terminate()
        proc.wait(timeout=20)
        return
    except Exception:
        pass
    try:
        proc.kill()
        proc.wait(timeout=20)
    except Exception:
        pass


def read_jsonl_summary(path: Path) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "total_s": 0.0, "max_ms": 0.0}
    )
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            event = json.loads(line)
            name = event.get("event")
            elapsed = float(event.get("elapsed_s", 0.0))
            item = totals[name]
            item["count"] += 1
            item["total_s"] += elapsed
            item["max_ms"] = max(item["max_ms"], elapsed * 1000.0)
    for item in totals.values():
        count = int(item["count"])
        item["avg_ms"] = item["total_s"] / count * 1000.0 if count else 0.0
    return dict(totals)


def read_memory_summary(summary_prefix: Path) -> dict[str, dict[str, float]]:
    events: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "total_s": 0.0, "max_ms": 0.0}
    )
    for path in sorted(summary_prefix.parent.glob(f"{summary_prefix.stem}.pid*.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        for name, value in payload.get("events", {}).items():
            item = events[name]
            item["count"] += int(value.get("count", 0))
            item["total_s"] += float(value.get("total_s", 0.0))
            item["max_ms"] = max(item["max_ms"], float(value.get("max_ms", 0.0)))
    for item in events.values():
        count = int(item["count"])
        item["avg_ms"] = item["total_s"] / count * 1000.0 if count else 0.0
    return dict(events)


def total(events: dict[str, dict[str, float]], name: str) -> float:
    return float(events.get(name, {}).get("total_s", 0.0))


def compute_outer_gap(events: dict[str, dict[str, float]]) -> dict[str, float]:
    decode_total = total(events, "decode_cpu_path_layer_total")
    metadata_total = total(events, "decode_metadata_d2h")
    scheduler = total(events, "decode_cpu_scheduler_metadata")
    metadata_excl_scheduler = max(0.0, metadata_total - scheduler)
    non_overlap = {
        "cpu_attention_cpp_total": total(events, "cpu_attention_cpp_total"),
        "metadata_excl_scheduler": metadata_excl_scheduler,
        "decode_cpu_scheduler_metadata": scheduler,
        "decode_kv_d2h": total(events, "decode_kv_d2h"),
        "decode_kv_cache_write_total": total(events, "decode_kv_cache_write_total"),
        "decode_output_h2d": total(events, "decode_output_h2d"),
        "decode_wait_prefill_d2h": total(events, "decode_wait_prefill_d2h"),
        "decode_query_d2h": total(events, "decode_query_d2h"),
        "decode_output_cpu_alloc": total(events, "decode_output_cpu_alloc"),
    }
    known_sum = sum(non_overlap.values())
    outer_gap = decode_total - known_sum if decode_total > 0 else 0.0
    return {
        "decode_cpu_path_layer_total": decode_total,
        "known_sum": known_sum,
        "outer_gap": outer_gap,
        "outer_gap_share": outer_gap / decode_total if decode_total > 0 else 0.0,
        **non_overlap,
    }


def run_group(group: dict[str, Any]) -> dict[str, Any]:
    name = group["name"]
    print(f"[{now_utc()}] run {name}", flush=True)
    log_path = OUT_DIR / f"{name}_server.log"
    jsonl_path = OUT_DIR / f"{name}_outer_events.jsonl"
    summary_prefix = OUT_DIR / f"{name}_summary.json"
    inner_path = OUT_DIR / f"{name}_inner_events.jsonl"

    for path in [log_path, jsonl_path, inner_path]:
        if path.exists():
            path.unlink()
    for path in OUT_DIR.glob(f"{summary_prefix.stem}.pid*.json"):
        path.unlink()

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "OMP_NUM_THREADS": str(OMP_THREADS),
            "VLLM_HETER_CPU_ATTN_LIB": str(CPU_ATTN_LIB),
            "no_proxy": "localhost,127.0.0.1",
            "NO_PROXY": "localhost,127.0.0.1",
        }
    )
    if group["profile"]:
        env["VLLM_HETER_PROFILE"] = "1"
        env["VLLM_HETER_PROFILE_MODE"] = group["mode"]
        env["VLLM_HETER_PROFILE_PATH"] = str(jsonl_path)
        env["VLLM_HETER_PROFILE_SUMMARY_PATH"] = str(summary_prefix)
        env["VLLM_HETER_PROFILE_FLUSH_INTERVAL_S"] = "2.0"
    else:
        env["VLLM_HETER_PROFILE"] = "0"
        env.pop("VLLM_HETER_PROFILE_MODE", None)
    if group["inner"]:
        env["VLLM_HETER_CPU_ATTN_INNER_PROFILE_PATH"] = str(inner_path)

    cmd = [
        "taskset",
        "-c",
        AFFINITY,
        str(VLLM),
        "serve",
        str(MODEL),
        "--port",
        str(PORT),
        "--gpu-memory-utilization",
        "0.25",
        "--max-model-len",
        "10000",
        "--trust-remote-code",
    ]
    with log_path.open("wb") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
    try:
        wait_ready(PORT, 600.0, log_path)
        prompt = make_prompt(PROMPT_LEN)
        payload = {
            "model": str(MODEL),
            "prompt": prompt,
            "max_tokens": OUTPUT_LEN,
            "temperature": 0.0,
            "ignore_eos": True,
        }
        start = time.perf_counter()
        response = http_json(
            f"http://127.0.0.1:{PORT}/v1/completions",
            payload,
            timeout_s=900.0,
        )
        wall_s = time.perf_counter() - start
        usage = response.get("usage", {})
        finish_reason = response.get("choices", [{}])[0].get("finish_reason")
    finally:
        terminate_process(proc)

    if group["mode"] == "jsonl":
        events = read_jsonl_summary(jsonl_path)
    elif group["profile"]:
        events = read_memory_summary(summary_prefix)
    else:
        events = {}
    gap = compute_outer_gap(events)
    result = {
        "name": name,
        "mode": group["mode"],
        "description": group["desc"],
        "request_wall_s": wall_s,
        "usage": usage,
        "finish_reason": finish_reason,
        "server_log": str(log_path),
        "outer_events_jsonl": str(jsonl_path) if jsonl_path.exists() else None,
        "summary_files": [
            str(p) for p in sorted(OUT_DIR.glob(f"{summary_prefix.stem}.pid*.json"))
        ],
        "events": events,
        "gap": gap,
    }
    print(
        f"[{now_utc()}] done {name}: wall={wall_s:.3f}s "
        f"decode={gap['decode_cpu_path_layer_total']:.3f}s "
        f"gap={gap['outer_gap']:.3f}s",
        flush=True,
    )
    return result


def pct(value: float, total: float) -> float:
    return value / total * 100.0 if total else 0.0


def short_event_name(name: str) -> str:
    replacements = [
        ("decode_gap_after_", "gap after "),
        ("decode_", ""),
        ("metadata_", "meta "),
        ("cpu_attention_", "attn "),
        ("profile_record_self_time", "profile record self"),
        ("_before_", " before "),
        ("_to_", " to "),
        ("_total", " total"),
    ]
    shortened = name
    for old, new in replacements:
        shortened = shortened.replace(old, new)
    return shortened.replace("_", " ")


def make_pdf(results: dict[str, Any]) -> None:
    groups = results["groups"]
    names = [g["name"].replace("_", "\n") for g in groups]
    request = [g["request_wall_s"] for g in groups]
    decode = [g["gap"].get("decode_cpu_path_layer_total", 0.0) for g in groups]
    gap = [g["gap"].get("outer_gap", 0.0) for g in groups]

    with PdfPages(RESULT_PDF) as pdf:
        fig, axes = plt.subplots(3, 1, figsize=(13.5, 12.0), constrained_layout=True)
        fig.suptitle("Outer Gap Attribution: profile mode A/B", fontsize=16, fontproperties=FONT)
        x = range(len(groups))
        axes[0].bar(x, request, color="#4C78A8")
        axes[0].set_title("Request wall time", loc="left", fontproperties=FONT)
        axes[0].set_ylabel("seconds", fontproperties=FONT)
        axes[0].set_xticks(x, names, fontproperties=FONT)
        axes[0].grid(axis="y", alpha=0.25)
        axes[1].bar(x, decode, color="#F58518")
        axes[1].set_title("decode_cpu_path_layer_total", loc="left", fontproperties=FONT)
        axes[1].set_ylabel("seconds", fontproperties=FONT)
        axes[1].set_xticks(x, names, fontproperties=FONT)
        axes[1].grid(axis="y", alpha=0.25)
        axes[2].bar(x, gap, color="#D9A441")
        axes[2].set_title("Outer gap", loc="left", fontproperties=FONT)
        axes[2].set_ylabel("seconds", fontproperties=FONT)
        axes[2].set_xticks(x, names, fontproperties=FONT)
        axes[2].grid(axis="y", alpha=0.25)
        for ax, values in zip(axes, [request, decode, gap]):
            max_v = max(values) if values else 1.0
            for i, value in enumerate(values):
                ax.text(i, value + max_v * 0.02, f"{value:.3f}", ha="center", fontsize=8, fontproperties=FONT)
        pdf.savefig(fig)
        plt.close(fig)

        group_b = next((g for g in groups if g["name"] == "B_memory_aggregate"), None)
        group_e = next((g for g in groups if g["name"] == "E_fine_memory"), None)
        fig, axes = plt.subplots(1, 2, figsize=(16.0, 8.5), constrained_layout=True)
        if group_b:
            pieces = group_b["gap"]
            labels = [
                "C++ op",
                "metadata excl sched",
                "scheduler",
                "KV D2H",
                "KV write",
                "output H2D",
                "wait prefill",
                "Q D2H",
                "alloc",
                "outer gap",
            ]
            values = [
                pieces.get("cpu_attention_cpp_total", 0.0),
                pieces.get("metadata_excl_scheduler", 0.0),
                pieces.get("decode_cpu_scheduler_metadata", 0.0),
                pieces.get("decode_kv_d2h", 0.0),
                pieces.get("decode_kv_cache_write_total", 0.0),
                pieces.get("decode_output_h2d", 0.0),
                pieces.get("decode_wait_prefill_d2h", 0.0),
                pieces.get("decode_query_d2h", 0.0),
                pieces.get("decode_output_cpu_alloc", 0.0),
                pieces.get("outer_gap", 0.0),
            ]
            order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
            labels = [labels[i] for i in order]
            values = [values[i] for i in order]
            total_value = sum(values)
            colors = [
                "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
                "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC", "#D9A441",
            ]
            y = range(len(values))
            axes[0].barh(y, values, color=colors[: len(values)])
            axes[0].set_yticks(y, labels, fontproperties=FONT)
            axes[0].invert_yaxis()
            axes[0].set_xlabel("seconds", fontproperties=FONT)
            axes[0].set_title(
                "B memory aggregate: Layer 1 non-overlap breakdown",
                fontproperties=FONT,
            )
            axes[0].grid(axis="x", alpha=0.25)
            max_value = max(values) if values else 1.0
            for idx, value in enumerate(values):
                axes[0].text(
                    value + max_value * 0.02,
                    idx,
                    f"{value:.3f}s ({pct(value, total_value):.1f}%)",
                    va="center",
                    fontsize=8,
                    fontproperties=FONT,
                )
        if group_e:
            events = group_e["events"]
            fine_items = []
            for name in FINE_OUTER_GAP_EVENTS + FINE_METADATA_EVENTS + FINE_CPP_WRAPPER_EVENTS:
                value = total(events, name)
                if value > 0:
                    fine_items.append((name, value))
            fine_items.sort(key=lambda item: item[1], reverse=True)
            top_items = fine_items[:12]
            fine_labels = [short_event_name(name) for name, _ in top_items]
            fine_values = [value for _, value in top_items]
            order = list(reversed(range(len(fine_values))))
            fine_labels = [fine_labels[i] for i in order]
            fine_values = [fine_values[i] for i in order]
            axes[1].barh(fine_labels, fine_values, color="#54A24B")
            axes[1].set_title(
                "E fine memory: top fine timers (perturbed run)",
                fontproperties=FONT,
            )
            axes[1].set_xlabel("seconds", fontproperties=FONT)
            axes[1].grid(axis="x", alpha=0.25)
            max_value = max(fine_values) if fine_values else 1.0
            for idx, value in enumerate(fine_values):
                axes[1].text(
                    value + max_value * 0.02,
                    idx,
                    f"{value:.3f}s",
                    va="center",
                    fontsize=8,
                    fontproperties=FONT,
                )
            axes[1].text(
                0.98,
                0.03,
                "E fine mode heavily perturbs runtime; use it only as a diagnostic probe.",
                transform=axes[1].transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                fontproperties=FONT,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 2.0},
            )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(16.0, 9.0), constrained_layout=True)
        ax.axis("off")
        lines = [
            f"Generated: {results['timestamp_utc']}",
            f"Workload: prompt={PROMPT_LEN}, output={OUTPUT_LEN}, batch=1, OMP_NUM_THREADS={OMP_THREADS}",
            "Key interpretation:",
        ]
        group_a = next((g for g in groups if g["name"] == "A_current_jsonl"), None)
        group_b = next((g for g in groups if g["name"] == "B_memory_aggregate"), None)
        group_d = next((g for g in groups if g["name"] == "D_profile_off"), None)
        group_e = next((g for g in groups if g["name"] == "E_fine_memory"), None)
        if group_a and group_b:
            lines.append(
                f"- JSONL -> memory aggregate outer gap delta: "
                f"{group_a['gap']['outer_gap'] - group_b['gap']['outer_gap']:.3f}s."
            )
        if group_a and group_d:
            lines.append(
                f"- Profile-on JSONL request wall overhead vs profile-off: "
                f"{group_a['request_wall_s'] - group_d['request_wall_s']:.3f}s."
            )
        if group_e:
            fine_sum = sum(total(group_e["events"], n) for n in FINE_OUTER_GAP_EVENTS)
            lines.append(f"- E fine outer-gap explicit fine timers sum: {fine_sum:.3f}s.")
            lines.append("- E fine mode is intentionally diagnostic and heavily perturbs runtime.")
        y = 0.95
        for line in lines:
            ax.text(0.02, y, line, fontsize=12, va="top", fontproperties=FONT)
            y -= 0.08
        pdf.savefig(fig)
        plt.close(fig)


def write_md(results: dict[str, Any]) -> None:
    lines = [
        "# Outer Gap Attribution 实验记录",
        "",
        f"- 时间：`{results['timestamp_utc']}`",
        f"- workload：`prompt={PROMPT_LEN}`，`output={OUTPUT_LEN}`，`batch=1`",
        f"- OpenMP：`OMP_NUM_THREADS={OMP_THREADS}`",
        f"- PDF：`{RESULT_PDF}`",
        f"- JSON：`{RESULT_JSON}`",
        "",
        "## 实验组",
        "",
        "| 组别 | profile 行为 | request wall | decode_cpu_path_layer_total | outer gap | gap 占比 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group in results["groups"]:
        gap = group["gap"]
        lines.append(
            f"| `{group['name']}` | {group['description']} | "
            f"{group['request_wall_s']:.3f} s | "
            f"{gap.get('decode_cpu_path_layer_total', 0.0):.3f} s | "
            f"{gap.get('outer_gap', 0.0):.3f} s | "
            f"{gap.get('outer_gap_share', 0.0) * 100.0:.2f}% |"
        )
    lines.extend(["", "## 关键解释", ""])
    group_a = next((g for g in results["groups"] if g["name"] == "A_current_jsonl"), None)
    group_b = next((g for g in results["groups"] if g["name"] == "B_memory_aggregate"), None)
    group_d = next((g for g in results["groups"] if g["name"] == "D_profile_off"), None)
    group_e = next((g for g in results["groups"] if g["name"] == "E_fine_memory"), None)
    if group_a and group_b:
        gap_delta = group_a["gap"]["outer_gap"] - group_b["gap"]["outer_gap"]
        lines.append(
            f"- A 到 B 的 outer gap 差值为 "
            f"`{gap_delta:.3f} s`，"
            "用于估计逐 event JSONL 写入对 gap 的影响。"
        )
        if group_a["gap"]["outer_gap"] > 0:
            lines.append(
                f"- 旧版 `outer gap` 中约 "
                f"`{pct(gap_delta, group_a['gap']['outer_gap']):.2f}%` "
                "来自逐 event JSONL 写入/记录扰动。"
            )
        lines.append(
            f"- 去掉 JSONL 后，B 组剩余 `outer gap` 为 "
            f"`{group_b['gap']['outer_gap']:.3f} s`，占 "
            f"`decode_cpu_path_layer_total` 的 "
            f"`{group_b['gap']['outer_gap_share'] * 100.0:.2f}%`。"
        )
    if group_a and group_d:
        lines.append(
            f"- A 相比 D 的 request wall 差值为 "
            f"`{group_a['request_wall_s'] - group_d['request_wall_s']:.3f} s`，"
            "用于估计当前 profile 总体观察者效应。"
        )
    if group_b and group_d:
        lines.append(
            f"- B 相比 D 的 request wall 差值为 "
            f"`{group_b['request_wall_s'] - group_d['request_wall_s']:.3f} s`，"
            "说明内存聚合模式的观察者效应远小于旧 JSONL 模式。"
        )
    group_c = next((g for g in results["groups"] if g["name"] == "C_outer_only"), None)
    if group_c:
        lines.append(
            "- C 组只记录 `decode_cpu_path_layer_total`，所以 "
            "`outer gap=decode_cpu_path_layer_total` 是实验设计导致的数学结果，"
            "不能解释为真实 gap。"
        )
    if group_e:
        lines.append(
            "- E 组 fine timer 明显扰动运行，因此 E 组细项不能直接按真实运行比例分摊，"
            "只能用于定位进一步插桩时哪些边界会引入明显观察者效应。"
        )
        lines.extend(["", "## E 组细粒度 timer", ""])
        lines.append(
            "注意：E 组因为每层每 token 增加大量 fine timer，运行时间严重膨胀。"
            "下表只用于判断哪些新增插桩点开销/覆盖范围大，不作为真实系统耗时比例。"
        )
        lines.append("")
        lines.append("| event | total s | count | avg ms |")
        lines.append("|---|---:|---:|---:|")
        events = group_e["events"]
        fine_names = FINE_OUTER_GAP_EVENTS + FINE_METADATA_EVENTS + FINE_CPP_WRAPPER_EVENTS
        for name in fine_names:
            item = events.get(name)
            if not item:
                continue
            lines.append(
                f"| `{name}` | {item['total_s']:.6f} | "
                f"{int(item['count'])} | {item['avg_ms']:.6f} |"
            )
    lines.extend([
        "",
        "## 临时环境变量",
        "",
        "- 仅在测试子进程内设置 `OMP_NUM_THREADS`。",
        "- 仅在测试子进程内设置 `VLLM_HETER_PROFILE`、`VLLM_HETER_PROFILE_MODE`、`VLLM_HETER_PROFILE_PATH`、`VLLM_HETER_PROFILE_SUMMARY_PATH`。",
        "- 仅在测试子进程内设置 `VLLM_HETER_CPU_ATTN_LIB`。",
        "- 仅在测试子进程内设置 `no_proxy/NO_PROXY`。",
        "- 未修改系统持久环境变量，未修改系统超线程或 CPU online 设置。",
        "",
        "## 结论",
        "",
        "1. 之前三层图里的 `outer gap` 主要不是 CPU attention 算子，也不是 PCIe 传输，而是旧 profile 方式造成的大量逐 event JSONL 写入/记录扰动。",
        "2. 后续 profile 应优先使用内存聚合/summary 模式，不应再用逐 event JSONL 作为性能口径。",
        "3. 若要继续拆剩余 gap，需要重新设计低扰动 fine timer，避免像 E 组这样在每层每 token 上记录过多事件。",
    ])
    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "timestamp_utc": now_utc(),
        "model": str(MODEL),
        "prompt_len": PROMPT_LEN,
        "output_len": OUTPUT_LEN,
        "batch_size": 1,
        "omp_threads": OMP_THREADS,
        "port": PORT,
        "affinity": AFFINITY,
        "groups": [],
    }
    for group in GROUPS:
        results["groups"].append(run_group(group))
        RESULT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    make_pdf(results)
    write_md(results)
    RESULT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(RESULT_JSON)
    print(RESULT_PDF)
    print(RESULT_MD)


if __name__ == "__main__":
    main()
