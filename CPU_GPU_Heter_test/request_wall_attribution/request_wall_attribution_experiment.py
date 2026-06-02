#!/usr/bin/env python3
"""Attribute request wall time around the heter CPU decode path."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
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
OUT_DIR = ROOT / "CPU_GPU_Heter_test" / "request_wall_attribution"
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

RESULT_JSON = OUT_DIR / "request_wall_attribution_results.json"
RESULT_PDF = OUT_DIR / "request_wall_attribution.pdf"
RESULT_MD = OUT_DIR / "request_wall_attribution_记录.md"

LATIN_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
CJK_FONT = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")

GROUPS = [
    {
        "name": "A_baseline_no_request_profile",
        "request_profile": "off",
        "attention_profile": "memory",
        "desc": "低扰动 attention memory profile；关闭 request profile。",
    },
    {
        "name": "B_request_memory",
        "request_profile": "memory",
        "attention_profile": "memory",
        "desc": "request profile 内存聚合；attention profile 也使用 memory。",
    },
    {
        "name": "C_request_jsonl",
        "request_profile": "jsonl",
        "attention_profile": "memory",
        "desc": "request profile 高频 JSONL；仅用于定位，不作为原始耗时口径。",
    },
]

REQUEST_EVENTS = [
    "openai_completion_create_total",
    "openai_completion_render_request",
    "openai_completion_generate_iteration",
    "openai_completion_build_response",
    "engine_core_step_total",
    "engine_core_step_enqueue_total",
    "engine_core_schedule",
    "engine_core_execute_model_submit",
    "engine_core_wait_execute_model_future",
    "engine_core_wait_model_or_sample_future",
    "engine_core_sample_tokens_call",
    "engine_core_sample_tokens_call_submit",
    "engine_core_scheduler_update_from_output",
    "gpu_worker_execute_model_total",
    "gpu_worker_model_runner_execute_model",
    "gpu_worker_sample_tokens_total",
    "gpu_model_runner_execute_model_total",
    "gpu_model_runner_preprocess_total",
    "gpu_model_runner_debug_log_write",
    "gpu_model_runner_model_forward",
    "gpu_model_runner_postprocess_total",
    "gpu_model_runner_sample_tokens_total",
    "gpu_model_runner_sample_kernel_total",
    "gpu_model_runner_bookkeeping_sync",
    "gpu_model_runner_build_output",
    "output_processor_process_outputs_total",
    "output_processor_process_one_output",
]

ATTENTION_EVENTS = [
    "decode_cpu_path_layer_total",
    "cpu_attention_cpp_total",
    "decode_metadata_d2h",
    "decode_kv_d2h",
    "decode_kv_cache_write_total",
    "decode_output_h2d",
    "decode_wait_prefill_d2h",
    "decode_query_d2h",
    "decode_output_cpu_alloc",
    "gpu_attention_forward",
    "prefill_kv_d2h_enqueue",
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
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_ready(port: int, timeout_s: float, log_path: Path) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.time() + timeout_s
    last_error = None
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


def summarize_jsonl(path: Path) -> dict[str, dict[str, float]]:
    events: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "total_s": 0.0, "max_ms": 0.0}
    )
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            name = payload.get("event")
            elapsed = float(payload.get("elapsed_s", 0.0))
            item = events[name]
            item["count"] += 1
            item["total_s"] += elapsed
            item["max_ms"] = max(item["max_ms"], elapsed * 1000.0)
    for item in events.values():
        count = int(item["count"])
        item["avg_ms"] = item["total_s"] / count * 1000.0 if count else 0.0
    return dict(events)


def read_summaries(prefix: Path) -> dict[str, dict[str, float]]:
    events: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "total_s": 0.0, "max_ms": 0.0}
    )
    for path in sorted(prefix.parent.glob(f"{prefix.stem}.pid*.json")):
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


def event_total(events: dict[str, dict[str, float]], name: str) -> float:
    return float(events.get(name, {}).get("total_s", 0.0))


def run_group(group: dict[str, str], prompt: str) -> dict[str, Any]:
    name = group["name"]
    print(f"[{now_utc()}] run {name}", flush=True)
    log_path = OUT_DIR / f"{name}_server.log"
    request_jsonl_path = OUT_DIR / f"{name}_request_events.jsonl"
    request_summary_prefix = OUT_DIR / f"{name}_request_summary.json"
    attention_summary_prefix = OUT_DIR / f"{name}_attention_summary.json"

    for path in [log_path, request_jsonl_path]:
        if path.exists():
            path.unlink()
    for prefix in [request_summary_prefix, attention_summary_prefix]:
        for path in OUT_DIR.glob(f"{prefix.stem}.pid*.json"):
            path.unlink()

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "OMP_NUM_THREADS": str(OMP_THREADS),
            "VLLM_HETER_CPU_ATTN_LIB": str(CPU_ATTN_LIB),
            "VLLM_HETER_PROFILE": "1",
            "VLLM_HETER_PROFILE_MODE": group["attention_profile"],
            "VLLM_HETER_PROFILE_SUMMARY_PATH": str(attention_summary_prefix),
            "VLLM_HETER_PROFILE_FLUSH_INTERVAL_S": "2.0",
            "VLLM_HETER_REQUEST_PROFILE": "0",
            "no_proxy": "localhost,127.0.0.1",
            "NO_PROXY": "localhost,127.0.0.1",
        }
    )
    if group["request_profile"] != "off":
        env["VLLM_HETER_REQUEST_PROFILE"] = "1"
        env["VLLM_HETER_REQUEST_PROFILE_MODE"] = group["request_profile"]
        env["VLLM_HETER_REQUEST_PROFILE_SUMMARY_PATH"] = str(request_summary_prefix)
        env["VLLM_HETER_REQUEST_PROFILE_PATH"] = str(request_jsonl_path)
        env["VLLM_HETER_REQUEST_PROFILE_FLUSH_INTERVAL_S"] = "2.0"

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

    request_events = (
        summarize_jsonl(request_jsonl_path)
        if group["request_profile"] == "jsonl"
        else read_summaries(request_summary_prefix)
    )
    attention_events = read_summaries(attention_summary_prefix)
    result = {
        "name": name,
        "description": group["desc"],
        "request_profile": group["request_profile"],
        "attention_profile": group["attention_profile"],
        "request_wall_s": wall_s,
        "usage": usage,
        "finish_reason": finish_reason,
        "request_events": request_events,
        "attention_events": attention_events,
        "server_log": str(log_path),
        "request_jsonl": str(request_jsonl_path) if request_jsonl_path.exists() else None,
        "request_summary_files": [
            str(p) for p in sorted(OUT_DIR.glob(f"{request_summary_prefix.stem}.pid*.json"))
        ],
        "attention_summary_files": [
            str(p) for p in sorted(OUT_DIR.glob(f"{attention_summary_prefix.stem}.pid*.json"))
        ],
    }
    print(
        f"[{now_utc()}] done {name}: wall={wall_s:.3f}s "
        f"decode={event_total(attention_events, 'decode_cpu_path_layer_total'):.3f}s",
        flush=True,
    )
    return result


def top_events(events: dict[str, dict[str, float]], names: list[str], limit: int) -> list[tuple[str, float]]:
    items = [(name, event_total(events, name)) for name in names]
    items = [(name, value) for name, value in items if value > 0]
    return sorted(items, key=lambda item: item[1], reverse=True)[:limit]


def scheduled_jsonl_breakdown(path_str: str | None, event_name: str) -> dict[str, dict[str, float]]:
    if not path_str:
        return {}
    path = Path(path_str)
    if not path.exists():
        return {}
    out: dict[str, dict[str, float]] = defaultdict(
        lambda: {"count": 0, "total_s": 0.0, "max_ms": 0.0}
    )
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event") != event_name:
                continue
            key = str(event.get("scheduled_tokens", "NA"))
            elapsed = float(event.get("elapsed_s", 0.0))
            item = out[key]
            item["count"] += 1
            item["total_s"] += elapsed
            item["max_ms"] = max(item["max_ms"], elapsed * 1000.0)
    for item in out.values():
        count = int(item["count"])
        item["avg_ms"] = item["total_s"] / count * 1000.0 if count else 0.0
    return dict(out)


def exclusive_request_breakdown(group: dict[str, Any]) -> list[tuple[str, float]]:
    request_events = group["request_events"]
    attention_events = group["attention_events"]
    wall = float(group["request_wall_s"])
    decode_cpu = event_total(attention_events, "decode_cpu_path_layer_total")
    model_forward = event_total(request_events, "gpu_model_runner_model_forward")
    preprocess = event_total(request_events, "gpu_model_runner_preprocess_total")
    postprocess = event_total(request_events, "gpu_model_runner_postprocess_total")
    sample = event_total(request_events, "gpu_model_runner_sample_tokens_total")
    model_forward_non_cpu_attn = max(0.0, model_forward - decode_cpu)
    other = max(
        0.0,
        wall - decode_cpu - model_forward_non_cpu_attn - preprocess - postprocess - sample,
    )
    return [
        ("Decode CPU attention path", decode_cpu),
        ("Other model forward", model_forward_non_cpu_attn),
        ("Model runner preprocess", preprocess),
        ("Sampling/bookkeeping", sample),
        ("Postprocess/logits", postprocess),
        ("API/scheduler/output other", other),
    ]


def make_pdf(results: dict[str, Any]) -> None:
    groups = results["groups"]
    names = [g["name"].replace("_", "\n") for g in groups]
    request_walls = [g["request_wall_s"] for g in groups]
    decode_totals = [
        event_total(g["attention_events"], "decode_cpu_path_layer_total") for g in groups
    ]

    with PdfPages(RESULT_PDF) as pdf:
        fig, axes = plt.subplots(2, 1, figsize=(13.5, 9.5), constrained_layout=True)
        fig.suptitle("Request Wall Attribution", fontsize=16, fontproperties=FONT)
        x = range(len(groups))
        axes[0].bar(x, request_walls, color="#4C78A8")
        axes[0].set_title("Request wall time", loc="left", fontproperties=FONT)
        axes[0].set_ylabel("seconds", fontproperties=FONT)
        axes[0].set_xticks(x, names, fontproperties=FONT)
        axes[0].grid(axis="y", alpha=0.25)
        axes[1].bar(x, decode_totals, color="#F58518")
        axes[1].set_title("decode_cpu_path_layer_total", loc="left", fontproperties=FONT)
        axes[1].set_ylabel("seconds", fontproperties=FONT)
        axes[1].set_xticks(x, names, fontproperties=FONT)
        axes[1].grid(axis="y", alpha=0.25)
        for ax, values in zip(axes, [request_walls, decode_totals]):
            max_v = max(values) if values else 1.0
            for idx, value in enumerate(values):
                ax.text(
                    idx,
                    value + max_v * 0.02,
                    f"{value:.3f}",
                    ha="center",
                    fontsize=9,
                    fontproperties=FONT,
                )
        pdf.savefig(fig)
        plt.close(fig)

        group_b = next(g for g in groups if g["name"] == "B_request_memory")
        breakdown = exclusive_request_breakdown(group_b)
        fig, ax = plt.subplots(figsize=(14.0, 8.5), constrained_layout=True)
        labels = [name for name, _ in reversed(breakdown)]
        values = [value for _, value in reversed(breakdown)]
        colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#72B7B2", "#BAB0AC"]
        ax.barh(labels, values, color=list(reversed(colors)))
        ax.set_title(
            "B request memory: mutually exclusive request-wall breakdown",
            fontproperties=FONT,
        )
        ax.set_xlabel("seconds", fontproperties=FONT)
        ax.grid(axis="x", alpha=0.25)
        max_v = max(values) if values else 1.0
        wall = group_b["request_wall_s"]
        for idx, value in enumerate(values):
            ax.text(
                value + max_v * 0.02,
                idx,
                f"{value:.3f}s ({value / wall * 100.0:.1f}%)",
                va="center",
                fontsize=8,
                fontproperties=FONT,
            )
        ax.text(
            0.98,
            0.04,
            "Other model forward = model_forward - decode_cpu_path_layer_total.",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
            fontproperties=FONT,
        )
        pdf.savefig(fig)
        plt.close(fig)

        group_c = next(g for g in groups if g["name"] == "C_request_jsonl")
        c_forward = scheduled_jsonl_breakdown(
            group_c.get("request_jsonl"),
            "gpu_model_runner_model_forward",
        )
        c_decode_cpu = event_total(group_c["attention_events"], "decode_cpu_path_layer_total")
        c_prefill_forward = c_forward.get(str(results["prompt_len"] + 1), {}).get(
            "total_s", 0.0
        )
        c_decode_forward = c_forward.get("1", {}).get("total_s", 0.0)
        c_decode_other = max(0.0, c_decode_forward - c_decode_cpu)
        c_sched_items = [
            ("Prefill model forward", c_prefill_forward),
            ("Decode CPU attention path", c_decode_cpu),
            ("Decode other model forward", c_decode_other),
        ]
        fig, ax = plt.subplots(figsize=(14.0, 7.0), constrained_layout=True)
        labels = [name for name, _ in reversed(c_sched_items)]
        values = [value for _, value in reversed(c_sched_items)]
        ax.barh(labels, values, color=["#E45756", "#4C78A8", "#F58518"])
        ax.set_title(
            "C JSONL scheduled_tokens split: prefill vs decode model forward",
            fontproperties=FONT,
        )
        ax.set_xlabel("seconds", fontproperties=FONT)
        ax.grid(axis="x", alpha=0.25)
        max_v = max(values) if values else 1.0
        total_v = sum(values)
        for idx, value in enumerate(values):
            ax.text(
                value + max_v * 0.02,
                idx,
                f"{value:.3f}s ({value / total_v * 100.0:.1f}%)",
                va="center",
                fontsize=8,
                fontproperties=FONT,
            )
        ax.text(
            0.98,
            0.04,
            "scheduled_tokens=8001 is prefill; scheduled_tokens=1 occurs for 399 decode steps.",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
            fontproperties=FONT,
        )
        pdf.savefig(fig)
        plt.close(fig)

        items = top_events(group_b["request_events"], REQUEST_EVENTS, 18)
        fig, ax = plt.subplots(figsize=(14.0, 9.5), constrained_layout=True)
        labels = [name.replace("_", " ") for name, _ in reversed(items)]
        values = [value for _, value in reversed(items)]
        ax.barh(labels, values, color="#54A24B")
        ax.set_title("B request memory: low-overhead request/profile events", fontproperties=FONT)
        ax.set_xlabel("seconds", fontproperties=FONT)
        ax.grid(axis="x", alpha=0.25)
        max_v = max(values) if values else 1.0
        for idx, value in enumerate(values):
            ax.text(value + max_v * 0.02, idx, f"{value:.3f}s", va="center", fontsize=8)
        pdf.savefig(fig)
        plt.close(fig)

        items = top_events(group_c["request_events"], REQUEST_EVENTS, 18)
        fig, ax = plt.subplots(figsize=(14.0, 9.5), constrained_layout=True)
        labels = [name.replace("_", " ") for name, _ in reversed(items)]
        values = [value for _, value in reversed(items)]
        ax.barh(labels, values, color="#E45756")
        ax.set_title("C request JSONL: high-frequency diagnostic events", fontproperties=FONT)
        ax.set_xlabel("seconds", fontproperties=FONT)
        ax.grid(axis="x", alpha=0.25)
        max_v = max(values) if values else 1.0
        for idx, value in enumerate(values):
            ax.text(value + max_v * 0.02, idx, f"{value:.3f}s", va="center", fontsize=8)
        ax.text(
            0.98,
            0.03,
            "JSONL is diagnostic and includes observer overhead.",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(14.0, 8.0), constrained_layout=True)
        ax.axis("off")
        group_a = next(g for g in groups if g["name"] == "A_baseline_no_request_profile")
        group_b = next(g for g in groups if g["name"] == "B_request_memory")
        decode_a = event_total(group_a["attention_events"], "decode_cpu_path_layer_total")
        decode_b = event_total(group_b["attention_events"], "decode_cpu_path_layer_total")
        b_breakdown = exclusive_request_breakdown(group_b)
        b_other_forward = dict(b_breakdown)["Other model forward"]
        lines = [
            f"Generated: {results['timestamp_utc']}",
            f"Workload: prompt={PROMPT_LEN}, output={OUTPUT_LEN}, batch=1, OMP_NUM_THREADS={OMP_THREADS}",
            f"12768 source: (completion_tokens - 1) * 32 = ({OUTPUT_LEN} - 1) * 32.",
            f"Baseline request wall - decode_cpu_path_layer_total = {group_a['request_wall_s'] - decode_a:.3f}s.",
            f"Memory request profile observer delta vs baseline = {group_b['request_wall_s'] - group_a['request_wall_s']:.3f}s.",
            f"Memory profile request wall - decode_cpu_path_layer_total = {group_b['request_wall_s'] - decode_b:.3f}s.",
            f"In B, other model forward after subtracting CPU attention = {b_other_forward:.3f}s.",
            "Interpretation: use A/B for low-overhead timing; use C JSONL only to locate detailed events.",
        ]
        y = 0.95
        for line in lines:
            ax.text(0.02, y, line, fontsize=12, va="top", fontproperties=FONT)
            y -= 0.10
        pdf.savefig(fig)
        plt.close(fig)


def write_md(results: dict[str, Any]) -> None:
    lines = [
        "# Request Wall Attribution 实验记录",
        "",
        f"- 时间：`{results['timestamp_utc']}`",
        f"- workload：`prompt={PROMPT_LEN}`，`output={OUTPUT_LEN}`，`batch=1`",
        f"- OpenMP：`OMP_NUM_THREADS={OMP_THREADS}`",
        f"- PDF：`{RESULT_PDF}`",
        f"- JSON：`{RESULT_JSON}`",
        "",
        "## 12768 的来源",
        "",
        "- 上一轮数据中的 `12768` 来自 `decode_cpu_path_layer_total.count`。",
        "- 本次请求 `completion_tokens=400`，第 1 个输出 token 来自 prefill forward 后的采样，不走 decode CPU attention。",
        "- 后续 decode step 数为 `400 - 1 = 399`。",
        "- Llama-3.1-8B 有 32 个 decoder layers，因此 decode CPU attention 调用次数为 `399 * 32 = 12768`。",
        "",
        "## 实验组",
        "",
        "| 组别 | request profile | request wall | decode_cpu_path_layer_total | wall - decode |",
        "|---|---|---:|---:|---:|",
    ]
    for group in results["groups"]:
        decode = event_total(group["attention_events"], "decode_cpu_path_layer_total")
        lines.append(
            f"| `{group['name']}` | {group['description']} | "
            f"{group['request_wall_s']:.3f} s | {decode:.3f} s | "
            f"{group['request_wall_s'] - decode:.3f} s |"
        )

    group_a = next(g for g in results["groups"] if g["name"] == "A_baseline_no_request_profile")
    group_b = next(g for g in results["groups"] if g["name"] == "B_request_memory")
    group_c = next(g for g in results["groups"] if g["name"] == "C_request_jsonl")
    b_breakdown = exclusive_request_breakdown(group_b)
    lines.extend(
        [
            "",
            "## 低扰动互斥拆解",
            "",
            "该表用 B 组 request memory profile 做互斥拆解，避免把嵌套事件重复相加。",
            "",
            "| component | total s | request wall 占比 | 说明 |",
            "|---|---:|---:|---|",
        ]
    )
    component_desc = {
        "Decode CPU attention path": "`decode_cpu_path_layer_total`，也就是异构 decode attention 外层路径。",
        "Other model forward": "`gpu_model_runner_model_forward - decode_cpu_path_layer_total`，主要对应 prefill forward 以及 decode 中除 self-attention 之外的 GPU 算子，例如 QKV/O projection、MLP、norm/residual、logits 前隐藏状态计算等。",
        "Model runner preprocess": "更新 batch 状态、准备输入、slot mapping、attention metadata 等。",
        "Sampling/bookkeeping": "采样、token 状态更新、异步输出准备等。",
        "Postprocess/logits": "model forward 之后的 hidden state 选择、logits 计算和 ModelRunner postprocess。",
        "API/scheduler/output other": "上述项之外的 API server、scheduler、output processor、计时误差和未覆盖边界。",
    }
    for name, value in b_breakdown:
        lines.append(
            f"| `{name}` | {value:.6f} | "
            f"{value / group_b['request_wall_s'] * 100.0:.2f}% | "
            f"{component_desc[name]} |"
        )

    c_forward = scheduled_jsonl_breakdown(
        group_c.get("request_jsonl"),
        "gpu_model_runner_model_forward",
    )
    c_prefill_forward = c_forward.get(str(results["prompt_len"] + 1), {}).get(
        "total_s", 0.0
    )
    c_decode_forward = c_forward.get("1", {}).get("total_s", 0.0)
    c_decode_cpu = event_total(group_c["attention_events"], "decode_cpu_path_layer_total")
    c_decode_other = max(0.0, c_decode_forward - c_decode_cpu)
    lines.extend(
        [
            "",
            "## 高频 JSONL 的 prefill/decode 定位",
            "",
            "该表只用于定位，因为 C 组包含 JSONL 记录开销；但它携带 `scheduled_tokens` 元数据，可以区分 prefill 和 decode。",
            "",
            "| item | total s | count/来源 |",
            "|---|---:|---|",
            f"| Prefill model forward | {c_prefill_forward:.6f} | `scheduled_tokens={results['prompt_len'] + 1}`，1 次 |",
            f"| Decode model forward | {c_decode_forward:.6f} | `scheduled_tokens=1`，399 次 |",
            f"| Decode CPU attention path | {c_decode_cpu:.6f} | attention profile，12768 次 layer 调用 |",
            f"| Decode other model forward | {c_decode_other:.6f} | `decode model forward - decode CPU attention path` |",
            "",
            "关键解释：`request wall - decode_cpu_path_layer_total` 大，并不是因为还有一个同量级的未知同步空洞；主要是因为完整 request wall 还包含 prefill forward，以及每个 decode step 中 self-attention 之外的 GPU 计算路径。",
        ]
    )
    lines.extend(
        [
            "",
            "## 低扰动 request profile 主要事件",
            "",
            "| event | total s | count | avg ms |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, _ in top_events(group_b["request_events"], REQUEST_EVENTS, 24):
        item = group_b["request_events"][name]
        lines.append(
            f"| `{name}` | {item['total_s']:.6f} | "
            f"{int(item['count'])} | {item['avg_ms']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## 高频 JSONL 诊断事件",
            "",
            "注意：本组包含逐 event JSONL 写入观察者开销，不能作为原始系统耗时比例，只用于定位细粒度阶段。",
            "",
            "| event | total s | count | avg ms |",
            "|---|---:|---:|---:|",
        ]
    )
    for name, _ in top_events(group_c["request_events"], REQUEST_EVENTS, 24):
        item = group_c["request_events"][name]
        lines.append(
            f"| `{name}` | {item['total_s']:.6f} | "
            f"{int(item['count'])} | {item['avg_ms']:.6f} |"
        )

    decode_a = event_total(group_a["attention_events"], "decode_cpu_path_layer_total")
    decode_b = event_total(group_b["attention_events"], "decode_cpu_path_layer_total")
    b_breakdown_dict = dict(b_breakdown)
    lines.extend(
        [
            "",
            "## 初步解释口径",
            "",
            f"- baseline 中 `request wall - decode_cpu_path_layer_total = {group_a['request_wall_s'] - decode_a:.3f} s`。",
            f"- 低扰动 request profile 相比 baseline 的 request wall 差值为 `{group_b['request_wall_s'] - group_a['request_wall_s']:.3f} s`，用于估计 request profile 观察者效应。",
            f"- 低扰动 request profile 中 `request wall - decode_cpu_path_layer_total = {group_b['request_wall_s'] - decode_b:.3f} s`。",
            f"- B 组中，`Other model forward = {b_breakdown_dict['Other model forward']:.3f} s`，占 request wall 的 `{b_breakdown_dict['Other model forward'] / group_b['request_wall_s'] * 100.0:.2f}%`；这是 wall-decode 差值的主要来源。",
            f"- C 组 JSONL 的 scheduled-token 拆分显示：prefill model forward 约 `{c_prefill_forward:.3f} s`，399 次 decode model forward 约 `{c_decode_forward:.3f} s`，其中 decode CPU attention path 约 `{c_decode_cpu:.3f} s`。",
            "- 结论应优先基于 baseline 与 memory profile；JSONL 组只辅助定位具体操作，不进入原始开销结论。",
            "",
            "## 临时环境变量",
            "",
            "- 仅在测试子进程内设置 `OMP_NUM_THREADS`。",
            "- 仅在测试子进程内设置 `VLLM_HETER_PROFILE*` 和 `VLLM_HETER_REQUEST_PROFILE*`。",
            "- 仅在测试子进程内设置 `VLLM_HETER_CPU_ATTN_LIB`。",
            "- 仅在测试子进程内设置 `no_proxy/NO_PROXY`。",
            "- 未修改系统持久环境变量，未修改系统超线程或 CPU online 设置。",
        ]
    )
    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompt = make_prompt(PROMPT_LEN)
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
        results["groups"].append(run_group(group, prompt))
        RESULT_JSON.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    make_pdf(results)
    write_md(results)
    RESULT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(RESULT_JSON)
    print(RESULT_PDF)
    print(RESULT_MD)


if __name__ == "__main__":
    main()
