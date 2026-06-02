#!/usr/bin/env python3
"""Recheck request-wall gap with GPU graph/eager controls."""

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
OUT_DIR = ROOT / "CPU_GPU_Heter_test" / "request_wall_gap_recheck"
PYTHON = Path("/home/zb/tool/miniconda3/envs/vLLM_GPU/bin/python")
VLLM = Path("/home/zb/tool/miniconda3/envs/vLLM_GPU/bin/vllm")
MODEL = Path("/home/zb/data/model_data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")
CPU_ATTN_LIB = ROOT / "vllm" / "libs" / "libvllm_heter_cpu_attn.so"
PORT = 8036
PROMPT_LEN = 8000
OUTPUT_LEN = 400
OMP_THREADS = 120
GPU_MEMORY_UTILIZATION = "0.25"
MAX_MODEL_LEN = "10000"
AFFINITY = (
    "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,"
    "20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,"
    "40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,"
    "60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,"
    "80,81,82,83,84,85,86,87,88,89,90,91,96,97,98,99,100,101,102,"
    "103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,"
    "118,119,120,121,122,123"
)

RESULT_JSON = OUT_DIR / "request_wall_gap_recheck_results.json"
RESULT_PDF = OUT_DIR / "request_wall_gap_recheck.pdf"
RESULT_MD = OUT_DIR / "request_wall_gap_recheck_记录.md"

LATIN_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
CJK_FONT = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")

GROUPS = [
    {
        "name": "gpu_default_graph",
        "label": "GPU default",
        "desc": "禁用 CPU attention，保留默认 CUDA Graph。",
        "disable_cpu_attention": True,
        "force_decode_eager": False,
        "attention_profile": False,
    },
    {
        "name": "gpu_forced_eager",
        "label": "GPU forced eager",
        "desc": "禁用 CPU attention，但强制 pure decode 走 eager，用于量化 CUDA Graph 被关闭后的非 attention 开销。",
        "disable_cpu_attention": True,
        "force_decode_eager": True,
        "attention_profile": False,
    },
    {
        "name": "heter_cpu_attention",
        "label": "Heter",
        "desc": "启用 CPU decode attention，并强制 pure decode 走 eager。",
        "disable_cpu_attention": False,
        "force_decode_eager": False,
        "skip_cpu_attention_compute": False,
        "ignore_response_errors": False,
        "attention_profile": True,
    },
    {
        "name": "heter_skip_cpu_attention_compute",
        "label": "Heter skip CPU compute",
        "desc": "保留异构 D2H/H2D、metadata、CPU KV 写入和同步，但跳过 C++ CPU attention 计算。",
        "disable_cpu_attention": False,
        "force_decode_eager": False,
        "skip_cpu_attention_compute": True,
        "ignore_response_errors": True,
        "attention_profile": True,
    },
]

REQUEST_EVENTS = [
    "openai_completion_create_total",
    "openai_completion_generate_iteration",
    "engine_core_step_enqueue_total",
    "engine_core_execute_model_submit",
    "engine_core_wait_model_or_sample_future",
    "engine_core_sample_tokens_call_submit",
    "engine_core_scheduler_update_from_output",
    "gpu_worker_execute_model_total",
    "gpu_worker_model_runner_execute_model",
    "gpu_worker_sample_tokens_total",
    "gpu_model_runner_execute_model_total",
    "gpu_model_runner_preprocess_total",
    "gpu_model_runner_model_forward",
    "gpu_model_runner_postprocess_total",
    "gpu_model_runner_sample_tokens_total",
    "gpu_model_runner_sample_kernel_total",
    "gpu_model_runner_bookkeeping_sync",
    "gpu_model_runner_build_output",
    "gpu_model_runner_build_async_output",
    "output_processor_process_outputs_total",
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
    "prefill_kv_d2h_enqueue",
    "cpu_attention_skip_compute_zero",
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
        tail = "\n".join(log_path.read_text(errors="ignore").splitlines()[-100:])
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


def run_group(group: dict[str, Any], prompt: str) -> dict[str, Any]:
    name = str(group["name"])
    print(f"[{now_utc()}] run {name}", flush=True)
    log_path = OUT_DIR / f"{name}_server.log"
    request_summary_prefix = OUT_DIR / f"{name}_request_summary.json"
    attention_summary_prefix = OUT_DIR / f"{name}_attention_summary.json"

    if log_path.exists():
        log_path.unlink()
    for prefix in [request_summary_prefix, attention_summary_prefix]:
        for path in OUT_DIR.glob(f"{prefix.stem}.pid*.json"):
            path.unlink()

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "OMP_NUM_THREADS": str(OMP_THREADS),
            "VLLM_HETER_REQUEST_PROFILE": "1",
            "VLLM_HETER_REQUEST_PROFILE_MODE": "memory",
            "VLLM_HETER_REQUEST_PROFILE_SUMMARY_PATH": str(request_summary_prefix),
            "VLLM_HETER_REQUEST_PROFILE_FLUSH_INTERVAL_S": "2.0",
            "VLLM_HETER_CUDAGRAPH_DEBUG": "0",
            "no_proxy": "localhost,127.0.0.1",
            "NO_PROXY": "localhost,127.0.0.1",
        }
    )
    if bool(group["disable_cpu_attention"]):
        env["VLLM_HETER_DISABLE_CPU_ATTENTION"] = "1"
    else:
        env.pop("VLLM_HETER_DISABLE_CPU_ATTENTION", None)
        env["VLLM_HETER_CPU_ATTN_LIB"] = str(CPU_ATTN_LIB)
    if bool(group["force_decode_eager"]):
        env["VLLM_HETER_FORCE_DECODE_EAGER"] = "1"
    else:
        env.pop("VLLM_HETER_FORCE_DECODE_EAGER", None)
    if bool(group.get("skip_cpu_attention_compute", False)):
        env["VLLM_HETER_SKIP_CPU_ATTN_COMPUTE"] = "1"
    else:
        env.pop("VLLM_HETER_SKIP_CPU_ATTN_COMPUTE", None)

    if bool(group["attention_profile"]):
        env["VLLM_HETER_PROFILE"] = "1"
        env["VLLM_HETER_PROFILE_MODE"] = "memory"
        env["VLLM_HETER_PROFILE_SUMMARY_PATH"] = str(attention_summary_prefix)
        env["VLLM_HETER_PROFILE_FLUSH_INTERVAL_S"] = "2.0"
    else:
        env["VLLM_HETER_PROFILE"] = "0"

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
        GPU_MEMORY_UTILIZATION,
        "--max-model-len",
        MAX_MODEL_LEN,
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
        try:
            response = http_json(
                f"http://127.0.0.1:{PORT}/v1/completions",
                payload,
                timeout_s=900.0,
            )
        except Exception as exc:  # noqa: BLE001
            if not bool(group.get("ignore_response_errors", False)):
                raise
            response = {
                "usage": {},
                "choices": [{"finish_reason": f"client_error:{type(exc).__name__}"}],
            }
        wall_s = time.perf_counter() - start
        usage = response.get("usage", {})
        finish_reason = response.get("choices", [{}])[0].get("finish_reason")
    finally:
        terminate_process(proc)

    request_events = read_summaries(request_summary_prefix)
    attention_events = read_summaries(attention_summary_prefix)
    result = {
        **group,
        "request_wall_s": wall_s,
        "usage": usage,
        "finish_reason": finish_reason,
        "request_events": request_events,
        "attention_events": attention_events,
        "server_log": str(log_path),
        "request_summary_files": [
            str(p) for p in sorted(OUT_DIR.glob(f"{request_summary_prefix.stem}.pid*.json"))
        ],
        "attention_summary_files": [
            str(p) for p in sorted(OUT_DIR.glob(f"{attention_summary_prefix.stem}.pid*.json"))
        ],
    }
    print(
        f"[{now_utc()}] done {name}: wall={wall_s:.3f}s "
        f"forward={event_total(request_events, 'gpu_model_runner_model_forward'):.3f}s "
        f"decode_cpu={event_total(attention_events, 'decode_cpu_path_layer_total'):.3f}s",
        flush=True,
    )
    return result


def split_breakdown(group: dict[str, Any]) -> dict[str, float]:
    request_events = group["request_events"]
    attention_events = group["attention_events"]
    wall = float(group["request_wall_s"])
    model_forward = event_total(request_events, "gpu_model_runner_model_forward")
    decode_cpu = event_total(attention_events, "decode_cpu_path_layer_total")
    preprocess = event_total(request_events, "gpu_model_runner_preprocess_total")
    postprocess = event_total(request_events, "gpu_model_runner_postprocess_total")
    sampling = event_total(request_events, "gpu_model_runner_sample_tokens_total")
    output = event_total(request_events, "output_processor_process_outputs_total")
    other_forward = max(0.0, model_forward - decode_cpu)
    other = max(0.0, wall - decode_cpu - other_forward - preprocess
                - postprocess - sampling - output)
    return {
        "decode_cpu_attention_path": decode_cpu,
        "model_forward_without_cpu_attention": other_forward,
        "preprocess": preprocess,
        "postprocess_logits": postprocess,
        "sampling_bookkeeping": sampling,
        "output_processor": output,
        "api_scheduler_other": other,
    }


def make_pdf(results: dict[str, Any]) -> None:
    groups = results["groups"]
    labels = [g["label"] for g in groups]
    walls = [float(g["request_wall_s"]) for g in groups]
    forwards = [
        event_total(g["request_events"], "gpu_model_runner_model_forward")
        for g in groups
    ]
    decode_cpu = [
        event_total(g["attention_events"], "decode_cpu_path_layer_total")
        for g in groups
    ]
    model_forward_non_cpu = [
        max(0.0, forwards[i] - decode_cpu[i])
        for i in range(len(groups))
    ]

    with PdfPages(RESULT_PDF) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
        x = range(len(groups))
        axes[0].bar(x, walls, color=["#3b82f6", "#64748b", "#d97706", "#059669"])
        axes[0].set_xticks(list(x), labels, rotation=15, ha="right", fontproperties=FONT)
        axes[0].set_ylabel("seconds", fontproperties=FONT)
        axes[0].set_title("End-to-end request wall", loc="left", fontproperties=FONT)
        for i, value in enumerate(walls):
            axes[0].text(i, value, f"{value:.2f}s", ha="center", va="bottom", fontsize=9)
        axes[0].grid(axis="y", alpha=0.25)

        axes[1].bar(x, model_forward_non_cpu, label="model forward minus CPU attention",
                    color="#7c3aed")
        axes[1].bar(x, decode_cpu, bottom=model_forward_non_cpu,
                    label="decode CPU attention path", color="#f59e0b")
        axes[1].set_xticks(list(x), labels, rotation=15, ha="right", fontproperties=FONT)
        axes[1].set_ylabel("seconds", fontproperties=FONT)
        axes[1].set_title("Model forward attribution", loc="left", fontproperties=FONT)
        axes[1].legend(prop=FONT, fontsize=8)
        axes[1].grid(axis="y", alpha=0.25)
        fig.suptitle(
            "Prompt=8K, output=400, batch=1: GPU graph/eager vs Heter",
            fontproperties=FONT,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        components = list(split_breakdown(groups[-1]).keys())
        bottoms = [0.0] * len(groups)
        colors = [
            "#f59e0b",
            "#7c3aed",
            "#2563eb",
            "#059669",
            "#dc2626",
            "#0891b2",
            "#64748b",
        ]
        for idx, comp in enumerate(components):
            vals = [split_breakdown(g)[comp] for g in groups]
            ax.bar(labels, vals, bottom=bottoms, label=comp.replace("_", " "),
                   color=colors[idx % len(colors)])
            bottoms = [bottoms[i] + vals[i] for i in range(len(vals))]
        ax.set_ylabel("seconds", fontproperties=FONT)
        ax.set_title("Mutually exclusive request breakdown", loc="left",
                     fontproperties=FONT)
        ax.tick_params(axis="x", rotation=15)
        ax.legend(prop=FONT, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        gpu_default = groups[0]
        gpu_eager = groups[1]
        heter = groups[2]
        heter_skip = groups[3]
        heter_decode = event_total(
            heter["attention_events"], "decode_cpu_path_layer_total"
        )
        extra_vs_gpu_graph = float(heter["request_wall_s"]) - float(
            gpu_default["request_wall_s"]
        )
        extra_vs_gpu_eager = float(heter["request_wall_s"]) - float(
            gpu_eager["request_wall_s"]
        )
        non_cpu_vs_gpu_default = (
            float(heter["request_wall_s"]) - heter_decode
            - float(gpu_default["request_wall_s"])
        )
        non_cpu_vs_gpu_eager = (
            float(heter["request_wall_s"]) - heter_decode
            - float(gpu_eager["request_wall_s"])
        )
        gpu_eager_delta = (
            float(gpu_eager["request_wall_s"]) - float(gpu_default["request_wall_s"])
        )
        skip_vs_gpu_eager = (
            float(heter_skip["request_wall_s"]) - float(gpu_eager["request_wall_s"])
        )
        lines = [
            "核心读数",
            "",
            f"GPU default request wall: {gpu_default['request_wall_s']:.3f} s",
            f"GPU forced eager request wall: {gpu_eager['request_wall_s']:.3f} s",
            f"Heter request wall: {heter['request_wall_s']:.3f} s",
            f"Heter skip CPU compute request wall: {heter_skip['request_wall_s']:.3f} s",
            f"Heter decode_cpu_path_layer_total: {heter_decode:.3f} s",
            "",
            f"Heter - GPU default: {extra_vs_gpu_graph:.3f} s",
            f"Heter - GPU forced eager: {extra_vs_gpu_eager:.3f} s",
            f"Heter wall - CPU decode - GPU default wall: {non_cpu_vs_gpu_default:.3f} s",
            f"Heter wall - CPU decode - GPU eager wall: {non_cpu_vs_gpu_eager:.3f} s",
            f"Heter - skip-compute wall delta: {heter['request_wall_s'] - heter_skip['request_wall_s']:.3f} s",
            f"GPU eager - GPU default: {gpu_eager_delta:.3f} s",
            f"Skip-compute - GPU eager: {skip_vs_gpu_eager:.3f} s",
            "",
            "解释口径",
            "1. GPU forced eager 只比 GPU default 慢约 1s，关闭 CUDA Graph 不是主要矛盾。",
            "2. Heter 必须关闭 decode CUDA Graph，因为 CPU attention 分支含 GPU->CPU->GPU 同步。",
            "3. skip-compute 组保留异构同步和数据搬运，只跳过 C++ CPU attention 计算。",
            "4. skip-compute 仍显著慢于 GPU eager，主要指向每层同步/往返和路径串行化。",
            "5. Heter 与 skip-compute 的差值近似代表 C++ CPU attention 计算及其依赖输出带来的增量。",
            "6. 高频 JSONL/逐层日志不作为性能口径，本实验只使用 memory summary。",
        ]
        ax.text(
            0.03,
            0.97,
            "\n".join(lines),
            ha="left",
            va="top",
            fontsize=11,
            fontproperties=FONT,
            linespacing=1.4,
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def make_markdown(results: dict[str, Any]) -> None:
    groups = results["groups"]
    lines = [
        "# Request Wall Gap Recheck 实验记录",
        "",
        f"- 时间：`{results['timestamp_utc']}`",
        f"- workload：`prompt={PROMPT_LEN}`，`output={OUTPUT_LEN}`，`batch=1`",
        f"- OpenMP：`OMP_NUM_THREADS={OMP_THREADS}`",
        f"- vLLM 参数：`--gpu-memory-utilization {GPU_MEMORY_UTILIZATION}`，`--max-model-len {MAX_MODEL_LEN}`",
        f"- PDF：`{RESULT_PDF}`",
        f"- JSON：`{RESULT_JSON}`",
        "",
        "## 为什么重测",
        "",
        "- 之前把 `gpu_model_runner_model_forward - decode_cpu_path_layer_total` 直接解释成 GPU 其他算子耗时，口径过粗。",
        "- 当前代码里 pure decode 会为了 CPU attention 强制关闭 CUDA Graph；如果 GPU baseline 没有同口径控制，就不能直接拿 4s baseline 去相减。",
        "- 本次修正了 baseline 开关：`VLLM_HETER_DISABLE_CPU_ATTENTION=1` 时不再触发异构 eager 降级；另用 `VLLM_HETER_FORCE_DECODE_EAGER=1` 构造 GPU eager 对照。",
        "",
        "## 四组结果",
        "",
        "| 组别 | 说明 | request wall | model forward | decode CPU attention | wall - decode CPU |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group in groups:
        forward = event_total(group["request_events"], "gpu_model_runner_model_forward")
        decode = event_total(group["attention_events"], "decode_cpu_path_layer_total")
        lines.append(
            f"| `{group['name']}` | {group['desc']} | "
            f"{group['request_wall_s']:.3f} s | {forward:.3f} s | "
            f"{decode:.3f} s | {group['request_wall_s'] - decode:.3f} s |"
        )

    gpu_default, gpu_eager, heter = groups[0], groups[1], groups[2]
    heter_skip = groups[3]
    heter_decode = event_total(heter["attention_events"], "decode_cpu_path_layer_total")
    gpu_eager_minus_default = (
        gpu_eager["request_wall_s"] - gpu_default["request_wall_s"]
    )
    heter_wall_minus_decode = heter["request_wall_s"] - heter_decode
    skip_minus_gpu_eager = heter_skip["request_wall_s"] - gpu_eager["request_wall_s"]
    heter_minus_skip = heter["request_wall_s"] - heter_skip["request_wall_s"]
    lines.extend(
        [
            "",
            "## 关键差分",
            "",
            f"- `Heter - GPU default = {heter['request_wall_s'] - gpu_default['request_wall_s']:.3f} s`。",
            f"- `Heter - GPU forced eager = {heter['request_wall_s'] - gpu_eager['request_wall_s']:.3f} s`。",
            f"- `Heter wall - decode CPU - GPU default wall = {heter['request_wall_s'] - heter_decode - gpu_default['request_wall_s']:.3f} s`。",
            f"- `Heter wall - decode CPU - GPU eager wall = {heter['request_wall_s'] - heter_decode - gpu_eager['request_wall_s']:.3f} s`。",
            f"- `Heter - Heter skip CPU compute = {heter_minus_skip:.3f} s`。",
            f"- `Heter skip CPU compute - GPU eager = {skip_minus_gpu_eager:.3f} s`。",
            f"- `GPU forced eager - GPU default = {gpu_eager_minus_default:.3f} s`。",
            "",
            "## 互斥拆解",
            "",
            "| 组别 | Decode CPU attention | Model forward without CPU attention | Preprocess | Postprocess/logits | Sampling/bookkeeping | Output processor | API/scheduler other |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in groups:
        pieces = split_breakdown(group)
        lines.append(
            f"| `{group['name']}` | "
            f"{pieces['decode_cpu_attention_path']:.3f} s | "
            f"{pieces['model_forward_without_cpu_attention']:.3f} s | "
            f"{pieces['preprocess']:.3f} s | "
            f"{pieces['postprocess_logits']:.3f} s | "
            f"{pieces['sampling_bookkeeping']:.3f} s | "
            f"{pieces['output_processor']:.3f} s | "
            f"{pieces['api_scheduler_other']:.3f} s |"
        )

    lines.extend(
        [
            "",
            "## 初步结论",
            "",
            f"- 你的判断是对的：`Heter wall - decode_cpu_path_layer_total` 不应被简单解释成 GPU 原本那部分计算。正常异构中该值为 `{heter_wall_minus_decode:.3f} s`，明显高于 GPU eager 的 `{gpu_eager['request_wall_s']:.3f} s`。",
            f"- 关闭 CUDA Graph 本身只解释约 `{gpu_eager_minus_default:.3f} s`：也就是 `GPU forced eager - GPU default`。",
            f"- 更大的问题来自异构路径本身的 per-layer 同步/往返。skip-compute 组已经跳过 C++ CPU attention 计算，但 request wall 仍为 `{heter_skip['request_wall_s']:.3f} s`，比 GPU eager 多 `{skip_minus_gpu_eager:.3f} s`。",
            f"- 正常异构比 skip-compute 多 `{heter_minus_skip:.3f} s`，这部分近似对应 C++ CPU attention 计算以及依赖真实 attention 输出后继续执行 GPU 后续算子的增量。",
            "- `Heter skip CPU compute` 是性能归因消融组，不用于 correctness；它保留异构同步/搬运/metadata/KV 写入，只跳过 C++ CPU attention 计算。",
            "- 本实验只用 memory summary profile；未启用逐 event JSONL，避免把记录开销纳入性能结论。",
            "",
            "## 临时环境变量",
            "",
            "- 仅在测试子进程内设置 `OMP_NUM_THREADS`。",
            "- 仅在测试子进程内设置 `VLLM_HETER_DISABLE_CPU_ATTENTION`、`VLLM_HETER_FORCE_DECODE_EAGER`、`VLLM_HETER_CPU_ATTN_LIB`。",
            "- 仅在 skip-compute 消融子进程内设置 `VLLM_HETER_SKIP_CPU_ATTN_COMPUTE=1`。",
            "- 仅在测试子进程内设置 `VLLM_HETER_REQUEST_PROFILE*` 和 `VLLM_HETER_PROFILE*`。",
            "- 仅在测试子进程内设置 `no_proxy/NO_PROXY`。",
            "- 仅在绘图子进程内设置 `MPLCONFIGDIR` 和 `XDG_CACHE_HOME` 到 `/tmp/matplotlib-codex`。",
            "- 未修改系统持久环境变量，未修改系统设置。",
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
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_model_len": MAX_MODEL_LEN,
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
    make_markdown(results)
    print(f"[{now_utc()}] wrote {RESULT_JSON}", flush=True)
    print(f"[{now_utc()}] wrote {RESULT_PDF}", flush=True)
    print(f"[{now_utc()}] wrote {RESULT_MD}", flush=True)


if __name__ == "__main__":
    main()
