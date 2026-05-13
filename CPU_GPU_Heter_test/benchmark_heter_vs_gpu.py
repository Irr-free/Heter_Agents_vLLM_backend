"""
吞吐基准：纯 GPU baseline vs CPU-GPU 异构 decode attention。

实验维度：
- context length: 128, 512, 1024
- batch size: 1, 2

说明：
- baseline 通过 VLLM_HETER_DISABLE_CPU_ATTENTION=1 临时禁用异构路径。
- heter 使用默认异构路径。
- 所有输出文件保存在 CPU_GPU_Heter_test 下。
"""

import json
import math
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from transformers import AutoTokenizer

MODEL_PATH = "/home/zb/data/model_data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
VLLM_BIN = "/home/zb/tool/miniconda3/envs/vLLM_GPU/bin/vllm"
PORT = 8001
BASE_URL = f"http://localhost:{PORT}/v1"
LOG_DIR = Path(__file__).parent
RESULT_JSON = LOG_DIR / "throughput_benchmark_results.json"
FIG_SVG = LOG_DIR / "throughput_benchmark_figure.svg"
CPP_ATTN_LIB = LOG_DIR.parent / "vllm/libs/libvllm_heter_cpu_attn.so"

CONTEXT_LENGTHS = [128, 512, 1024]
BATCH_SIZES = [1, 2]
WARMUP_CONTEXT_LEN = 128
WARMUP_BATCH_SIZE = 1
MAX_TOKENS = 8
MAX_MODEL_LEN = 2048
KV_CACHE_MEMORY_BYTES = 1 << 30
TIMEOUT_START = 600
TIMEOUT_REQ = 900

os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"


def wait_for_server(timeout: int = TIMEOUT_START) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{BASE_URL}/models", timeout=5)
            if resp.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("vLLM 服务启动超时")


def start_server(mode: str) -> tuple[subprocess.Popen, Path]:
    log_path = LOG_DIR / f"throughput_{mode}.log"
    env = os.environ.copy()
    env["no_proxy"] = "localhost,127.0.0.1"
    env["NO_PROXY"] = "localhost,127.0.0.1"
    if mode == "gpu":
        env["VLLM_HETER_DISABLE_CPU_ATTENTION"] = "1"
    else:
        env.pop("VLLM_HETER_DISABLE_CPU_ATTENTION", None)
        if CPP_ATTN_LIB.exists():
            env["VLLM_HETER_CPU_ATTN_LIB"] = str(CPP_ATTN_LIB)

    cmd = [
        VLLM_BIN,
        "serve",
        MODEL_PATH,
        "--port",
        str(PORT),
        "--max-num-seqs",
        str(max(BATCH_SIZES)),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--kv-cache-memory-bytes",
        str(KV_CACHE_MEMORY_BYTES),
        "--gpu-memory-utilization",
        "0.30",
        "--generation-config",
        "vllm",
    ]
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )
    log_fh.close()
    wait_for_server()
    return proc, log_path


def stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def build_prompt(tokenizer, target_context_len: int, request_id: int) -> str:
    prefix = (
        f"Request {request_id}. Summarize the following deterministic notes in "
        "one short sentence. "
    )
    filler = (
        "GPU prefill writes KV cache. CPU decode attention reads the copied "
        "paged KV cache. The benchmark measures throughput and latency. "
    )
    prompt = prefix
    while len(tokenizer.encode(prompt, add_special_tokens=False)) < target_context_len:
        prompt += filler
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)[:target_context_len]
    return tokenizer.decode(token_ids)


def send_completion(prompt: str) -> dict:
    payload = {
        "model": MODEL_PATH,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
    }
    start = time.perf_counter()
    resp = requests.post(f"{BASE_URL}/completions", json=payload, timeout=TIMEOUT_REQ)
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    data = resp.json()
    return {
        "latency_s": elapsed,
        "usage": data["usage"],
        "text": data["choices"][0]["text"],
    }


def run_case(mode: str, tokenizer, context_len: int, batch_size: int) -> dict:
    prompts = [
        build_prompt(tokenizer, context_len, request_id=i)
        for i in range(batch_size)
    ]

    wall_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        responses = list(executor.map(send_completion, prompts))
    wall_s = time.perf_counter() - wall_start

    prompt_tokens = sum(item["usage"]["prompt_tokens"] for item in responses)
    completion_tokens = sum(item["usage"]["completion_tokens"] for item in responses)
    total_tokens = prompt_tokens + completion_tokens
    latencies = [item["latency_s"] for item in responses]
    return {
        "mode": mode,
        "context_len_target": context_len,
        "batch_size": batch_size,
        "max_tokens": MAX_TOKENS,
        "wall_s": wall_s,
        "request_latency_s_avg": sum(latencies) / len(latencies),
        "request_latency_s_max": max(latencies),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "decode_tok_per_s": completion_tokens / wall_s,
        "total_tok_per_s": total_tokens / wall_s,
        "responses": responses,
    }


def svg_text(x, y, text, size=12, anchor="middle", weight="normal"):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" '
        f'font-family="Arial, sans-serif" text-anchor="{anchor}" '
        f'font-weight="{weight}">{text}</text>'
    )


def make_svg(results: list[dict]) -> None:
    width, height = 1080, 760
    margin_l, margin_r, margin_t, margin_b = 88, 36, 72, 74
    panel_gap = 74
    row_gap = 72
    top_h = 300
    bottom_h = 210
    panel_w = (width - margin_l - margin_r - panel_gap) / 2
    colors = {"gpu": "#2b6cb0", "heter": "#c2410c"}
    labels = {"gpu": "GPU baseline", "heter": "CPU-GPU heter"}

    values = [item["decode_tok_per_s"] for item in results]
    min_y = 0.1
    max_y = 300.0
    log_min = math.log10(min_y)
    log_max = math.log10(max_y)

    def log_y(value: float, panel_h: float) -> float:
        value = max(value, min_y)
        ratio = (math.log10(value) - log_min) / (log_max - log_min)
        return panel_h - ratio * panel_h

    def find_result(batch_size: int, context_len: int, mode: str) -> dict:
        return next(
            r for r in results
            if r["batch_size"] == batch_size
            and r["context_len_target"] == context_len
            and r["mode"] == mode
        )

    ratio_max = 0.0
    for batch_size in BATCH_SIZES:
        for ctx in CONTEXT_LENGTHS:
            gpu = find_result(batch_size, ctx, "gpu")["decode_tok_per_s"]
            heter = find_result(batch_size, ctx, "heter")["decode_tok_per_s"]
            ratio_max = max(ratio_max, heter / gpu * 100.0)
    ratio_max = max(3.0, math.ceil(ratio_max * 1.15 * 10.0) / 10.0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 28,
                 "Decode Throughput: GPU Baseline vs. CPU-GPU Heterogeneous Path",
                 size=18, weight="bold"),
        svg_text(width / 2, 50,
                 f"Llama-3.1-8B-Instruct, max_new_tokens={MAX_TOKENS}; upper row uses log scale",
                 size=12),
    ]

    for panel_idx, batch_size in enumerate(BATCH_SIZES):
        x0 = margin_l + panel_idx * (panel_w + panel_gap)
        y0 = margin_t
        parts.append(f'<g transform="translate({x0},{y0})">')
        parts.append(
            f'<rect x="0" y="0" width="{panel_w}" height="{top_h}" '
            'fill="#ffffff" stroke="#222" stroke-width="1"/>'
        )
        for value in [0.1, 0.3, 1, 3, 10, 30, 100, 300]:
            y = log_y(value, top_h)
            parts.append(
                f'<line x1="0" y1="{y:.1f}" x2="{panel_w}" y2="{y:.1f}" '
                'stroke="#e5e7eb" stroke-width="1"/>'
            )
            parts.append(svg_text(-8, y + 4, f"{value:g}", size=10, anchor="end"))

        group_w = panel_w / len(CONTEXT_LENGTHS)
        bar_w = group_w * 0.28
        base_y = log_y(min_y, top_h)
        for i, ctx in enumerate(CONTEXT_LENGTHS):
            group_x = i * group_w + group_w / 2
            for j, mode in enumerate(["gpu", "heter"]):
                item = find_result(batch_size, ctx, mode)
                y = log_y(item["decode_tok_per_s"], top_h)
                h = base_y - y
                x = group_x + (j - 0.5) * bar_w - bar_w / 2
                parts.append(
                    f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                    f'height="{h:.1f}" fill="{colors[mode]}"/>'
                )
                parts.append(
                    svg_text(x + bar_w / 2, y - 4,
                             f'{item["decode_tok_per_s"]:.1f}',
                             size=9)
                )
            parts.append(svg_text(group_x, top_h + 22, str(ctx), size=11))

        parts.append(svg_text(panel_w / 2, -18, f"Batch size = {batch_size}",
                              size=14, weight="bold"))
        parts.append(svg_text(panel_w / 2, top_h + 48, "Context length (tokens)",
                              size=12))
        parts.append("</g>")

        y1 = margin_t + top_h + row_gap
        parts.append(f'<g transform="translate({x0},{y1})">')
        parts.append(
            f'<rect x="0" y="0" width="{panel_w}" height="{bottom_h}" '
            'fill="#ffffff" stroke="#222" stroke-width="1"/>'
        )
        for tick in range(6):
            value = ratio_max * tick / 5
            y = bottom_h - value / ratio_max * bottom_h
            parts.append(
                f'<line x1="0" y1="{y:.1f}" x2="{panel_w}" y2="{y:.1f}" '
                'stroke="#e5e7eb" stroke-width="1"/>'
            )
            parts.append(svg_text(-8, y + 4, f"{value:.1f}", size=10, anchor="end"))

        points = []
        for i, ctx in enumerate(CONTEXT_LENGTHS):
            group_x = i * group_w + group_w / 2
            gpu = find_result(batch_size, ctx, "gpu")["decode_tok_per_s"]
            heter = find_result(batch_size, ctx, "heter")["decode_tok_per_s"]
            ratio = heter / gpu * 100.0
            y = bottom_h - ratio / ratio_max * bottom_h
            points.append((group_x, y, ratio))
            parts.append(svg_text(group_x, bottom_h + 22, str(ctx), size=11))
        polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)
        parts.append(
            f'<polyline points="{polyline}" fill="none" stroke="#111827" '
            'stroke-width="1.8"/>'
        )
        for x, y, ratio in points:
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#111827"/>'
            )
            parts.append(svg_text(x, y - 8, f"{ratio:.2f}%", size=9))
        parts.append(svg_text(panel_w / 2, -18,
                              f"Heterogeneous throughput / GPU throughput, batch={batch_size}",
                              size=13, weight="bold"))
        parts.append(svg_text(panel_w / 2, bottom_h + 48,
                              "Context length (tokens)", size=12))
        parts.append("</g>")

    parts.append(
        f'<g transform="translate(24,{margin_t + top_h / 2}) rotate(-90)">'
        + svg_text(0, 0, "Decode throughput (tokens/s, log scale)", size=13)
        + "</g>"
    )
    parts.append(
        f'<g transform="translate(24,{margin_t + top_h + row_gap + bottom_h / 2}) rotate(-90)">'
        + svg_text(0, 0, "Relative throughput (%)", size=13)
        + "</g>"
    )

    legend_x = width - 285
    legend_y = 22
    for i, mode in enumerate(["gpu", "heter"]):
        y = legend_y + i * 22
        parts.append(
            f'<rect x="{legend_x}" y="{y}" width="14" height="14" '
            f'fill="{colors[mode]}"/>'
        )
        parts.append(svg_text(legend_x + 22, y + 12, labels[mode],
                              size=12, anchor="start"))

    parts.append("</svg>")
    FIG_SVG.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--plot-only":
        report = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
        make_svg(report["results"])
        print(f"图: {FIG_SVG}")
        return 0

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    all_results = []
    server_logs = {}

    for mode in ["gpu", "heter"]:
        proc = None
        try:
            proc, log_path = start_server(mode)
            server_logs[mode] = str(log_path)
            warmup = run_case(mode, tokenizer, WARMUP_CONTEXT_LEN, WARMUP_BATCH_SIZE)
            print(
                f"{mode} warmup ctx={WARMUP_CONTEXT_LEN} "
                f"batch={WARMUP_BATCH_SIZE} "
                f"decode_tok/s={warmup['decode_tok_per_s']:.3f} "
                f"wall={warmup['wall_s']:.3f}s"
            )
            for context_len in CONTEXT_LENGTHS:
                for batch_size in BATCH_SIZES:
                    result = run_case(mode, tokenizer, context_len, batch_size)
                    print(
                        f"{mode} ctx={context_len} batch={batch_size} "
                        f"decode_tok/s={result['decode_tok_per_s']:.3f} "
                        f"wall={result['wall_s']:.3f}s"
                    )
                    all_results.append(result)
        finally:
            if proc is not None:
                stop_server(proc)

    report = {
        "model": MODEL_PATH,
        "context_lengths": CONTEXT_LENGTHS,
        "batch_sizes": BATCH_SIZES,
        "warmup": {
            "context_len": WARMUP_CONTEXT_LEN,
            "batch_size": WARMUP_BATCH_SIZE,
            "included_in_results": False,
        },
        "max_tokens": MAX_TOKENS,
        "server_logs": server_logs,
        "results": all_results,
    }
    RESULT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_svg(all_results)
    print(f"结果: {RESULT_JSON}")
    print(f"图: {FIG_SVG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
