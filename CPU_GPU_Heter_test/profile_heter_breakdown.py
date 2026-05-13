"""
CPU-GPU 异构路径细粒度 profile。

输出：
- heter_profile_events.jsonl: attention 内部 profile 原始事件
- heter_profile_results.json: 汇总结果
- heter_profile_breakdown.svg: 可视化图
- heter_profile_记录.md: 中文记录
"""

import json
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
from transformers import AutoTokenizer

MODEL_PATH = "/home/zb/data/model_data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
VLLM_BIN = "/home/zb/tool/miniconda3/envs/vLLM_GPU/bin/vllm"
PORT = 8001
BASE_URL = f"http://localhost:{PORT}/v1"
OUT_DIR = Path(__file__).parent
PROFILE_EVENTS = OUT_DIR / "heter_profile_events.jsonl"
PROFILE_JSON = OUT_DIR / "heter_profile_results.json"
PROFILE_SVG = OUT_DIR / "heter_profile_breakdown.svg"
PROFILE_CPU_SVG = OUT_DIR / "heter_profile_cpu_attention_breakdown.svg"
PROFILE_MD = OUT_DIR / "heter_profile_记录.md"
LOG_PATH = OUT_DIR / "heter_profile_server.log"

CONTEXT_LENGTHS = [128, 512]
BATCH_SIZE = 1
MAX_TOKENS = 8
TIMEOUT_START = 600
TIMEOUT_REQ = 900

os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

BREAKDOWN_EVENTS = [
    ("gpu_attention_forward", "GPU attention / prefill"),
    ("prefill_kv_d2h_enqueue", "Prefill KV D2H enqueue"),
    ("decode_wait_prefill_d2h", "Wait prefill D2H"),
    ("decode_kv_d2h", "Decode KV D2H"),
    ("decode_kv_cache_write_total", "Decode KV cache write"),
    ("decode_query_d2h", "Query D2H"),
    ("decode_output_cpu_alloc", "CPU output alloc"),
    ("decode_metadata_d2h", "Metadata D2H"),
    ("cpu_attention_python_total", "CPU attention Python"),
    ("decode_output_h2d", "Output H2D"),
]

CPU_ATTENTION_EVENTS = [
    ("cpu_attention_collect_kv", "Collect/concat paged KV"),
    ("cpu_attention_repeat_kv", "GQA repeat KV"),
    ("cpu_attention_qk", "QK matmul"),
    ("cpu_attention_softmax", "Softmax"),
    ("cpu_attention_pv", "PV matmul"),
]


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


def start_server() -> subprocess.Popen:
    env = os.environ.copy()
    env["no_proxy"] = "localhost,127.0.0.1"
    env["NO_PROXY"] = "localhost,127.0.0.1"
    env["VLLM_HETER_PROFILE"] = "1"
    env["VLLM_HETER_PROFILE_PATH"] = str(PROFILE_EVENTS)
    env.pop("VLLM_HETER_DISABLE_CPU_ATTENTION", None)
    cmd = [
        VLLM_BIN,
        "serve",
        MODEL_PATH,
        "--port",
        str(PORT),
        "--max-num-seqs",
        str(BATCH_SIZE),
        "--gpu-memory-utilization",
        "0.30",
        "--generation-config",
        "vllm",
    ]
    log_fh = open(LOG_PATH, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )
    log_fh.close()
    wait_for_server()
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def build_prompt(tokenizer, target_context_len: int) -> str:
    prefix = "Profile this deterministic heterogeneous inference request. "
    filler = (
        "GPU prefill writes KV cache. CPU decode attention reads copied paged "
        "KV cache. The profiler records transfer, metadata, attention, and "
        "output stages. "
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
    wall_s = time.perf_counter() - start
    resp.raise_for_status()
    data = resp.json()
    return {
        "wall_s": wall_s,
        "usage": data["usage"],
        "text": data["choices"][0]["text"],
    }


def read_events() -> list[dict]:
    if not PROFILE_EVENTS.exists():
        return []
    events = []
    for line in PROFILE_EVENTS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def summarize_events(events: list[dict]) -> dict:
    by_event = defaultdict(lambda: {"count": 0, "total_s": 0.0, "max_s": 0.0})
    for event in events:
        name = event["event"]
        elapsed = float(event["elapsed_s"])
        by_event[name]["count"] += 1
        by_event[name]["total_s"] += elapsed
        by_event[name]["max_s"] = max(by_event[name]["max_s"], elapsed)

    summary = {}
    for name, item in sorted(by_event.items()):
        count = item["count"]
        summary[name] = {
            "count": count,
            "total_s": item["total_s"],
            "avg_ms": item["total_s"] / count * 1000.0 if count else 0.0,
            "max_ms": item["max_s"] * 1000.0,
        }
    return summary


def svg_text(x, y, text, size=12, anchor="middle", weight="normal"):
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" '
        f'font-family="Arial, sans-serif" text-anchor="{anchor}" '
        f'font-weight="{weight}">{text}</text>'
    )


def make_svg(cases: list[dict]) -> None:
    width, height = 1120, 640
    margin_l, margin_t = 245, 68
    plot_w, row_h = 760, 38
    colors = [
        "#2563eb", "#7c3aed", "#0891b2", "#dc2626", "#ea580c",
        "#16a34a", "#64748b", "#ca8a04", "#111827", "#be123c",
    ]
    max_total = max(
        sum(case["breakdown_s"].values()) for case in cases
    )
    max_total = max(max_total, 1e-9)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 28, "Fine-grained Profile of CPU-GPU Heterogeneous Decode Path", size=18, weight="bold"),
        svg_text(width / 2, 50, f"Llama-3.1-8B-Instruct, batch={BATCH_SIZE}, max_new_tokens={MAX_TOKENS}", size=12),
    ]

    for i, case in enumerate(cases):
        y = margin_t + i * row_h * 1.8
        label = f"ctx={case['context_len_target']} wall={case['wall_s']:.2f}s"
        parts.append(svg_text(margin_l - 12, y + 18, label, size=12, anchor="end"))
        x = margin_l
        total = sum(case["breakdown_s"].values())
        for idx, (event, _label) in enumerate(BREAKDOWN_EVENTS):
            value = case["breakdown_s"].get(event, 0.0)
            if value <= 0:
                continue
            w = value / max_total * plot_w
            parts.append(
                f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="26" fill="{colors[idx]}"/>'
            )
            if w > 42:
                parts.append(svg_text(x + w / 2, y + 18, f"{value:.1f}s", size=10, anchor="middle"))
            x += w
        parts.append(svg_text(margin_l + total / max_total * plot_w + 8, y + 18, f"{total:.2f}s", size=11, anchor="start"))

    axis_y = margin_t + len(cases) * row_h * 1.8 + 12
    for tick in range(6):
        value = max_total * tick / 5
        x = margin_l + value / max_total * plot_w
        parts.append(f'<line x1="{x:.1f}" y1="{axis_y - 6}" x2="{x:.1f}" y2="{axis_y}" stroke="#222"/>')
        parts.append(svg_text(x, axis_y + 18, f"{value:.1f}s", size=10))
    parts.append(f'<line x1="{margin_l}" y1="{axis_y}" x2="{margin_l + plot_w}" y2="{axis_y}" stroke="#222"/>')
    parts.append(svg_text(margin_l + plot_w / 2, axis_y + 42, "Accumulated instrumented time", size=12))

    legend_x, legend_y = 60, axis_y + 78
    for idx, (event, label) in enumerate(BREAKDOWN_EVENTS):
        col = idx % 2
        row = idx // 2
        x = legend_x + col * 430
        y = legend_y + row * 24
        parts.append(f'<rect x="{x}" y="{y}" width="14" height="14" fill="{colors[idx]}"/>')
        parts.append(svg_text(x + 22, y + 12, label, size=11, anchor="start"))

    parts.append("</svg>")
    PROFILE_SVG.write_text("\n".join(parts), encoding="utf-8")


def make_cpu_attention_svg(cases: list[dict]) -> None:
    width, height = 1040, 470
    margin_l, margin_t = 230, 70
    plot_w, row_h = 700, 42
    colors = ["#1d4ed8", "#7c3aed", "#dc2626", "#16a34a", "#ea580c"]
    max_total = max(
        sum(case["cpu_attention_breakdown_s"].values()) for case in cases
    )
    max_total = max(max_total, 1e-9)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 28, "CPU Attention Python Fallback Internal Breakdown", size=18, weight="bold"),
        svg_text(width / 2, 50, "Accumulated over all profiled decode layers/tokens", size=12),
    ]

    for i, case in enumerate(cases):
        y = margin_t + i * row_h * 1.55
        label = f"ctx={case['context_len_target']}"
        parts.append(svg_text(margin_l - 12, y + 18, label, size=12, anchor="end"))
        x = margin_l
        total = sum(case["cpu_attention_breakdown_s"].values())
        for idx, (event, _label) in enumerate(CPU_ATTENTION_EVENTS):
            value = case["cpu_attention_breakdown_s"].get(event, 0.0)
            if value <= 0:
                continue
            w = value / max_total * plot_w
            parts.append(
                f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" height="28" fill="{colors[idx]}"/>'
            )
            if w > 48:
                parts.append(svg_text(x + w / 2, y + 19, f"{value:.2f}s", size=10))
            x += w
        parts.append(svg_text(margin_l + total / max_total * plot_w + 8, y + 19, f"{total:.2f}s", size=11, anchor="start"))

    axis_y = margin_t + len(cases) * row_h * 1.55 + 10
    parts.append(f'<line x1="{margin_l}" y1="{axis_y}" x2="{margin_l + plot_w}" y2="{axis_y}" stroke="#222"/>')
    for tick in range(6):
        value = max_total * tick / 5
        x = margin_l + value / max_total * plot_w
        parts.append(f'<line x1="{x:.1f}" y1="{axis_y - 6}" x2="{x:.1f}" y2="{axis_y}" stroke="#222"/>')
        parts.append(svg_text(x, axis_y + 18, f"{value:.1f}s", size=10))
    parts.append(svg_text(margin_l + plot_w / 2, axis_y + 42, "Accumulated CPU attention time", size=12))

    legend_x, legend_y = 100, axis_y + 78
    for idx, (_event, label) in enumerate(CPU_ATTENTION_EVENTS):
        x = legend_x + (idx % 2) * 380
        y = legend_y + (idx // 2) * 24
        parts.append(f'<rect x="{x}" y="{y}" width="14" height="14" fill="{colors[idx]}"/>')
        parts.append(svg_text(x + 22, y + 12, label, size=11, anchor="start"))

    parts.append("</svg>")
    PROFILE_CPU_SVG.write_text("\n".join(parts), encoding="utf-8")


def write_markdown(report: dict) -> None:
    lines = [
        "# 异构系统细粒度 Profile 记录",
        "",
        f"实验时间：{report['timestamp']}",
        "",
        "## 配置",
        "",
        f"- 模型：`{MODEL_PATH}`",
        f"- batch size：`{BATCH_SIZE}`",
        f"- max_tokens：`{MAX_TOKENS}`",
        f"- context length：`{', '.join(str(x) for x in CONTEXT_LENGTHS)}`",
        "- profile 开关：`VLLM_HETER_PROFILE=1`",
        f"- profile 原始事件：`{PROFILE_EVENTS}`",
        f"- 可视化图：`{PROFILE_SVG}`",
        f"- CPU attention 内部分解图：`{PROFILE_CPU_SVG}`",
        "",
        "## 端到端与主要阶段",
        "",
        "| Context | Wall time s | Completion tokens | Instrumented sum s | CPU attention Python s | Output H2D s | Decode KV D2H s | Metadata D2H s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        b = case["breakdown_s"]
        lines.append(
            f"| {case['context_len_target']} | {case['wall_s']:.3f} | "
            f"{case['completion_tokens']} | {sum(b.values()):.3f} | "
            f"{b.get('cpu_attention_python_total', 0.0):.3f} | "
            f"{b.get('decode_output_h2d', 0.0):.3f} | "
            f"{b.get('decode_kv_d2h', 0.0):.3f} | "
            f"{b.get('decode_metadata_d2h', 0.0):.3f} |"
        )
    lines += [
        "",
        "## CPU Attention Python Fallback 内部分解",
        "",
        "| Context | CPU attention total s | Collect/concat KV s | Repeat KV s | QK s | Softmax s | PV s | Collect/total |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        cpu_b = case["cpu_attention_breakdown_s"]
        total = case["breakdown_s"].get("cpu_attention_python_total", 0.0)
        collect = cpu_b.get("cpu_attention_collect_kv", 0.0)
        lines.append(
            f"| {case['context_len_target']} | {total:.3f} | "
            f"{collect:.3f} | "
            f"{cpu_b.get('cpu_attention_repeat_kv', 0.0):.3f} | "
            f"{cpu_b.get('cpu_attention_qk', 0.0):.3f} | "
            f"{cpu_b.get('cpu_attention_softmax', 0.0):.3f} | "
            f"{cpu_b.get('cpu_attention_pv', 0.0):.3f} | "
            f"{collect / total * 100.0 if total else 0.0:.2f}% |"
        )
    lines += [
        "",
        "## 解释",
        "",
        "本 profile 的 `wall time` 是 API 端到端请求时间。`Instrumented sum` 是 attention 内部事件的累计和，包含逐层、逐 token 事件，因此用于定位热点，而不是严格等同于端到端 wall time。",
        "",
        "本次结果中 `cpu_attention_python_total` 占主导，并且其内部主要耗时来自 `cpu_attention_collect_kv`，即 Python fallback 每层每 token 都从 paged KV block 中逐块取出、`torch.cat` 拼接 K/V，再做 GQA repeat。真正的 QK、softmax、PV 计算占比很小。",
        "",
        "因此，当前几十秒开销不是由 prefill GPU 计算主导，也不是由 KV D2H/H2D 主导，而是由 Python fallback 的 paged KV 收集/拼接路径主导。",
        "",
    ]
    PROFILE_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--plot-only":
        report = json.loads(PROFILE_JSON.read_text(encoding="utf-8"))
        for case in report["cases"]:
            if "cpu_attention_breakdown_s" not in case:
                summary = case["summary"]
                case["cpu_attention_breakdown_s"] = {
                    event: summary.get(event, {}).get("total_s", 0.0)
                    for event, _ in CPU_ATTENTION_EVENTS
                }
        PROFILE_JSON.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        make_svg(report["cases"])
        make_cpu_attention_svg(report["cases"])
        write_markdown(report)
        print(f"图: {PROFILE_SVG}")
        print(f"CPU attention 图: {PROFILE_CPU_SVG}")
        print(f"记录: {PROFILE_MD}")
        return 0

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    proc = None
    cases = []
    try:
        proc = start_server()
        # Warmup 后清空事件，避免服务启动和首请求噪声进入正式 profile。
        warmup_prompt = build_prompt(tokenizer, CONTEXT_LENGTHS[0])
        send_completion(warmup_prompt)

        for context_len in CONTEXT_LENGTHS:
            PROFILE_EVENTS.write_text("", encoding="utf-8")
            prompt = build_prompt(tokenizer, context_len)
            response = send_completion(prompt)
            events = read_events()
            summary = summarize_events(events)
            breakdown_s = {
                event: summary.get(event, {}).get("total_s", 0.0)
                for event, _ in BREAKDOWN_EVENTS
            }
            cpu_attention_breakdown_s = {
                event: summary.get(event, {}).get("total_s", 0.0)
                for event, _ in CPU_ATTENTION_EVENTS
            }
            case = {
                "context_len_target": context_len,
                "wall_s": response["wall_s"],
                "prompt_tokens": response["usage"]["prompt_tokens"],
                "completion_tokens": response["usage"]["completion_tokens"],
                "total_tokens": response["usage"]["total_tokens"],
                "event_count": len(events),
                "summary": summary,
                "breakdown_s": breakdown_s,
                "cpu_attention_breakdown_s": cpu_attention_breakdown_s,
                "text": response["text"],
            }
            print(
                f"ctx={context_len} wall={case['wall_s']:.3f}s "
                f"events={case['event_count']} "
                f"cpu_attn={breakdown_s.get('cpu_attention_python_total', 0.0):.3f}s"
            )
            cases.append(case)
    finally:
        if proc is not None:
            stop_server(proc)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.gmtime()),
        "model": MODEL_PATH,
        "batch_size": BATCH_SIZE,
        "max_tokens": MAX_TOKENS,
        "context_lengths": CONTEXT_LENGTHS,
        "profile_events": str(PROFILE_EVENTS),
        "server_log": str(LOG_PATH),
        "cases": cases,
    }
    PROFILE_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    make_svg(cases)
    make_cpu_attention_svg(cases)
    write_markdown(report)
    print(f"结果: {PROFILE_JSON}")
    print(f"图: {PROFILE_SVG}")
    print(f"CPU attention 图: {PROFILE_CPU_SVG}")
    print(f"记录: {PROFILE_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
