#!/usr/bin/env python3
"""GPU-only vs GPU-CPU heter throughput benchmark."""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, fontManager
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "CPU_GPU_Heter_test" / "gpu_vs_heter_throughput_20260528"
PYTHON = Path("/home/zb/tool/miniconda3/envs/vLLM_GPU/bin/python")
VLLM = Path("/home/zb/tool/miniconda3/envs/vLLM_GPU/bin/vllm")
MODEL = Path("/home/zb/data/model_data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")
CPU_ATTN_LIB = ROOT / "vllm" / "libs" / "libvllm_heter_cpu_attn.so"

PORT = 8036
PROMPT_LENS = [2000, 4000, 8000]
OUTPUT_LENS = [200, 400, 600]
GPU_MEMORY_UTILIZATION = "0.25"
MAX_MODEL_LEN = "10000"
REQUEST_TIMEOUT_S = 1200.0
SERVER_READY_TIMEOUT_S = 700.0
CASE_MAX_ATTEMPTS = 3
CASE_RETRY_SLEEP_S = 20.0
HETER_OMP_THREADS = 64
HETER_CPU_AFFINITY = "0-63"

RESULT_JSON = OUT_DIR / "gpu_vs_heter_throughput_results.json"
RESULT_CSV = OUT_DIR / "gpu_vs_heter_throughput_raw.csv"
RESULT_PDF = OUT_DIR / "gpu_vs_heter_throughput.pdf"
RESULT_MD = OUT_DIR / "gpu_vs_heter_throughput_record.md"
ITERATION_APPEND_MD = ROOT / "CPU_GPU_Heter_test" / "迭代记录.md"

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


def now_local_title() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def make_prompt(tokenizer: Any, token_count: int, case_tag: str) -> str:
    seed = (
        f"Case {case_tag}. This deterministic benchmark prompt contains "
        "repeated factual clauses about heterogeneous inference scheduling, "
        "GPU kernels, CPU attention, and request throughput. "
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


def wait_ready(port: int, timeout_s: float, log_path: Path,
               proc: subprocess.Popen[Any]) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = ""
            if log_path.exists():
                tail = "\n".join(log_path.read_text(errors="ignore").splitlines()[-120:])
            raise RuntimeError(
                f"vLLM server exited before ready with code {proc.returncode}\n{tail}"
            )
        try:
            with opener.open(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2)
    tail = ""
    if log_path.exists():
        tail = "\n".join(log_path.read_text(errors="ignore").splitlines()[-120:])
    raise RuntimeError(f"vLLM server not ready: {last_error}\n{tail}")


def terminate_process(proc: subprocess.Popen[Any]) -> None:
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


def launch_server(mode: str, log_path: Path) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "no_proxy": "localhost,127.0.0.1",
            "NO_PROXY": "localhost,127.0.0.1",
        }
    )
    if mode == "gpu":
        env["VLLM_HETER_DISABLE_CPU_ATTENTION"] = "1"
        env.pop("VLLM_HETER_CPU_ATTN_LIB", None)
    elif mode == "heter":
        env.pop("VLLM_HETER_DISABLE_CPU_ATTENTION", None)
        env["VLLM_HETER_CPU_ATTN_LIB"] = str(CPU_ATTN_LIB)
        env["OMP_NUM_THREADS"] = str(HETER_OMP_THREADS)
    else:
        raise ValueError(mode)

    cmd = [
        str(VLLM),
        "serve",
        str(MODEL),
        "--port",
        str(PORT),
        "--gpu-memory-utilization",
        GPU_MEMORY_UTILIZATION,
        "--max-model-len",
        MAX_MODEL_LEN,
        "--generation-config",
        "vllm",
        "--trust-remote-code",
    ]
    if mode == "heter":
        cmd = ["taskset", "-c", HETER_CPU_AFFINITY] + cmd
    with log_path.open("wb") as log_f:
        return subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )


def run_case(mode: str, prompt_len: int, output_len: int, prompt: str) -> dict[str, Any]:
    name = f"{mode}_p{prompt_len}_o{output_len}"
    last_error: Exception | None = None
    for attempt in range(1, CASE_MAX_ATTEMPTS + 1):
        suffix = "" if attempt == 1 else f"_attempt{attempt}"
        log_path = OUT_DIR / f"{name}_server{suffix}.log"
        attempt_msg = f" attempt {attempt}/{CASE_MAX_ATTEMPTS}"
        print(f"[{now_utc()}] start {name}{attempt_msg}", flush=True)
        proc = launch_server(mode, log_path)
        try:
            wait_ready(PORT, SERVER_READY_TIMEOUT_S, log_path, proc)
            payload = {
                "model": str(MODEL),
                "prompt": prompt,
                "max_tokens": output_len,
                "temperature": 0.0,
                "ignore_eos": True,
            }
            start = time.perf_counter()
            response = http_json(
                f"http://127.0.0.1:{PORT}/v1/completions",
                payload,
                timeout_s=REQUEST_TIMEOUT_S,
            )
            wall_s = time.perf_counter() - start
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= CASE_MAX_ATTEMPTS:
                raise
            print(
                f"[{now_utc()}] retry {name} after error: {exc}",
                flush=True,
            )
            time.sleep(CASE_RETRY_SLEEP_S)
        finally:
            terminate_process(proc)
    else:
        raise RuntimeError(f"{name} failed") from last_error

    usage = response.get("usage", {})
    completion_tokens = int(usage.get("completion_tokens", output_len))
    throughput = completion_tokens / wall_s if wall_s > 0 else 0.0
    result = {
        "mode": mode,
        "prompt_len_target": prompt_len,
        "output_len_target": output_len,
        "request_wall_s": wall_s,
        "completion_tokens_per_s": throughput,
        "usage": usage,
        "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
        "server_log": str(log_path),
        "server_start_attempt": attempt,
        "omp_threads": HETER_OMP_THREADS if mode == "heter" else None,
        "cpu_affinity": HETER_CPU_AFFINITY if mode == "heter" else None,
    }
    print(
        f"[{now_utc()}] done {name}: wall={wall_s:.3f}s "
        f"tok/s={throughput:.3f}",
        flush=True,
    )
    return result


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "mode",
        "prompt_len_target",
        "output_len_target",
        "request_wall_s",
        "completion_tokens_per_s",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "finish_reason",
        "server_log",
    ]
    with RESULT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            usage = row.get("usage", {})
            writer.writerow(
                {
                    "mode": row["mode"],
                    "prompt_len_target": row["prompt_len_target"],
                    "output_len_target": row["output_len_target"],
                    "request_wall_s": row["request_wall_s"],
                    "completion_tokens_per_s": row["completion_tokens_per_s"],
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "finish_reason": row.get("finish_reason"),
                    "server_log": row.get("server_log"),
                }
            )


def make_pdf(rows: list[dict[str, Any]]) -> None:
    data = {
        (row["mode"], int(row["prompt_len_target"]), int(row["output_len_target"])): row
        for row in rows
    }
    colors = {"gpu": "#2563eb", "heter": "#dc2626"}
    labels = {"gpu": "GPU-only", "heter": "GPU-CPU heter"}

    with PdfPages(RESULT_PDF) as pdf:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
        x = range(len(PROMPT_LENS))
        width = 0.35
        for ax, output_len in zip(axes, OUTPUT_LENS):
            gpu_vals = [
                data[("gpu", prompt_len, output_len)]["completion_tokens_per_s"]
                for prompt_len in PROMPT_LENS
            ]
            heter_vals = [
                data[("heter", prompt_len, output_len)]["completion_tokens_per_s"]
                for prompt_len in PROMPT_LENS
            ]
            ax.bar(
                [i - width / 2 for i in x],
                gpu_vals,
                width,
                label=labels["gpu"],
                color=colors["gpu"],
            )
            ax.bar(
                [i + width / 2 for i in x],
                heter_vals,
                width,
                label=labels["heter"],
                color=colors["heter"],
            )
            ax.set_title(f"Output length = {output_len}", fontproperties=FONT)
            ax.set_xticks(list(x), [f"{p // 1000}K" for p in PROMPT_LENS])
            ax.set_xlabel("Prompt length", fontproperties=FONT)
            ax.grid(axis="y", alpha=0.25)
            for i, (g, h) in enumerate(zip(gpu_vals, heter_vals)):
                ax.text(i - width / 2, g, f"{g:.1f}", ha="center", va="bottom",
                        fontsize=8)
                ax.text(i + width / 2, h, f"{h:.1f}", ha="center", va="bottom",
                        fontsize=8)
        axes[0].set_ylabel("Throughput (completion tokens/s)", fontproperties=FONT)
        axes[-1].legend(prop=FONT, loc="upper right")
        fig.suptitle(
            "End-to-End Throughput: GPU-only vs GPU-CPU Heterogeneous",
            fontproperties=FONT,
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 6))
        cases = []
        ratios = []
        for prompt_len in PROMPT_LENS:
            for output_len in OUTPUT_LENS:
                gpu = data[("gpu", prompt_len, output_len)]["completion_tokens_per_s"]
                heter = data[("heter", prompt_len, output_len)]["completion_tokens_per_s"]
                cases.append(f"P{prompt_len // 1000}K/O{output_len}")
                ratios.append(heter / gpu * 100.0 if gpu > 0 else 0.0)
        ax.bar(cases, ratios, color="#7c3aed")
        ax.set_ylabel("Heterogeneous / GPU-only throughput (%)", fontproperties=FONT)
        ax.set_xlabel("Workload", fontproperties=FONT)
        ax.set_title("Relative Throughput", fontproperties=FONT)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=35)
        for i, value in enumerate(ratios):
            ax.text(i, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("off")
        lines = [
            "Experiment details",
            "",
            "Metric: completion_tokens / request_wall_time.",
            "Each workload is run with a fresh vLLM server.",
            "Prefix caching is not disabled, but each server handles only one measured request.",
            "This preserves default vLLM behavior while avoiding repeated-prompt cache hits.",
            f"Heterogeneous mode: OMP_NUM_THREADS={HETER_OMP_THREADS}, taskset -c {HETER_CPU_AFFINITY}.",
            f"GPU memory utilization: {GPU_MEMORY_UTILIZATION}",
            f"Max model length: {MAX_MODEL_LEN}",
            "Batch size: 1",
        ]
        ax.text(
            0.02,
            0.95,
            "\n".join(lines),
            va="top",
            ha="left",
            fontsize=12,
            fontproperties=FONT,
            linespacing=1.5,
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def make_markdown(rows: list[dict[str, Any]]) -> None:
    data = {
        (row["mode"], int(row["prompt_len_target"]), int(row["output_len_target"])): row
        for row in rows
    }
    lines = [
        "# GPU-only vs GPU-CPU Heter Throughput",
        "",
        f"- Time: `{now_utc()}`",
        f"- Model: `{MODEL}`",
        f"- Prompt lengths: `{PROMPT_LENS}`",
        f"- Output lengths: `{OUTPUT_LENS}`",
        "- Batch size: `1`",
        f"- PDF: `{RESULT_PDF}`",
        f"- JSON: `{RESULT_JSON}`",
        f"- CSV: `{RESULT_CSV}`",
        "",
        "## Method",
        "",
        "- Metric: `completion_tokens / request_wall_time`.",
        "- GPU-only uses `VLLM_HETER_DISABLE_CPU_ATTENTION=1` in the server subprocess.",
        "- Heter uses `VLLM_HETER_CPU_ATTN_LIB` in the server subprocess.",
        f"- Heter uses `OMP_NUM_THREADS={HETER_OMP_THREADS}` and `taskset -c {HETER_CPU_AFFINITY}`.",
        "- Prefix caching is not disabled, but each workload is measured in a fresh server with one measured request.",
        "- No persistent environment variables or system settings were changed.",
        "",
        "## Results",
        "",
        "| Prompt | Output | GPU-only tok/s | Heter tok/s | Heter/GPU | GPU wall s | Heter wall s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for prompt_len in PROMPT_LENS:
        for output_len in OUTPUT_LENS:
            gpu = data[("gpu", prompt_len, output_len)]
            heter = data[("heter", prompt_len, output_len)]
            gpu_tps = float(gpu["completion_tokens_per_s"])
            heter_tps = float(heter["completion_tokens_per_s"])
            ratio = heter_tps / gpu_tps * 100.0 if gpu_tps > 0 else 0.0
            lines.append(
                f"| {prompt_len} | {output_len} | {gpu_tps:.3f} | "
                f"{heter_tps:.3f} | {ratio:.2f}% | "
                f"{gpu['request_wall_s']:.3f} | {heter['request_wall_s']:.3f} |"
            )
    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_iteration_record() -> None:
    lines = [
        "",
        f"# codex update {now_local_title()}",
        "",
        "本次补充 GPU-only 与 GPU-CPU 异构系统端到端吞吐对比实验。",
        "",
        f"- 实验目录：`{OUT_DIR}`",
        f"- prompt：`{PROMPT_LENS}`",
        f"- output：`{OUTPUT_LENS}`",
        "- batch size：`1`",
        "- 吞吐口径：`completion_tokens / request_wall_time`。",
        f"- 异构服务子进程设置：`OMP_NUM_THREADS={HETER_OMP_THREADS}`。",
        f"- 异构服务子进程显式绑核：`taskset -c {HETER_CPU_AFFINITY}`。",
        "- 未禁用 prefix caching；但每个 workload 使用独立 vLLM server，只发送一次正式请求，避免重复 prompt cache hit 污染。",
        "- PDF 中所有文字均为英文。",
        "",
        "输出文件：",
        "",
        f"- `{RESULT_JSON}`",
        f"- `{RESULT_CSV}`",
        f"- `{RESULT_PDF}`",
        f"- `{RESULT_MD}`",
        "",
        "环境变量记录：",
        "",
        "- GPU-only server 子进程临时设置 `VLLM_HETER_DISABLE_CPU_ATTENTION=1`。",
        "- Heter server 子进程临时设置 `VLLM_HETER_CPU_ATTN_LIB`。",
        f"- Heter server 子进程临时设置 `OMP_NUM_THREADS={HETER_OMP_THREADS}`。",
        f"- Heter server 子进程通过 `taskset -c {HETER_CPU_AFFINITY}` 显式绑定 CPU affinity。",
        "- 所有请求子进程临时设置 `no_proxy/NO_PROXY`。",
        "- 未修改系统持久环境变量或系统设置。",
    ]
    with ITERATION_APPEND_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def load_existing_rows() -> list[dict[str, Any]]:
    if not RESULT_JSON.exists():
        return []
    try:
        payload = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def row_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["mode"]),
        int(row["prompt_len_target"]),
        int(row["output_len_target"]),
    )


def keep_existing_row(row: dict[str, Any]) -> bool:
    if row.get("mode") != "heter":
        return True
    return (
        int(row.get("omp_threads", -1)) == HETER_OMP_THREADS
        and str(row.get("cpu_affinity")) == HETER_CPU_AFFINITY
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    prompts = {
        prompt_len: make_prompt(tokenizer, prompt_len, f"P{prompt_len}")
        for prompt_len in PROMPT_LENS
    }
    rows: list[dict[str, Any]] = [
        row for row in load_existing_rows() if keep_existing_row(row)
    ]
    completed = {row_key(row) for row in rows}
    for mode in ["gpu", "heter"]:
        for prompt_len in PROMPT_LENS:
            for output_len in OUTPUT_LENS:
                key = (mode, prompt_len, output_len)
                if key in completed:
                    print(f"[{now_utc()}] skip completed {key}", flush=True)
                    continue
                row = run_case(mode, prompt_len, output_len, prompts[prompt_len])
                rows.append(row)
                completed.add(key)
                partial = {
                    "timestamp_utc": now_utc(),
                    "rows": rows,
                    "note": "partial result",
                }
                RESULT_JSON.write_text(
                    json.dumps(partial, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                write_csv(rows)
    results = {
        "timestamp_utc": now_utc(),
        "model": str(MODEL),
        "prompt_lens": PROMPT_LENS,
        "output_lens": OUTPUT_LENS,
        "batch_size": 1,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_model_len": MAX_MODEL_LEN,
        "metric": "completion_tokens / request_wall_time",
        "prefix_cache_policy": (
            "default vLLM behavior; each measured workload uses a fresh server"
        ),
        "rows": rows,
    }
    RESULT_JSON.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(rows)
    make_pdf(rows)
    make_markdown(rows)
    append_iteration_record()
    print(f"[{now_utc()}] wrote {RESULT_JSON}", flush=True)
    print(f"[{now_utc()}] wrote {RESULT_CSV}", flush=True)
    print(f"[{now_utc()}] wrote {RESULT_PDF}", flush=True)
    print(f"[{now_utc()}] wrote {RESULT_MD}", flush=True)


if __name__ == "__main__":
    main()
