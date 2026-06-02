#!/usr/bin/env python3
"""Repeat the anomalous heter P2K/O200 point."""

from __future__ import annotations

import csv
import json
import statistics
import time
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from transformers import AutoTokenizer

import gpu_vs_heter_throughput_experiment as base


PROMPT_LEN = 2000
OUTPUT_LEN = 200
REPEATS = 5
OUT_DIR = base.OUT_DIR / "p2k_o200_anomaly_recheck"
RESULT_JSON = OUT_DIR / "p2k_o200_anomaly_recheck_results.json"
RESULT_CSV = OUT_DIR / "p2k_o200_anomaly_recheck_raw.csv"
RESULT_MD = OUT_DIR / "p2k_o200_anomaly_recheck_record.md"
RESULT_PDF = OUT_DIR / "p2k_o200_anomaly_recheck.pdf"


def now_utc() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())


def now_local_title() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def load_matrix_reference() -> dict[str, Any]:
    if not base.RESULT_JSON.exists():
        return {}
    data = json.loads(base.RESULT_JSON.read_text(encoding="utf-8"))
    refs: dict[str, Any] = {}
    for row in data.get("rows", []):
        if (int(row.get("prompt_len_target", -1)) == PROMPT_LEN
                and int(row.get("output_len_target", -1)) == OUTPUT_LEN):
            refs[str(row.get("mode"))] = row
    return refs


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "repeat",
        "request_wall_s",
        "completion_tokens_per_s",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "server_start_attempt",
        "server_log",
    ]
    with RESULT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            usage = row.get("usage", {})
            writer.writerow(
                {
                    "repeat": row["repeat"],
                    "request_wall_s": row["request_wall_s"],
                    "completion_tokens_per_s": row["completion_tokens_per_s"],
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "server_start_attempt": row["server_start_attempt"],
                    "server_log": row["server_log"],
                }
            )


def run_repeat(prompt: str, repeat: int) -> dict[str, Any]:
    name = f"heter_p{PROMPT_LEN}_o{OUTPUT_LEN}_repeat{repeat}"
    last_error: Exception | None = None
    for attempt in range(1, base.CASE_MAX_ATTEMPTS + 1):
        log_path = OUT_DIR / f"{name}_attempt{attempt}_server.log"
        print(
            f"[{now_utc()}] start {name} attempt {attempt}/"
            f"{base.CASE_MAX_ATTEMPTS}",
            flush=True,
        )
        proc = base.launch_server("heter", log_path)
        try:
            base.wait_ready(base.PORT, base.SERVER_READY_TIMEOUT_S, log_path, proc)
            payload = {
                "model": str(base.MODEL),
                "prompt": prompt,
                "max_tokens": OUTPUT_LEN,
                "temperature": 0.0,
                "ignore_eos": True,
            }
            start = time.perf_counter()
            response = base.http_json(
                f"http://127.0.0.1:{base.PORT}/v1/completions",
                payload,
                timeout_s=base.REQUEST_TIMEOUT_S,
            )
            wall_s = time.perf_counter() - start
            break
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= base.CASE_MAX_ATTEMPTS:
                raise
            print(f"[{now_utc()}] retry {name} after error: {exc}", flush=True)
            time.sleep(base.CASE_RETRY_SLEEP_S)
        finally:
            base.terminate_process(proc)
    else:
        raise RuntimeError(f"{name} failed") from last_error

    usage = response.get("usage", {})
    completion_tokens = int(usage.get("completion_tokens", OUTPUT_LEN))
    throughput = completion_tokens / wall_s if wall_s > 0 else 0.0
    row = {
        "repeat": repeat,
        "mode": "heter",
        "prompt_len_target": PROMPT_LEN,
        "output_len_target": OUTPUT_LEN,
        "request_wall_s": wall_s,
        "completion_tokens_per_s": throughput,
        "usage": usage,
        "finish_reason": response.get("choices", [{}])[0].get("finish_reason"),
        "server_log": str(log_path),
        "server_start_attempt": attempt,
        "omp_threads": base.HETER_OMP_THREADS,
        "cpu_affinity": base.HETER_CPU_AFFINITY,
    }
    print(
        f"[{now_utc()}] done {name}: wall={wall_s:.3f}s "
        f"tok/s={throughput:.3f}",
        flush=True,
    )
    return row


def make_pdf(rows: list[dict[str, Any]], refs: dict[str, Any]) -> None:
    repeats = [int(row["repeat"]) for row in rows]
    tps = [float(row["completion_tokens_per_s"]) for row in rows]
    wall = [float(row["request_wall_s"]) for row in rows]
    previous = refs.get("heter", {}).get("completion_tokens_per_s")

    with PdfPages(RESULT_PDF) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].bar(repeats, tps, color="#dc2626")
        if previous is not None:
            axes[0].axhline(
                float(previous),
                color="#111827",
                linestyle="--",
                linewidth=1.5,
                label="Previous matrix point",
            )
            axes[0].legend(prop=base.FONT)
        axes[0].set_title("Heter P2K/O200 Throughput Recheck",
                          fontproperties=base.FONT)
        axes[0].set_xlabel("Repeat", fontproperties=base.FONT)
        axes[0].set_ylabel("Throughput (completion tokens/s)",
                           fontproperties=base.FONT)
        axes[0].grid(axis="y", alpha=0.25)
        for x, y in zip(repeats, tps):
            axes[0].text(x, y, f"{y:.1f}", ha="center", va="bottom", fontsize=8)

        axes[1].bar(repeats, wall, color="#2563eb")
        axes[1].set_title("Request Wall Time", fontproperties=base.FONT)
        axes[1].set_xlabel("Repeat", fontproperties=base.FONT)
        axes[1].set_ylabel("Seconds", fontproperties=base.FONT)
        axes[1].grid(axis="y", alpha=0.25)
        for x, y in zip(repeats, wall):
            axes[1].text(x, y, f"{y:.1f}", ha="center", va="bottom", fontsize=8)

        fig.suptitle(
            "Anomaly Recheck: Heterogeneous P2K/O200, OMP=64, taskset=0-63",
            fontproperties=base.FONT,
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        pdf.savefig(fig)
        plt.close(fig)


def make_markdown(rows: list[dict[str, Any]], refs: dict[str, Any]) -> None:
    tps = [float(row["completion_tokens_per_s"]) for row in rows]
    walls = [float(row["request_wall_s"]) for row in rows]
    mean_tps = statistics.mean(tps)
    median_tps = statistics.median(tps)
    stdev_tps = statistics.stdev(tps) if len(tps) >= 2 else 0.0
    previous = refs.get("heter", {})
    gpu_ref = refs.get("gpu", {})
    previous_tps = float(previous.get("completion_tokens_per_s", 0.0) or 0.0)
    gpu_tps = float(gpu_ref.get("completion_tokens_per_s", 0.0) or 0.0)
    ratio_to_gpu = mean_tps / gpu_tps * 100.0 if gpu_tps > 0 else 0.0

    lines = [
        "# P2K/O200 Anomaly Recheck",
        "",
        f"- Time: `{now_utc()}`",
        f"- Workload: `prompt={PROMPT_LEN}, output={OUTPUT_LEN}, batch=1`",
        f"- Repeats: `{REPEATS}`",
        f"- Heter OMP: `OMP_NUM_THREADS={base.HETER_OMP_THREADS}`",
        f"- Heter CPU affinity: `taskset -c {base.HETER_CPU_AFFINITY}`",
        "- Prefix caching is not disabled; each repeat uses a fresh vLLM server and one measured request.",
        "- Metric: `completion_tokens / request_wall_time`.",
        "",
        "## Summary",
        "",
        f"- Previous matrix heter point: `{previous_tps:.3f} tok/s`.",
        f"- Repeat mean: `{mean_tps:.3f} tok/s`.",
        f"- Repeat median: `{median_tps:.3f} tok/s`.",
        f"- Repeat stdev: `{stdev_tps:.3f} tok/s`.",
        f"- GPU matrix reference: `{gpu_tps:.3f} tok/s`.",
        f"- Repeat mean / GPU reference: `{ratio_to_gpu:.2f}%`.",
        f"- Wall mean: `{statistics.mean(walls):.3f} s`.",
        "",
        "## Repeats",
        "",
        "| Repeat | Heter tok/s | Wall s | Attempt |",
        "|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['repeat']} | {row['completion_tokens_per_s']:.3f} | "
            f"{row['request_wall_s']:.3f} | {row['server_start_attempt']} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"- PDF: `{RESULT_PDF}`",
            f"- JSON: `{RESULT_JSON}`",
            f"- CSV: `{RESULT_CSV}`",
        ]
    )
    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_iteration_record(rows: list[dict[str, Any]]) -> None:
    tps = [float(row["completion_tokens_per_s"]) for row in rows]
    lines = [
        "",
        f"# codex update {now_local_title()}",
        "",
        "针对 `P2K/O200` 绑核异构吞吐异常点做单独重复测试。",
        "",
        f"- 实验目录：`{OUT_DIR}`",
        f"- workload：`prompt={PROMPT_LEN}, output={OUTPUT_LEN}, batch=1`",
        f"- repeats：`{REPEATS}`",
        f"- 异构服务子进程：`OMP_NUM_THREADS={base.HETER_OMP_THREADS}`。",
        f"- 异构服务子进程显式绑核：`taskset -c {base.HETER_CPU_AFFINITY}`。",
        "- 未禁用 prefix caching；每次 repeat 使用 fresh vLLM server 且只发送一次正式请求。",
        f"- 重复测试平均吞吐：`{statistics.mean(tps):.3f} tok/s`。",
        f"- 重复测试中位吞吐：`{statistics.median(tps):.3f} tok/s`。",
        "",
        "输出文件：",
        "",
        f"- `{RESULT_JSON}`",
        f"- `{RESULT_CSV}`",
        f"- `{RESULT_PDF}`",
        f"- `{RESULT_MD}`",
    ]
    with base.ITERATION_APPEND_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(base.MODEL), trust_remote_code=True)
    prompt = base.make_prompt(tokenizer, PROMPT_LEN, "P2K_O200_RECHECK")
    refs = load_matrix_reference()
    rows: list[dict[str, Any]] = []
    for repeat in range(1, REPEATS + 1):
        rows.append(run_repeat(prompt, repeat))
        RESULT_JSON.write_text(
            json.dumps(
                {
                    "timestamp_utc": now_utc(),
                    "note": "partial result",
                    "reference": refs,
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        write_csv(rows)

    result = {
        "timestamp_utc": now_utc(),
        "workload": {
            "prompt_len": PROMPT_LEN,
            "output_len": OUTPUT_LEN,
            "batch_size": 1,
        },
        "omp_threads": base.HETER_OMP_THREADS,
        "cpu_affinity": base.HETER_CPU_AFFINITY,
        "reference": refs,
        "rows": rows,
    }
    RESULT_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(rows)
    make_pdf(rows, refs)
    make_markdown(rows, refs)
    append_iteration_record(rows)
    print(f"[{now_utc()}] wrote {RESULT_JSON}", flush=True)
    print(f"[{now_utc()}] wrote {RESULT_CSV}", flush=True)
    print(f"[{now_utc()}] wrote {RESULT_PDF}", flush=True)
    print(f"[{now_utc()}] wrote {RESULT_MD}", flush=True)


if __name__ == "__main__":
    main()
