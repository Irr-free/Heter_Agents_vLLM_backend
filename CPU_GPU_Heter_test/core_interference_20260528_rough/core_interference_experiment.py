#!/usr/bin/env python3
"""Run coarse core-interference sensitivity tests for heter CPU attention."""

from __future__ import annotations

import csv
import json
import os
import re
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties, fontManager
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "CPU_GPU_Heter_test" / "core_interference_20260528_rough"
PYTHON = Path("/home/zb/tool/miniconda3/envs/vLLM_GPU/bin/python")
VLLM = Path("/home/zb/tool/miniconda3/envs/vLLM_GPU/bin/vllm")
MODEL = Path("/home/zb/data/model_data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct")
CPU_ATTN_LIB = ROOT / "vllm" / "libs" / "libvllm_heter_cpu_attn.so"
CORE_BURNER_SRC = OUT_DIR / "core_burner.c"
CORE_BURNER_BIN = OUT_DIR / "core_burner"

PORT = 8036
PROMPT_LEN = 8000
OUTPUT_LEN = 400
GPU_MEMORY_UTILIZATION = "0.25"
MAX_MODEL_LEN = "10000"

LLM_CORE_COUNTS = [8, 16, 32, 64, 96, 128]
INTERFERENCE_CORE_COUNTS = [0, 10, 20, 40, 60, 80, 100, 120]
AFFINITY_MODES = ["unbound", "bound"]
REPEATS = 1
REQUEST_TIMEOUT_S = 300.0
TIMEOUT_SYNTHETIC_TPS = 0.1
SERVER_READY_TIMEOUT_S = 700.0

RESULT_JSON = OUT_DIR / "core_interference_results.json"
RESULT_CSV = OUT_DIR / "core_interference_raw.csv"
RESULT_MD = OUT_DIR / "core_interference_summary.md"
ITERATION_APPEND_MD = ROOT / "CPU_GPU_Heter_test" / "迭代记录.md"

LATIN_FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
CJK_FONT = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")

REQUEST_EVENTS = [
    "openai_completion_create_total",
    "openai_completion_generate_iteration",
    "engine_core_execute_model_submit",
    "engine_core_wait_model_or_sample_future",
    "engine_core_scheduler_update_from_output",
    "gpu_worker_execute_model_total",
    "gpu_model_runner_execute_model_total",
    "gpu_model_runner_preprocess_total",
    "gpu_model_runner_model_forward",
    "gpu_model_runner_postprocess_total",
    "gpu_model_runner_sample_tokens_total",
    "output_processor_process_outputs_total",
]

ATTENTION_EVENTS = [
    "decode_cpu_path_layer_total",
    "cpu_attention_cpp_total",
    "decode_metadata_d2h",
    "decode_kv_d2h",
    "decode_kv_cache_write_total",
    "decode_query_d2h",
    "decode_output_h2d",
]


@dataclass(frozen=True)
class CpuInfo:
    cpu: int
    core: int
    socket: int
    node: int


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


def now_local_title() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def cpu_list_to_str(cpus: list[int]) -> str:
    return ",".join(str(c) for c in cpus)


def parse_lscpu() -> list[CpuInfo]:
    output = subprocess.check_output(
        ["lscpu", "-e=CPU,CORE,SOCKET,NODE,ONLINE"],
        text=True,
    )
    infos: list[CpuInfo] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or line.startswith("CPU"):
            continue
        parts = line.split()
        if len(parts) < 5 or parts[4] != "yes":
            continue
        infos.append(
            CpuInfo(
                cpu=int(parts[0]),
                core=int(parts[1]),
                socket=int(parts[2]),
                node=int(parts[3]),
            )
        )
    return sorted(infos, key=lambda x: (x.socket, x.core, x.cpu))


def physical_key(info: CpuInfo) -> tuple[int, int]:
    return (info.socket, info.core)


def primary_then_smt_order(infos: list[CpuInfo]) -> list[int]:
    by_core: dict[tuple[int, int], list[int]] = defaultdict(list)
    for info in infos:
        by_core[physical_key(info)].append(info.cpu)
    for cpus in by_core.values():
        cpus.sort()

    first_pass: list[int] = []
    later_pass: list[int] = []
    for key in sorted(by_core):
        cpus = by_core[key]
        first_pass.append(cpus[0])
        later_pass.extend(cpus[1:])
    return first_pass + later_pass


def build_cpuset(
    infos: list[CpuInfo],
    llm_count: int,
    interference_count: int,
    bound: bool,
) -> dict[str, Any]:
    ordered = primary_then_smt_order(infos)
    cpu_to_info = {info.cpu: info for info in infos}
    all_cpus = set(ordered)

    if bound:
        llm_cpus = ordered[:llm_count]
        remaining = [cpu for cpu in ordered if cpu not in set(llm_cpus)]
        interference_cpus = remaining[:interference_count]
        if len(interference_cpus) < interference_count:
            need = interference_count - len(interference_cpus)
            interference_cpus.extend(llm_cpus[:need])
    else:
        llm_cpus = []
        interference_cpus = ordered[:interference_count]

    llm_set = set(llm_cpus) if bound else all_cpus
    int_set = set(interference_cpus)
    logical_overlap = sorted(llm_set & int_set) if bound else []

    llm_physical = {
        physical_key(cpu_to_info[cpu]) for cpu in llm_set if cpu in cpu_to_info
    }
    int_physical = {
        physical_key(cpu_to_info[cpu]) for cpu in int_set if cpu in cpu_to_info
    }
    physical_overlap = sorted(llm_physical & int_physical)

    sibling_overlap = 0
    for key in llm_physical & int_physical:
        llm_siblings = {cpu for cpu in llm_set if physical_key(cpu_to_info[cpu]) == key}
        int_siblings = {cpu for cpu in int_set if physical_key(cpu_to_info[cpu]) == key}
        if llm_siblings and int_siblings and not (llm_siblings & int_siblings):
            sibling_overlap += 1

    return {
        "llm_cpus": llm_cpus,
        "interference_cpus": interference_cpus,
        "logical_overlap": logical_overlap,
        "physical_overlap_count": len(physical_overlap),
        "smt_sibling_overlap_count": sibling_overlap,
        "logical_disjoint": len(logical_overlap) == 0,
        "physical_disjoint": len(physical_overlap) == 0,
        "all_available_cpus": ordered,
    }


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


def wait_ready(
    port: int,
    timeout_s: float,
    log_path: Path,
    proc: subprocess.Popen[Any] | None = None,
) -> None:
    url = f"http://127.0.0.1:{port}/v1/models"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
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


def terminate_process(proc: subprocess.Popen[Any], timeout_s: float = 45.0) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=timeout_s)
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


def compile_burner() -> None:
    if CORE_BURNER_BIN.exists() and (
        CORE_BURNER_BIN.stat().st_mtime >= CORE_BURNER_SRC.stat().st_mtime
    ):
        return
    subprocess.check_call(
        [
            "gcc",
            "-O3",
            "-march=native",
            "-pthread",
            str(CORE_BURNER_SRC),
            "-lm",
            "-o",
            str(CORE_BURNER_BIN),
        ]
    )


def start_core_burner(cpus: list[int], log_path: Path) -> subprocess.Popen[bytes] | None:
    if not cpus:
        return None
    compile_burner()
    with log_path.open("wb") as log_f:
        proc = subprocess.Popen(
            [str(CORE_BURNER_BIN), "--cpus", cpu_list_to_str(cpus)],
            cwd=str(ROOT),
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
    time.sleep(2.0)
    if proc.poll() is not None:
        raise RuntimeError(f"core_burner exited early; see {log_path}")
    return proc


def read_summary_files(prefix: Path) -> dict[str, dict[str, float]]:
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


def event_count(events: dict[str, dict[str, float]], name: str) -> int:
    return int(events.get(name, {}).get("count", 0))


def diff_events(
    after: dict[str, dict[str, float]],
    before: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for name in set(after) | set(before):
        count = event_count(after, name) - event_count(before, name)
        total = event_total(after, name) - event_total(before, name)
        max_ms = float(after.get(name, {}).get("max_ms", 0.0))
        out[name] = {
            "count": max(0, count),
            "total_s": max(0.0, total),
            "max_ms": max_ms,
            "avg_ms": total / count * 1000.0 if count > 0 else 0.0,
        }
    return out


def read_proc_status(pid: int) -> dict[str, int]:
    path = Path("/proc") / str(pid) / "status"
    result = {
        "voluntary_ctxt_switches": 0,
        "nonvoluntary_ctxt_switches": 0,
    }
    if not path.exists():
        return result
    text = path.read_text(errors="ignore")
    for key in result:
        match = re.search(rf"^{key}:\s+(\d+)$", text, re.MULTILINE)
        if match:
            result[key] = int(match.group(1))
    return result


def run_request(prompt: str) -> tuple[float, dict[str, Any], str | None]:
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
        timeout_s=REQUEST_TIMEOUT_S,
    )
    wall_s = time.perf_counter() - start
    finish_reason = response.get("choices", [{}])[0].get("finish_reason")
    return wall_s, response.get("usage", {}), finish_reason


def make_timeout_row(
    *,
    affinity_mode: str,
    llm_core_count: int,
    interference_count: int,
    repeat: int,
    cpuset: dict[str, Any],
    server_pid: int,
    log_path: Path,
    burner_log: Path,
    before_status: dict[str, int],
    after_status: dict[str, int],
    request_events: dict[str, dict[str, float]],
    attention_events: dict[str, dict[str, float]],
) -> dict[str, Any]:
    return {
        "timestamp_utc": now_utc(),
        "affinity_mode": affinity_mode,
        "llm_core_count": llm_core_count,
        "interference_core_count": interference_count,
        "repeat": repeat,
        "request_wall_s": REQUEST_TIMEOUT_S,
        "timed_out": True,
        "timeout_policy": (
            f"request exceeded {REQUEST_TIMEOUT_S:.0f}s; "
            f"throughput recorded as {TIMEOUT_SYNTHETIC_TPS} tok/s"
        ),
        "usage": {
            "prompt_tokens": PROMPT_LEN + 1,
            "completion_tokens": OUTPUT_LEN,
            "total_tokens": PROMPT_LEN + 1 + OUTPUT_LEN,
        },
        "finish_reason": "timeout_synthetic",
        "request_events": request_events,
        "attention_events": attention_events,
        "logical_disjoint": cpuset["logical_disjoint"],
        "physical_disjoint": cpuset["physical_disjoint"],
        "logical_overlap": cpuset["logical_overlap"],
        "physical_overlap_count": cpuset["physical_overlap_count"],
        "smt_sibling_overlap_count": cpuset["smt_sibling_overlap_count"],
        "llm_cpuset": cpu_list_to_str(cpuset["llm_cpus"])
        if affinity_mode == "bound"
        else "unbound",
        "interference_cpuset": cpu_list_to_str(cpuset["interference_cpus"]),
        "server_pid": server_pid,
        "server_log": str(log_path),
        "burner_log": str(burner_log) if burner_log.exists() else None,
        "server_voluntary_ctxt_switches_delta": (
            after_status["voluntary_ctxt_switches"]
            - before_status["voluntary_ctxt_switches"]
        ),
        "server_nonvoluntary_ctxt_switches_delta": (
            after_status["nonvoluntary_ctxt_switches"]
            - before_status["nonvoluntary_ctxt_switches"]
        ),
    }


def make_synthetic_timeout_row(
    *,
    affinity_mode: str,
    llm_core_count: int,
    interference_count: int,
    repeat: int,
    cpuset: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "timestamp_utc": now_utc(),
        "affinity_mode": affinity_mode,
        "llm_core_count": llm_core_count,
        "interference_core_count": interference_count,
        "repeat": repeat,
        "request_wall_s": REQUEST_TIMEOUT_S,
        "timed_out": True,
        "synthetic_skipped": True,
        "timeout_policy": reason,
        "usage": {
            "prompt_tokens": PROMPT_LEN + 1,
            "completion_tokens": OUTPUT_LEN,
            "total_tokens": PROMPT_LEN + 1 + OUTPUT_LEN,
        },
        "finish_reason": "timeout_synthetic_skipped",
        "request_events": {},
        "attention_events": {},
        "logical_disjoint": cpuset["logical_disjoint"],
        "physical_disjoint": cpuset["physical_disjoint"],
        "logical_overlap": cpuset["logical_overlap"],
        "physical_overlap_count": cpuset["physical_overlap_count"],
        "smt_sibling_overlap_count": cpuset["smt_sibling_overlap_count"],
        "llm_cpuset": cpu_list_to_str(cpuset["llm_cpus"])
        if affinity_mode == "bound"
        else "unbound",
        "interference_cpuset": cpu_list_to_str(cpuset["interference_cpus"]),
        "server_pid": -1,
        "server_log": None,
        "burner_log": None,
        "server_voluntary_ctxt_switches_delta": 0,
        "server_nonvoluntary_ctxt_switches_delta": 0,
    }


def row_key(row: dict[str, Any]) -> tuple[str, int, int, int]:
    return (
        str(row["affinity_mode"]),
        int(row["llm_core_count"]),
        int(row["interference_core_count"]),
        int(row["repeat"]),
    )


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for row in rows:
        out[row_key(row)] = row
    return normalize_timeout_rows(
        [out[key] for key in sorted(out, key=lambda x: (x[0], x[1], x[2], x[3]))]
    )


def load_existing_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if RESULT_JSON.exists():
        try:
            payload = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
            rows.extend(payload.get("rows", []))
        except Exception as exc:  # noqa: BLE001
            print(f"[{now_utc()}] warning: failed to load {RESULT_JSON}: {exc}",
                  flush=True)
    for path in sorted(OUT_DIR.glob("*_partial.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.extend(payload.get("rows", []))
        except Exception as exc:  # noqa: BLE001
            print(f"[{now_utc()}] warning: failed to load {path}: {exc}",
                  flush=True)
    return dedupe_rows(rows)


def save_progress(results: dict[str, Any]) -> None:
    results["rows"] = dedupe_rows(results["rows"])
    results["aggregated"] = aggregate(results["rows"])
    RESULT_JSON.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(results["rows"])


def launch_server(
    affinity_mode: str,
    llm_core_count: int,
    cpuset: dict[str, Any],
    request_summary_prefix: Path,
    attention_summary_prefix: Path,
    log_path: Path,
) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT),
            "OMP_NUM_THREADS": str(llm_core_count),
            "VLLM_HETER_CPU_ATTN_LIB": str(CPU_ATTN_LIB),
            "VLLM_HETER_PROFILE": "1",
            "VLLM_HETER_PROFILE_MODE": "memory",
            "VLLM_HETER_PROFILE_SUMMARY_PATH": str(attention_summary_prefix),
            "VLLM_HETER_PROFILE_FLUSH_INTERVAL_S": "1.0",
            "VLLM_HETER_REQUEST_PROFILE": "1",
            "VLLM_HETER_REQUEST_PROFILE_MODE": "memory",
            "VLLM_HETER_REQUEST_PROFILE_SUMMARY_PATH": str(request_summary_prefix),
            "VLLM_HETER_REQUEST_PROFILE_FLUSH_INTERVAL_S": "1.0",
            "VLLM_HETER_CUDAGRAPH_DEBUG": "0",
            "no_proxy": "localhost,127.0.0.1",
            "NO_PROXY": "localhost,127.0.0.1",
        }
    )
    env.pop("VLLM_HETER_DISABLE_CPU_ATTENTION", None)
    env.pop("VLLM_HETER_SKIP_CPU_ATTN_COMPUTE", None)

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
        "--no-enable-prefix-caching",
        "--trust-remote-code",
    ]
    if affinity_mode == "bound":
        cmd = ["taskset", "-c", cpu_list_to_str(cpuset["llm_cpus"])] + cmd

    with log_path.open("wb") as log_f:
        return subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )


def cleanup_prefix(prefix: Path) -> None:
    for path in prefix.parent.glob(f"{prefix.stem}.pid*.json"):
        path.unlink()


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    attention = row["attention_events"]
    request = row["request_events"]
    usage = row.get("usage", {})
    completion_tokens = usage.get("completion_tokens", OUTPUT_LEN)
    prompt_tokens = usage.get("prompt_tokens", PROMPT_LEN)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
    if row.get("timed_out", False):
        completion_tps = TIMEOUT_SYNTHETIC_TPS
    else:
        completion_tps = (
            float(completion_tokens) / float(row["request_wall_s"])
            if row["request_wall_s"] > 0
            else 0.0
        )
    return {
        "affinity_mode": row["affinity_mode"],
        "llm_core_count": row["llm_core_count"],
        "interference_core_count": row["interference_core_count"],
        "repeat": row["repeat"],
        "request_wall_s": row["request_wall_s"],
        "timed_out": row.get("timed_out", False),
        "completion_tokens_per_s": completion_tps,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "finish_reason": row.get("finish_reason"),
        "decode_cpu_path_layer_total_s": event_total(
            attention, "decode_cpu_path_layer_total"
        ),
        "cpu_attention_cpp_total_s": event_total(attention, "cpu_attention_cpp_total"),
        "decode_kv_d2h_s": event_total(attention, "decode_kv_d2h"),
        "decode_query_d2h_s": event_total(attention, "decode_query_d2h"),
        "decode_output_h2d_s": event_total(attention, "decode_output_h2d"),
        "gpu_model_runner_model_forward_s": event_total(
            request, "gpu_model_runner_model_forward"
        ),
        "logical_disjoint": row["logical_disjoint"],
        "physical_disjoint": row["physical_disjoint"],
        "logical_overlap_count": len(row["logical_overlap"]),
        "physical_overlap_count": row["physical_overlap_count"],
        "smt_sibling_overlap_count": row["smt_sibling_overlap_count"],
        "llm_cpuset": row["llm_cpuset"],
        "interference_cpuset": row["interference_cpuset"],
        "server_pid": row["server_pid"],
        "server_voluntary_ctxt_switches_delta": row.get(
            "server_voluntary_ctxt_switches_delta", 0
        ),
        "server_nonvoluntary_ctxt_switches_delta": row.get(
            "server_nonvoluntary_ctxt_switches_delta", 0
        ),
    }


def normalize_timeout_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        if float(row.get("request_wall_s", 0.0)) > REQUEST_TIMEOUT_S:
            row["request_wall_s"] = REQUEST_TIMEOUT_S
            row["timed_out"] = True
            row["timeout_policy"] = (
                f"post-normalized because wall exceeded {REQUEST_TIMEOUT_S:.0f}s; "
                f"throughput recorded as {TIMEOUT_SYNTHETIC_TPS} tok/s"
            )
            row["finish_reason"] = "timeout_synthetic"
            usage = row.setdefault("usage", {})
            usage.setdefault("prompt_tokens", PROMPT_LEN + 1)
            usage.setdefault("completion_tokens", OUTPUT_LEN)
            usage.setdefault("total_tokens", PROMPT_LEN + 1 + OUTPUT_LEN)
        else:
            row.setdefault("timed_out", False)
    return rows


def write_csv(rows: list[dict[str, Any]]) -> None:
    flat = [flatten_row(row) for row in rows]
    if not flat:
        return
    with RESULT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["affinity_mode"],
            int(row["llm_core_count"]),
            int(row["interference_core_count"]),
        )
        buckets[key].append(flatten_row(row))

    out: list[dict[str, Any]] = []
    for (mode, llm, interference), items in sorted(buckets.items()):
        base = {
            "affinity_mode": mode,
            "llm_core_count": llm,
            "interference_core_count": interference,
            "repeat_count": len(items),
        }
        for metric in [
            "request_wall_s",
            "completion_tokens_per_s",
            "decode_cpu_path_layer_total_s",
            "cpu_attention_cpp_total_s",
            "gpu_model_runner_model_forward_s",
        ]:
            base[f"{metric}_median"] = median([float(item[metric]) for item in items])
        base["timed_out_count"] = sum(1 for item in items if bool(item["timed_out"]))
        base["logical_disjoint"] = all(bool(item["logical_disjoint"]) for item in items)
        base["physical_disjoint"] = all(bool(item["physical_disjoint"]) for item in items)
        base["max_logical_overlap_count"] = max(
            int(item["logical_overlap_count"]) for item in items
        )
        base["max_physical_overlap_count"] = max(
            int(item["physical_overlap_count"]) for item in items
        )
        out.append(base)

    idle: dict[tuple[str, int], dict[str, Any]] = {}
    for item in out:
        if item["interference_core_count"] == 0:
            idle[(item["affinity_mode"], item["llm_core_count"])] = item
    for item in out:
        baseline = idle.get((item["affinity_mode"], item["llm_core_count"]))
        if baseline:
            for metric in [
                "request_wall_s_median",
                "decode_cpu_path_layer_total_s_median",
                "cpu_attention_cpp_total_s_median",
            ]:
                base_value = float(baseline[metric])
                item[metric.replace("_median", "_slowdown")] = (
                    float(item[metric]) / base_value if base_value > 0 else 0.0
                )
            base_tps = float(baseline["completion_tokens_per_s_median"])
            item["completion_tokens_per_s_ratio"] = (
                float(item["completion_tokens_per_s_median"]) / base_tps
                if base_tps > 0
                else 0.0
            )
    return out


def plot_metric(
    pdf: PdfPages,
    aggregated: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.5), sharey=True)
    colors = {
        8: "#1f77b4",
        16: "#ff7f0e",
        32: "#2ca02c",
        64: "#d62728",
        96: "#9467bd",
        128: "#8c564b",
    }
    for ax, mode in zip(axes, AFFINITY_MODES):
        subset = [item for item in aggregated if item["affinity_mode"] == mode]
        for llm in LLM_CORE_COUNTS:
            points = sorted(
                [item for item in subset if item["llm_core_count"] == llm],
                key=lambda item: item["interference_core_count"],
            )
            if not points:
                continue
            x = [item["interference_core_count"] for item in points]
            y = [item.get(metric, 0.0) for item in points]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.8,
                markersize=4.5,
                color=colors[llm],
                label=f"LLM={llm}",
            )
            for xi, yi, item in zip(x, y, points):
                if mode == "bound" and not item.get("physical_disjoint", True):
                    ax.scatter([xi], [yi], marker="x", s=60, color=colors[llm])
                if item.get("timed_out_count", 0) > 0:
                    ax.scatter(
                        [xi],
                        [yi],
                        marker="s",
                        s=45,
                        facecolors="none",
                        edgecolors=colors[llm],
                    )
        ax.set_title(mode, loc="left", fontproperties=FONT)
        ax.set_xlabel("background core-bound workers", fontproperties=FONT)
        ax.grid(True, alpha=0.25)
        ax.set_xticks(INTERFERENCE_CORE_COUNTS)
    axes[0].set_ylabel(ylabel, fontproperties=FONT)
    axes[1].legend(prop=FONT, fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(title, fontproperties=FONT, fontsize=14)
    fig.text(
        0.01,
        0.01,
        "x marker: bound case cannot keep physical-core disjoint with interference.",
        fontsize=8,
        fontproperties=FONT,
    )
    fig.text(
        0.01,
        0.035,
        "open square: request exceeded 300s and is recorded with synthetic throughput 0.1 tok/s.",
        fontsize=8,
        fontproperties=FONT,
    )
    fig.tight_layout(rect=(0, 0.03, 0.91, 0.95))
    pdf.savefig(fig)
    plt.close(fig)


def make_pdf(results: dict[str, Any]) -> list[str]:
    aggregated = results["aggregated"]
    pdf_paths = {
        "request_wall": OUT_DIR / "core_interference_request_wall.pdf",
        "tokens_per_sec": OUT_DIR / "core_interference_tokens_per_sec.pdf",
        "decode_cpu": OUT_DIR / "core_interference_decode_cpu.pdf",
        "slowdown": OUT_DIR / "core_interference_slowdown.pdf",
    }
    with PdfPages(pdf_paths["request_wall"]) as pdf:
        plot_metric(
            pdf,
            aggregated,
            "request_wall_s_median",
            "request wall, seconds",
            "Core interference: request wall time",
        )
    with PdfPages(pdf_paths["tokens_per_sec"]) as pdf:
        plot_metric(
            pdf,
            aggregated,
            "completion_tokens_per_s_median",
            "completion tokens/s",
            "Core interference: generated token throughput",
        )
    with PdfPages(pdf_paths["decode_cpu"]) as pdf:
        plot_metric(
            pdf,
            aggregated,
            "decode_cpu_path_layer_total_s_median",
            "decode_cpu_path_layer_total, seconds",
            "Core interference: CPU decode attention path",
        )
    with PdfPages(pdf_paths["slowdown"]) as pdf:
        plot_metric(
            pdf,
            aggregated,
            "decode_cpu_path_layer_total_s_slowdown",
            "slowdown vs same-mode idle",
            "Core interference: CPU decode attention slowdown",
        )
    return [str(path) for path in pdf_paths.values()]


def make_markdown(results: dict[str, Any]) -> None:
    aggregated = results["aggregated"]
    pdfs = results["pdfs"]
    lines = [
        "# Core 占用干扰粗粒度实验记录",
        "",
        f"- 时间：`{results['timestamp_utc']}`",
        f"- workload：`prompt={PROMPT_LEN}`，`output={OUTPUT_LEN}`，`batch=1`",
        f"- LLM 核数/线程数：`{LLM_CORE_COUNTS}`",
        f"- 干扰核数：`{INTERFERENCE_CORE_COUNTS}`",
        f"- 重复次数：`{REPEATS}`",
        f"- 超时规则：请求超过 `{REQUEST_TIMEOUT_S:.0f}s` 未完成则停止该点，吞吐按 `{TIMEOUT_SYNTHETIC_TPS} tok/s` 记录。",
        "- 跳过规则：同一 LLM 核数/亲和性下，一旦某个干扰核数 timeout，后续更高干扰核数直接标记为 synthetic timeout，不再实际运行。",
        f"- vLLM 参数：`--gpu-memory-utilization {GPU_MEMORY_UTILIZATION}`，`--max-model-len {MAX_MODEL_LEN}`",
        "- Prefix cache：显式设置 `--no-enable-prefix-caching`，避免同一 server 内连续请求复用 8K prompt。",
        f"- JSON：`{RESULT_JSON}`",
        f"- CSV：`{RESULT_CSV}`",
        "",
        "## 输出 PDF",
        "",
    ]
    for path in pdfs:
        lines.append(f"- `{path}`")

    lines.extend(
        [
            "",
            "## 实验设计",
            "",
            "- `unbound`：vLLM server 不使用 `taskset` 绑定 CPU affinity，但设置 `OMP_NUM_THREADS=LLM核数`。",
            "- `bound`：vLLM server 使用 `taskset` 绑定到指定 LLM cpuset，同时设置 `OMP_NUM_THREADS=LLM核数`。",
            "- bound 模式下优先让干扰 cpuset 与 LLM cpuset disjoint；机器资源不足时记录 logical/physical overlap。",
            "- 干扰程序 `core_burner` 只做寄存器内算术循环，避免制造大规模 DRAM streaming。",
            "- profile 使用 memory summary 前后差分，不启用逐事件 JSONL。",
            "- 早期一次试跑发现 prefix cache hit rate 会随重复请求升高，因此已中止无效试跑并从本结果中排除。",
            "- 同一 LLM 配置下 timeout 后的更高干扰点按用户指定规则直接跳过并记为 timeout。",
            "",
            "## 汇总结果",
            "",
            "| mode | LLM cores | interference cores | wall median | tok/s median | decode CPU median | decode slowdown | timeout | physical disjoint | physical overlap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
    )
    for item in aggregated:
        lines.append(
            f"| {item['affinity_mode']} | {item['llm_core_count']} | "
            f"{item['interference_core_count']} | "
            f"{item['request_wall_s_median']:.3f} | "
            f"{item['completion_tokens_per_s_median']:.3f} | "
            f"{item['decode_cpu_path_layer_total_s_median']:.3f} | "
            f"{item.get('decode_cpu_path_layer_total_s_slowdown', 0.0):.3f} | "
            f"{item.get('timed_out_count', 0)} | "
            f"{item['physical_disjoint']} | "
            f"{item['max_physical_overlap_count']} |"
        )

    lines.extend(
        [
            "",
            "## 临时环境变量",
            "",
            "- 仅在 vLLM server 子进程内设置 `OMP_NUM_THREADS`。",
            "- 仅在 vLLM server 子进程内设置 `VLLM_HETER_CPU_ATTN_LIB`。",
            "- 仅在 vLLM server 子进程内设置 `VLLM_HETER_PROFILE*` 和 `VLLM_HETER_REQUEST_PROFILE*`。",
            "- 仅在 vLLM server 子进程内设置 `no_proxy/NO_PROXY`。",
            "- 绘图建议使用 `MPLCONFIGDIR=/tmp/matplotlib-codex` 和 `XDG_CACHE_HOME=/tmp/matplotlib-codex`，不修改持久系统环境变量。",
            "- 未修改系统设置。",
        ]
    )
    RESULT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_iteration_record(results: dict[str, Any]) -> None:
    lines = [
        "",
        f"# codex update {now_local_title()}",
        "",
        "本次新增并执行 Core 占用干扰粗粒度实验，所有新增文件位于：",
        "",
        f"- `{OUT_DIR}`",
        "",
        "实验内容：",
        "",
        f"- LLM CPU attention 线程/绑核规模：`{LLM_CORE_COUNTS}`。",
        f"- 干扰 core 数：`{INTERFERENCE_CORE_COUNTS}`。",
        f"- 请求超过 `{REQUEST_TIMEOUT_S:.0f}s` 的点按 `{TIMEOUT_SYNTHETIC_TPS} tok/s` 记录，并标记 timeout。",
        "- 同一 LLM 配置中若某个干扰核数已 timeout，后续更高干扰核数直接 synthetic timeout，不继续实际等待。",
        "- 对比 `unbound` 与 `bound` 两种 LLM CPU affinity 模式。",
        "- bound 模式优先保证 LLM cpuset 与干扰 cpuset disjoint，资源不足时记录 overlap。",
        "- 干扰负载使用寄存器内算术循环，尽量隔离 core 占用影响。",
        "",
        "输出文件：",
        "",
        f"- `{RESULT_JSON}`",
        f"- `{RESULT_CSV}`",
        f"- `{RESULT_MD}`",
    ]
    for path in results.get("pdfs", []):
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "环境变量记录：",
            "",
            "- 仅对子进程临时设置 `OMP_NUM_THREADS`、`VLLM_HETER_CPU_ATTN_LIB`、`VLLM_HETER_PROFILE*`、`VLLM_HETER_REQUEST_PROFILE*`、`no_proxy/NO_PROXY`。",
            "- 未修改系统持久环境变量或系统设置。",
        ]
    )
    with ITERATION_APPEND_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_one_server_matrix(
    prompt: str,
    infos: list[CpuInfo],
    affinity_mode: str,
    llm_core_count: int,
    completed_keys: set[tuple[str, int, int, int]],
    results: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    server_name = f"{affinity_mode}_llm{llm_core_count}"
    timed_out_cutoff = None
    for row in results["rows"]:
        if (row["affinity_mode"] == affinity_mode
                and int(row["llm_core_count"]) == llm_core_count
                and bool(row.get("timed_out", False))):
            value = int(row["interference_core_count"])
            timed_out_cutoff = (
                value if timed_out_cutoff is None else min(timed_out_cutoff, value)
            )

    for interference_count in INTERFERENCE_CORE_COUNTS:
        if timed_out_cutoff is None or interference_count < timed_out_cutoff:
            continue
        for repeat in range(REPEATS):
            key = (affinity_mode, llm_core_count, interference_count, repeat)
            if key in completed_keys:
                continue
            cpuset = build_cpuset(
                infos,
                llm_count=llm_core_count,
                interference_count=interference_count,
                bound=(affinity_mode == "bound"),
            )
            row = make_synthetic_timeout_row(
                affinity_mode=affinity_mode,
                llm_core_count=llm_core_count,
                interference_count=interference_count,
                repeat=repeat,
                cpuset=cpuset,
                reason=(
                    f"skipped because interference={timed_out_cutoff} already "
                    f"timed out for {affinity_mode} LLM={llm_core_count}; "
                    f"throughput recorded as {TIMEOUT_SYNTHETIC_TPS} tok/s"
                ),
            )
            rows.append(row)
            results["rows"].append(row)
            completed_keys.add(row_key(row))
            save_progress(results)
            print(
                f"[{now_utc()}] synthetic timeout "
                f"{affinity_mode}_llm{llm_core_count}_int{interference_count}_rep{repeat}",
                flush=True,
            )

    if not has_pending(completed_keys, affinity_mode, llm_core_count):
        return rows

    log_path = OUT_DIR / f"{server_name}_server.log"
    request_summary_prefix = OUT_DIR / f"{server_name}_request_summary.json"
    attention_summary_prefix = OUT_DIR / f"{server_name}_attention_summary.json"
    cleanup_prefix(request_summary_prefix)
    cleanup_prefix(attention_summary_prefix)
    if log_path.exists():
        log_path.unlink()

    base_cpuset = build_cpuset(
        infos,
        llm_count=llm_core_count,
        interference_count=0,
        bound=(affinity_mode == "bound"),
    )
    proc = launch_server(
        affinity_mode,
        llm_core_count,
        base_cpuset,
        request_summary_prefix,
        attention_summary_prefix,
        log_path,
    )
    try:
        wait_ready(PORT, SERVER_READY_TIMEOUT_S, log_path, proc)
        server_pid = proc.pid
        for interference_count in INTERFERENCE_CORE_COUNTS:
            cpuset = build_cpuset(
                infos,
                llm_count=llm_core_count,
                interference_count=interference_count,
                bound=(affinity_mode == "bound"),
            )
            for repeat in range(REPEATS):
                key = (affinity_mode, llm_core_count, interference_count, repeat)
                if key in completed_keys:
                    print(
                        f"[{now_utc()}] skip existing "
                        f"{affinity_mode}_llm{llm_core_count}_int{interference_count}_rep{repeat}",
                        flush=True,
                    )
                    continue
                run_name = (
                    f"{affinity_mode}_llm{llm_core_count}_int{interference_count}"
                    f"_rep{repeat}"
                )
                print(f"[{now_utc()}] run {run_name}", flush=True)
                burner_log = OUT_DIR / f"{run_name}_burner.log"
                before_request = read_summary_files(request_summary_prefix)
                before_attention = read_summary_files(attention_summary_prefix)
                before_status = read_proc_status(server_pid)
                burner_proc = start_core_burner(cpuset["interference_cpus"], burner_log)
                timed_out = False
                try:
                    try:
                        wall_s, usage, finish_reason = run_request(prompt)
                    except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
                        timed_out = True
                        print(
                            f"[{now_utc()}] timeout {run_name}: {type(exc).__name__}; "
                            "stop server and record synthetic throughput",
                            flush=True,
                        )
                finally:
                    if burner_proc is not None:
                        terminate_process(burner_proc, timeout_s=10.0)
                time.sleep(2.5)
                after_status = read_proc_status(server_pid)
                after_request = read_summary_files(request_summary_prefix)
                after_attention = read_summary_files(attention_summary_prefix)
                request_events = diff_events(after_request, before_request)
                attention_events = diff_events(after_attention, before_attention)
                if timed_out:
                    row = make_timeout_row(
                        affinity_mode=affinity_mode,
                        llm_core_count=llm_core_count,
                        interference_count=interference_count,
                        repeat=repeat,
                        cpuset=cpuset,
                        server_pid=server_pid,
                        log_path=log_path,
                        burner_log=burner_log,
                        before_status=before_status,
                        after_status=after_status,
                        request_events=request_events,
                        attention_events=attention_events,
                    )
                else:
                    row = {
                        "timestamp_utc": now_utc(),
                        "affinity_mode": affinity_mode,
                        "llm_core_count": llm_core_count,
                        "interference_core_count": interference_count,
                        "repeat": repeat,
                        "request_wall_s": wall_s,
                        "timed_out": False,
                        "usage": usage,
                        "finish_reason": finish_reason,
                        "request_events": request_events,
                        "attention_events": attention_events,
                        "logical_disjoint": cpuset["logical_disjoint"],
                        "physical_disjoint": cpuset["physical_disjoint"],
                        "logical_overlap": cpuset["logical_overlap"],
                        "physical_overlap_count": cpuset["physical_overlap_count"],
                        "smt_sibling_overlap_count": cpuset["smt_sibling_overlap_count"],
                        "llm_cpuset": cpu_list_to_str(cpuset["llm_cpus"])
                        if affinity_mode == "bound"
                        else "unbound",
                        "interference_cpuset": cpu_list_to_str(
                            cpuset["interference_cpus"]
                        ),
                        "server_pid": server_pid,
                        "server_log": str(log_path),
                        "burner_log": str(burner_log) if burner_log.exists() else None,
                        "server_voluntary_ctxt_switches_delta": (
                            after_status["voluntary_ctxt_switches"]
                            - before_status["voluntary_ctxt_switches"]
                        ),
                        "server_nonvoluntary_ctxt_switches_delta": (
                            after_status["nonvoluntary_ctxt_switches"]
                            - before_status["nonvoluntary_ctxt_switches"]
                        ),
                    }
                rows.append(row)
                results["rows"].append(row)
                completed_keys.add(row_key(row))
                save_progress(results)
                row_wall_s = float(row["request_wall_s"])
                print(
                    f"[{now_utc()}] done {run_name}: wall={row_wall_s:.3f}s "
                    f"decode_cpu={event_total(attention_events, 'decode_cpu_path_layer_total'):.3f}s "
                    f"tps={flatten_row(row)['completion_tokens_per_s']:.3f}"
                    f"{' timeout' if timed_out else ''}",
                    flush=True,
                )
                partial = {
                    "timestamp_utc": now_utc(),
                    "rows": rows,
                    "note": "partial result for current server matrix",
                }
                (OUT_DIR / f"{server_name}_partial.json").write_text(
                    json.dumps(partial, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                if timed_out:
                    terminate_process(proc, timeout_s=10.0)
                    return rows
    finally:
        terminate_process(proc)
    return rows


def has_pending(completed_keys: set[tuple[str, int, int, int]],
                affinity_mode: str, llm_core_count: int) -> bool:
    for interference_count in INTERFERENCE_CORE_COUNTS:
        for repeat in range(REPEATS):
            if (affinity_mode, llm_core_count, interference_count,
                    repeat) not in completed_keys:
                return True
    return False


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    compile_burner()
    infos = parse_lscpu()
    prompt = make_prompt(PROMPT_LEN)

    existing_rows = load_existing_rows()
    results: dict[str, Any] = {
        "timestamp_utc": now_utc(),
        "model": str(MODEL),
        "prompt_len": PROMPT_LEN,
        "output_len": OUTPUT_LEN,
        "batch_size": 1,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_model_len": MAX_MODEL_LEN,
        "port": PORT,
        "llm_core_counts": LLM_CORE_COUNTS,
        "interference_core_counts": INTERFERENCE_CORE_COUNTS,
        "affinity_modes": AFFINITY_MODES,
        "repeats": REPEATS,
        "cpu_topology": [info.__dict__ for info in infos],
        "rows": existing_rows,
    }
    completed_keys = {row_key(row) for row in existing_rows}
    if existing_rows:
        print(f"[{now_utc()}] loaded {len(existing_rows)} existing rows", flush=True)
        save_progress(results)

    for affinity_mode in AFFINITY_MODES:
        for llm_core_count in LLM_CORE_COUNTS:
            while has_pending(completed_keys, affinity_mode, llm_core_count):
                run_one_server_matrix(
                    prompt,
                    infos,
                    affinity_mode,
                    llm_core_count,
                    completed_keys,
                    results,
                )
            print(
                f"[{now_utc()}] complete {affinity_mode}_llm{llm_core_count}",
                flush=True,
            )

    results["aggregated"] = aggregate(results["rows"])
    results["pdfs"] = make_pdf(results)
    RESULT_JSON.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(results["rows"])
    make_markdown(results)
    append_iteration_record(results)
    print(f"[{now_utc()}] wrote {RESULT_JSON}", flush=True)
    print(f"[{now_utc()}] wrote {RESULT_CSV}", flush=True)
    print(f"[{now_utc()}] wrote {RESULT_MD}", flush=True)
    for path in results["pdfs"]:
        print(f"[{now_utc()}] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
