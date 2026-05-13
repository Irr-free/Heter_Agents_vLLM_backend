"""
端到端 correctness 基线：

1. 启动禁用异构 CPU Attention 的 vLLM 服务，得到纯 GPU baseline。
2. 启动默认异构服务，得到 CPU decode attention 输出。
3. 比较 deterministic greedy 输出文本和 token usage。

该脚本只在子进程环境中设置 no_proxy 与 VLLM_HETER_DISABLE_CPU_ATTENTION，
不会修改持久系统环境变量。
"""

import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

MODEL_PATH = "/home/zb/data/model_data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
VLLM_BIN = "/home/zb/tool/miniconda3/envs/vLLM_GPU/bin/vllm"
PORT = 8001
BASE_URL = f"http://localhost:{PORT}/v1"
TIMEOUT_START = 600
TIMEOUT_REQ = 300
LOG_DIR = Path(__file__).parent

CHAT_CASES = [
    {
        "name": "short_math",
        "messages": [{"role": "user", "content": "Hello, what is 2+2?"}],
        "max_tokens": 12,
    },
    {
        "name": "short_paris",
        "messages": [
            {"role": "user", "content": "Write one short sentence about Paris."}
        ],
        "max_tokens": 12,
    },
    {
        "name": "long_prompt_summary",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Read the following notes and answer with one concise sentence. "
                    + "The system has a GPU prefill stage and a CPU decode attention "
                    + "stage. KV cache is copied to CPU memory after prefill, and each "
                    + "new decode KV entry is also copied immediately. "
                    + "The goal is to preserve paged attention behavior while testing "
                    + "CPU-GPU heterogeneous execution. "
                )
                * 8,
            }
        ],
        "max_tokens": 16,
    },
    {
        "name": "multi_turn_chat",
        "messages": [
            {"role": "system", "content": "Answer briefly and deterministically."},
            {"role": "user", "content": "Name one primary color."},
            {"role": "assistant", "content": "Red."},
            {"role": "user", "content": "Now name one different primary color."},
        ],
        "max_tokens": 8,
    },
]

COMPLETION_CASES = [
    {
        "name": "completion_logprobs_short",
        "prompt": "The capital of France is",
        "max_tokens": 6,
    },
    {
        "name": "completion_logprobs_long",
        "prompt": (
            "In a heterogeneous inference system, GPU prefill creates the KV cache, "
            "then CPU decode attention reads that cache. The next generated word is"
        ),
        "max_tokens": 8,
    },
]

CONCURRENT_CHAT_CASES = [
    {
        "name": "concurrent_math",
        "messages": [{"role": "user", "content": "What is 3+5? Answer briefly."}],
        "max_tokens": 8,
    },
    {
        "name": "concurrent_city",
        "messages": [{"role": "user", "content": "Name the capital of Italy."}],
        "max_tokens": 8,
    },
]


def wait_for_server(timeout: int = TIMEOUT_START) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            resp = requests.get(f"{BASE_URL}/models", timeout=5)
            if resp.status_code == 200:
                return
        except Exception as exc:
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"服务启动超时，last_error={last_error}")


def start_server(name: str, disable_cpu_attention: bool) -> tuple[subprocess.Popen, Path]:
    log_path = LOG_DIR / f"correctness_{name}.log"
    env = os.environ.copy()
    env["no_proxy"] = "localhost,127.0.0.1"
    env["NO_PROXY"] = "localhost,127.0.0.1"
    if disable_cpu_attention:
        env["VLLM_HETER_DISABLE_CPU_ATTENTION"] = "1"
    else:
        env.pop("VLLM_HETER_DISABLE_CPU_ATTENTION", None)

    cmd = [
        VLLM_BIN,
        "serve",
        MODEL_PATH,
        "--port",
        str(PORT),
        "--max-num-seqs",
        "4",
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


def request_chat_case(case: dict) -> dict:
    payload = {
        "model": MODEL_PATH,
        "messages": case["messages"],
        "max_tokens": case["max_tokens"],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
    }
    resp = requests.post(
        f"{BASE_URL}/chat/completions", json=payload, timeout=TIMEOUT_REQ
    )
    resp.raise_for_status()
    data = resp.json()
    return {
        "name": case["name"],
        "kind": "chat",
        "messages": case["messages"],
        "content": data["choices"][0]["message"]["content"],
        "finish_reason": data["choices"][0]["finish_reason"],
        "usage": data["usage"],
    }


def request_completion_case(case: dict) -> dict:
    payload = {
        "model": MODEL_PATH,
        "prompt": case["prompt"],
        "max_tokens": case["max_tokens"],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "logprobs": 5,
        "return_token_ids": True,
    }
    resp = requests.post(f"{BASE_URL}/completions", json=payload, timeout=TIMEOUT_REQ)
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    logprobs = choice.get("logprobs") or {}
    return {
        "name": case["name"],
        "kind": "completion",
        "prompt": case["prompt"],
        "text": choice["text"],
        "finish_reason": choice["finish_reason"],
        "token_ids": choice.get("token_ids"),
        "tokens": logprobs.get("tokens"),
        "token_logprobs": logprobs.get("token_logprobs"),
        "top_logprobs": logprobs.get("top_logprobs"),
        "usage": data["usage"],
    }


def request_concurrent_cases() -> list[dict]:
    with ThreadPoolExecutor(max_workers=len(CONCURRENT_CHAT_CASES)) as executor:
        return list(executor.map(request_chat_case, CONCURRENT_CHAT_CASES))


def run_suite(name: str, disable_cpu_attention: bool) -> tuple[dict, Path]:
    proc = None
    try:
        proc, log_path = start_server(name, disable_cpu_attention)
        results = {
            "chat": [request_chat_case(case) for case in CHAT_CASES],
            "completion": [
                request_completion_case(case) for case in COMPLETION_CASES
            ],
            "concurrent_chat": request_concurrent_cases(),
        }
        return results, log_path
    finally:
        if proc is not None:
            stop_server(proc)


def max_float_list_abs_diff(lhs, rhs) -> float:
    assert len(lhs) == len(rhs), (lhs, rhs)
    max_diff = 0.0
    for left, right in zip(lhs, rhs):
        if left is None or right is None:
            assert left is right
        else:
            max_diff = max(max_diff, abs(left - right))
    return max_diff


def compare_chat_items(base_item: dict, heter_item: dict) -> None:
    assert base_item["name"] == heter_item["name"]
    assert base_item["content"] == heter_item["content"], (base_item, heter_item)
    assert base_item["finish_reason"] == heter_item["finish_reason"]
    assert base_item["usage"]["completion_tokens"] == heter_item["usage"][
        "completion_tokens"
    ]
    assert base_item["usage"]["prompt_tokens"] == heter_item["usage"]["prompt_tokens"]


def compare_completion_items(base_item: dict, heter_item: dict) -> dict:
    assert base_item["name"] == heter_item["name"]
    assert base_item["text"] == heter_item["text"], (base_item, heter_item)
    assert base_item["finish_reason"] == heter_item["finish_reason"]
    assert base_item["usage"]["completion_tokens"] == heter_item["usage"][
        "completion_tokens"
    ]
    assert base_item["token_ids"] == heter_item["token_ids"]
    assert base_item["tokens"] == heter_item["tokens"]
    max_logprob_abs_diff = max_float_list_abs_diff(
        base_item["token_logprobs"], heter_item["token_logprobs"]
    )
    assert max_logprob_abs_diff <= 0.1, (base_item, heter_item, max_logprob_abs_diff)
    return {
        "name": base_item["name"],
        "max_token_logprob_abs_diff": max_logprob_abs_diff,
    }


def main() -> int:
    baseline, baseline_log = run_suite("gpu_baseline", disable_cpu_attention=True)
    heter, heter_log = run_suite("heter_cpu_decode", disable_cpu_attention=False)

    summary = {
        "completion_logprob_diffs": [],
    }
    report = {
        "baseline_log": str(baseline_log),
        "heter_log": str(heter_log),
        "baseline": baseline,
        "heter": heter,
        "summary": summary,
    }

    for key in ("chat", "concurrent_chat"):
        for base_item, heter_item in zip(baseline[key], heter[key]):
            compare_chat_items(base_item, heter_item)

    for base_item, heter_item in zip(baseline["completion"], heter["completion"]):
        summary["completion_logprob_diffs"].append(
            compare_completion_items(base_item, heter_item)
        )

    heter_log_text = heter_log.read_text(encoding="utf-8", errors="replace")
    assert "query_shape=torch.Size([1, 32, 128])" in heter_log_text
    assert "query_shape=torch.Size([44, 32, 128])" not in heter_log_text
    summary["decode_cpu_fallback_q1_count"] = heter_log_text.count(
        "query_shape=torch.Size([1, 32, 128])"
    )
    summary["decode_cpu_fallback_q2_count"] = heter_log_text.count(
        "query_shape=torch.Size([2, 32, 128])"
    )
    summary["prefill_cpu_fallback_q44_count"] = heter_log_text.count(
        "query_shape=torch.Size([44, 32, 128])"
    )

    report_path = LOG_DIR / "correctness_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("端到端 correctness 基线通过")
    print(f"报告: {report_path}")
    print(f"GPU baseline log: {baseline_log}")
    print(f"Heter log: {heter_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
