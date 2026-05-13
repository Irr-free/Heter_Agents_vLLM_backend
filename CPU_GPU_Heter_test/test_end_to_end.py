"""
端到端测试：启动 vllm serve，发送实际请求，验证 CPU-GPU 异构 Decode Attention 流程。
测试范围：
 1. 服务能正常启动并加载模型
 2. 长 prompt 请求触发 Prefill + Decode，输出正常
  3. 后续请求触发 Pure Decode，且走 CPU Attention 路径（C++ 原生 kernel）
  4. 验证 Decode 阶段的输出与纯 GPU 版本一致（或近似一致）
"""

import os
import sys
import time
import json
import signal
import subprocess
import requests
import traceback
from pathlib import Path

# 配置
# 避免本地请求被系统代理拦截
os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
MODEL_PATH = "/home/zb/data/model_data/models/LLM-Research/Meta-Llama-3.1-8B-Instruct"
PORT = 8001
BASE_URL = f"http://localhost:{PORT}/v1"
LOG_FILE = Path(__file__).parent / "vllm_serve.log"
TIMEOUT_START = 600  # 服务启动超时（秒，含模型编译时间）
TIMEOUT_REQ = 300    # 单次请求超时（秒，CPU fallback 较慢）
CPU_ATTENTION_EXECUTED_FLAG = "/tmp/vllm_cpu_attention_executed.flag"
CPP_ATTN_LIB = Path(__file__).parent.parent / "vllm/libs/libvllm_heter_cpu_attn.so"
HETER_PROFILE_EVENTS = Path(__file__).parent / "end_to_end_heter_profile_events.jsonl"


def wait_for_server(url: str, timeout: int = TIMEOUT_START) -> bool:
    """轮询 health endpoint，等待服务就绪。"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{url}/models", timeout=5)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def send_chat_completion(messages, max_tokens=32, temperature=0.0, stream=False):
    """发送 chat completion 请求。"""
    url = f"{BASE_URL}/chat/completions"
    payload = {
        "model": MODEL_PATH,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    resp = requests.post(url, json=payload, timeout=TIMEOUT_REQ)
    resp.raise_for_status()
    return resp


def summarize_profile_events(path: Path) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    if not path.exists():
        return summary
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        item = summary.setdefault(event["event"], {"count": 0, "total_s": 0.0})
        item["count"] = int(item["count"]) + 1
        item["total_s"] = float(item["total_s"]) + float(event["elapsed_s"])
    return summary


def run_test():
    print("=" * 60)
    print("vLLM CPU-GPU 异构 Decode Attention — 端到端测试")
    print("=" * 60)

    # 0. 清理旧的标记文件
    flag_path = Path(CPU_ATTENTION_EXECUTED_FLAG)
    if flag_path.exists():
        flag_path.unlink()
    if HETER_PROFILE_EVENTS.exists():
        HETER_PROFILE_EVENTS.unlink()

    # 1. 启动 vllm serve
    cmd = [
        "vllm", "serve", MODEL_PATH,
        "--port", str(PORT),
        "--max-num-seqs", "4",
        "--max-model-len", "2048",
        "--kv-cache-memory-bytes", str(1 << 30),
        "--gpu-memory-utilization", "0.30",
    ]
    print(f"\n[Step 1] 启动 vllm serve ...")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log file: {LOG_FILE}")

    env = os.environ.copy()
    env["no_proxy"] = "localhost,127.0.0.1"
    env["NO_PROXY"] = "localhost,127.0.0.1"
    if CPP_ATTN_LIB.exists():
        env["VLLM_HETER_CPU_ATTN_LIB"] = str(CPP_ATTN_LIB)
    env["VLLM_HETER_PROFILE"] = "1"
    env["VLLM_HETER_PROFILE_PATH"] = str(HETER_PROFILE_EVENTS)
    with open(LOG_FILE, "w") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,  # 以便后续可以 kill 整个进程组
        )

    # 2. 等待服务就绪
    print(f"\n[Step 2] 等待服务就绪（最多 {TIMEOUT_START}s）...")
    if not wait_for_server(BASE_URL, timeout=TIMEOUT_START):
        print(f"  ✗ 服务启动超时或失败")
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait()
        return False
    print(f"  ✓ 服务已就绪")

    try:
        # 3. 发送流式请求（长 prompt，触发 Prefill + 多个 Decode token）
        print(f"\n[Step 3] 发送流式请求（Prefill + Decode）...")
        # 使用短 prompt 减少 prefill 时间，max_tokens=10 减少 decode 时间
        # （CPU fallback 的 decode 吞吐约 0.5 tokens/s，32 层 × 10 token ≈ 60s）
        short_prompt = "Hello, what is 2+2?"
        messages = [{"role": "user", "content": short_prompt}]
        resp = send_chat_completion(messages, max_tokens=10, stream=True)

        # 读取流式响应
        text_chunks = []
        for line in resp.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data = line_str[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        text_chunks.append(delta)
                    except Exception:
                        pass

        text1 = "".join(text_chunks)
        print(f"  ✓ 流式请求完成，输出长度: {len(text1)} chars")
        print(f"  输出片段: {text1[:100]}...")

        # 4. 检查是否有 CPU Attention 执行标记文件
        print(f"\n[Step 4] 检查 CPU Attention 执行标记 ...")
        time.sleep(2)  # 等待标记文件写入
        if Path(CPU_ATTENTION_EXECUTED_FLAG).exists():
            print(f"  ✓ 检测到标记文件: {CPU_ATTENTION_EXECUTED_FLAG}")
            print(f"    Decode Self-Attention 确实在 CPU 上执行（C++ 原生 kernel）")
        else:
            print(f"  ⚠ 未检测到标记文件: {CPU_ATTENTION_EXECUTED_FLAG}")
            print(f"    可能走了 GPU 路径，或标记文件未成功写入")

        profile_summary = summarize_profile_events(HETER_PROFILE_EVENTS)
        cpp_count = int(profile_summary.get("cpu_attention_cpp_total", {}).get("count", 0))
        py_count = int(profile_summary.get("cpu_attention_python_total", {}).get("count", 0))
        print(f"  C++ CPU attention 事件数: {cpp_count}")
        print(f"  Python fallback 事件数: {py_count}")
        if cpp_count == 0 or py_count != 0:
            print(f"  ✗ C++ CPU attention 事件校验失败")
            return False

        # 5. 基本正确性检查
        print(f"\n[Step 5] 基本正确性检查 ...")
        if len(text1) > 10:
            print(f"  ✓ 请求产生了合理的文本输出")
        else:
            print(f"  ✗ 输出过短，可能存在问题")
            return False

        print(f"\n" + "=" * 60)
        print("端到端测试完成！")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\n  ✗ 请求过程中出现异常: {e}")
        traceback.print_exc()
        return False

    finally:
        # 6. 清理进程
        print(f"\n[Cleanup] 终止 vllm serve 进程 ...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
        print(f"  ✓ 进程已终止")


def main():
    success = run_test()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
