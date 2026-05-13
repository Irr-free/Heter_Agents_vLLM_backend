"""
vLLM 原生 C++ CPU attention 接入测试。

该测试直接加载轻量 C++ 扩展，并比较原生 CPU attention 与 dense
PyTorch reference 的 decode 输出。测试不启动 vLLM 服务，不访问 GPU。
"""

import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def gather_sequence(
    token_table: torch.Tensor,
    blocks: list[int],
    block_size: int,
    seq_len: int,
) -> torch.Tensor:
    slots = []
    for block in blocks:
        start = block * block_size
        slots.extend(range(start, start + block_size))
    return token_table[slots[:seq_len]].transpose(0, 1).contiguous()


def dense_reference(
    query: torch.Tensor,
    key_table: torch.Tensor,
    value_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    scale: float,
    block_size: int,
) -> torch.Tensor:
    num_heads = query.shape[1]
    num_kv_heads = key_table.shape[1]
    num_queries_per_kv = num_heads // num_kv_heads
    output = torch.empty_like(query)

    for seq_idx in range(seq_lens.numel()):
        q_start = int(query_start_loc[seq_idx].item())
        q_end = int(query_start_loc[seq_idx + 1].item())
        seq_len = int(seq_lens[seq_idx].item())
        blocks = [int(x.item()) for x in block_table[seq_idx] if int(x.item()) >= 0]

        k_seq = gather_sequence(key_table, blocks, block_size, seq_len)
        v_seq = gather_sequence(value_table, blocks, block_size, seq_len)
        k_seq = k_seq.repeat_interleave(num_queries_per_kv, dim=0)
        v_seq = v_seq.repeat_interleave(num_queries_per_kv, dim=0)
        q_seq = query[q_start:q_end]

        scores = torch.einsum("qhd,hkd->qhk", q_seq, k_seq) * scale
        probs = torch.softmax(scores, dim=-1)
        output[q_start:q_end] = torch.einsum("qhk,hkd->qhd", probs, v_seq)

    return output


def load_native_ops():
    lib_path = Path(
        os.environ.get(
            "VLLM_HETER_CPU_ATTN_LIB",
            REPO_ROOT / "vllm/libs/libvllm_heter_cpu_attn.so",
        )
    )
    assert lib_path.exists(), (
        f"未找到 C++ 扩展: {lib_path}. "
        "请先运行 CPU_GPU_Heter_test/build_heter_cpu_attn.sh"
    )
    torch.ops.load_library(str(lib_path))
    assert hasattr(torch.ops, "_heter_cpu_attn_C")
    return torch.ops._heter_cpu_attn_C


def test_native_cpp_decode_matches_dense_reference():
    torch.manual_seed(0)
    ops = load_native_ops()

    num_blocks = 4
    block_size = 32
    num_kv_heads = 2
    num_heads = 4
    head_size = 32
    scale = head_size**-0.5

    key_table = torch.randn(num_blocks * block_size, num_kv_heads, head_size)
    value_table = torch.randn(num_blocks * block_size, num_kv_heads, head_size)
    key_cache = torch.empty(num_blocks, num_kv_heads, block_size, head_size)
    value_cache = torch.empty(num_blocks, num_kv_heads, block_size, head_size)
    slot_mapping = torch.arange(num_blocks * block_size, dtype=torch.int64)
    ops.reshape_and_cache(
        key_table,
        value_table,
        key_cache,
        value_cache,
        slot_mapping,
        "vec",
    )

    query = torch.randn(2, num_heads, head_size)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    seq_lens = torch.tensor([5, 7], dtype=torch.int32)
    block_table = torch.tensor([[2, 0, -1], [1, 3, -1]], dtype=torch.int32)

    scheduler_metadata = ops.get_scheduler_metadata(
        int(seq_lens.numel()),
        num_heads,
        num_kv_heads,
        head_size,
        seq_lens,
        query.dtype,
        query_start_loc,
        True,
        -1,
        "vec",
        False,
    )

    output = torch.empty_like(query)
    ops.attention_with_kv_cache(
        query,
        key_cache,
        value_cache,
        output,
        query_start_loc,
        seq_lens,
        scale,
        True,
        None,
        -1,
        -1,
        block_table,
        0.0,
        scheduler_metadata,
        None,
    )

    expected = dense_reference(
        query,
        key_table,
        value_table,
        query_start_loc,
        seq_lens,
        block_table,
        scale,
        block_size,
    )
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def run_all_tests():
    test_native_cpp_decode_matches_dense_reference()
    print("vLLM 原生 C++ CPU attention 接入测试通过")


if __name__ == "__main__":
    run_all_tests()
