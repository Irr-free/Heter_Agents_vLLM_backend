"""
CPU Attention fallback 数值基线测试。

不启动 vLLM 服务，直接构造 paged KV cache，比较 Python CPU fallback
与 dense PyTorch reference attention 的输出。
"""

import sys

import torch

sys.path.insert(0, "/home/zb/data/HeterAgents/vLLM server/vllm")

# 导入 attention 模块以注册 cpu_paged_attention_fallback custom op。
import vllm.model_executor.layers.attention.attention  # noqa: F401


def gather_sequence(cache: torch.Tensor, blocks: list[int], seq_len: int) -> torch.Tensor:
    pieces = [cache[block] for block in blocks]
    return torch.cat(pieces, dim=1)[:, :seq_len, :]


def dense_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    num_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    num_queries_per_kv = num_heads // num_kv_heads
    output = torch.empty_like(query)

    for seq_idx in range(seq_lens.numel()):
        q_start = int(query_start_loc[seq_idx].item())
        q_end = int(query_start_loc[seq_idx + 1].item())
        seq_len = int(seq_lens[seq_idx].item())
        blocks = [int(x.item()) for x in block_table[seq_idx] if int(x.item()) >= 0]

        k_seq = gather_sequence(key_cache, blocks, seq_len)
        v_seq = gather_sequence(value_cache, blocks, seq_len)
        k_seq = k_seq.repeat_interleave(num_queries_per_kv, dim=0)
        v_seq = v_seq.repeat_interleave(num_queries_per_kv, dim=0)
        q_seq = query[q_start:q_end]

        scores = torch.einsum("qhd,hkd->qhk", q_seq, k_seq) * scale
        probs = torch.softmax(scores, dim=-1)
        output[q_start:q_end] = torch.einsum("qhk,hkd->qhd", probs, v_seq)

    return output


def test_decode_matches_dense_reference():
    torch.manual_seed(0)

    num_blocks = 4
    block_size = 4
    num_kv_heads = 2
    num_heads = 4
    head_size = 8
    scale = head_size ** -0.5

    key_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_size)
    value_cache = torch.randn(num_blocks, num_kv_heads, block_size, head_size)
    query = torch.randn(2, num_heads, head_size)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    seq_lens = torch.tensor([5, 7], dtype=torch.int32)
    block_table = torch.tensor([[2, 0, -1], [1, 3, -1]], dtype=torch.int32)

    output = torch.empty_like(query)
    torch.ops.vllm.cpu_paged_attention_fallback(
        query,
        key_cache,
        value_cache,
        output,
        query_start_loc,
        seq_lens,
        scale,
        True,
        torch.empty(0),
        -1,
        -1,
        block_table,
        0.0,
        torch.empty(0),
        torch.empty(0),
    )

    expected = dense_reference(
        query, key_cache, value_cache, query_start_loc, seq_lens, block_table, scale
    )
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


def run_all_tests():
    test_decode_matches_dense_reference()
    print("CPU Attention fallback 数值基线测试通过")


if __name__ == "__main__":
    run_all_tests()
