"""
CPU-only 路由测试：验证 CPU-GPU 异构 attention 的 Prefill/Decode 判定。

这些测试不启动服务、不访问 GPU，重点防止 FlashAttentionMetadata 缺少
num_prefill_tokens 时被误判为 pure decode。
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

sys.path.insert(0, "/home/zb/data/HeterAgents/vLLM server/vllm")

from vllm.model_executor.layers.attention.attention import (
    _heter_is_pure_decode,
    _heter_should_sync_full_kv_to_cpu,
    unified_attention_with_output,
)


def make_meta(query_lens, seq_lens, max_query_len=None):
    query_start = [0]
    for query_len in query_lens:
        query_start.append(query_start[-1] + query_len)
    return SimpleNamespace(
        query_start_loc=torch.tensor(query_start, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        max_query_len=max(query_lens) if max_query_len is None else max_query_len,
    )


def test_prefill_not_decode_without_num_prefill_tokens():
    meta = make_meta(query_lens=[44], seq_lens=[44])

    assert not _heter_is_pure_decode(meta)
    assert _heter_should_sync_full_kv_to_cpu(meta)


def test_single_token_prefill_not_decode():
    meta = make_meta(query_lens=[1], seq_lens=[1], max_query_len=1)

    assert not _heter_is_pure_decode(meta)
    assert _heter_should_sync_full_kv_to_cpu(meta)


def test_single_token_decode_uses_cpu_path():
    meta = make_meta(query_lens=[1], seq_lens=[45], max_query_len=1)

    assert _heter_is_pure_decode(meta)
    assert not _heter_should_sync_full_kv_to_cpu(meta)


def test_mixed_prefill_decode_is_not_pure_decode():
    meta = make_meta(query_lens=[1, 44], seq_lens=[45, 44], max_query_len=44)

    assert not _heter_is_pure_decode(meta)
    assert _heter_should_sync_full_kv_to_cpu(meta)


def test_prefill_d2h_after_gpu_attention():
    meta = make_meta(query_lens=[44], seq_lens=[44])
    events = []
    attn = SimpleNamespace(
        cpu_kv_cache=torch.empty(1),
        impl=SimpleNamespace(
            forward=lambda *args, **kwargs: events.append("gpu_attention")
        ),
        _prefill_kv_to_cpu_async=lambda: events.append("d2h"),
    )
    query = torch.empty((44, 2, 4))
    key = torch.empty((44, 1, 4))
    value = torch.empty((44, 1, 4))
    output = torch.empty((44, 2, 4))
    kv_cache = torch.empty((2, 1, 16, 1, 4))

    with patch(
        "vllm.model_executor.layers.attention.attention.get_attention_context",
        MagicMock(return_value=(meta, attn, kv_cache, None)),
    ), patch(
        "vllm.model_executor.layers.attention.attention.is_cudagraph_capturing",
        MagicMock(return_value=False),
    ):
        unified_attention_with_output(query, key, value, output, "")

    assert events == ["gpu_attention", "d2h"]


def run_all_tests():
    tests = [
        test_prefill_not_decode_without_num_prefill_tokens,
        test_single_token_prefill_not_decode,
        test_single_token_decode_uses_cpu_path,
        test_mixed_prefill_decode_is_not_pure_decode,
        test_prefill_d2h_after_gpu_attention,
    ]
    for test in tests:
        test()
        print(f"  通过: {test.__name__}")
    print("异构 Decode 路由测试全部通过")


if __name__ == "__main__":
    run_all_tests()
