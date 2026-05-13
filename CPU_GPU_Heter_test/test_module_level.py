"""
模块级测试：验证 CPU-GPU 异构 Decode Attention 的核心组件是否正常工作。
测试范围：
  1. Attention 类能正常实例化，且包含 CPU KV Cache 相关成员
  2. init_cpu_kv_cache() 正确分配 CPU pinned memory
  3. forward() 在 Prefill / Decode / Graph Capture 条件下正确选择路径
"""

import sys
import os
import traceback

# 将 vllm 源码加入 Python path
sys.path.insert(0, "/home/zb/data/HeterAgents/vLLM server/vllm")

import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch, PropertyMock


def test_attention_import():
    """测试 Attention 类能正常导入且包含异构相关成员"""
    print("[Test 1] 验证 Attention 类导入与成员...")
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        attn = Attention(
            num_heads=8,
            head_size=128,
            scale=128 ** -0.5,
            num_kv_heads=4,
            alibi_slopes=None,
            logits_soft_cap=0.0,
            attn_type="decoder",
        )

    # 检查关键成员是否存在
    assert hasattr(attn, 'cpu_kv_cache'), "缺少 cpu_kv_cache 成员"
    assert hasattr(attn, '_d2h_stream'), "缺少 _d2h_stream 成员"
    assert hasattr(attn, 'init_cpu_kv_cache'), "缺少 init_cpu_kv_cache 方法"
    assert hasattr(attn, '_prefill_kv_to_cpu_async'), "缺少 _prefill_kv_to_cpu_async 方法"
    assert hasattr(attn, '_decode_kv_to_cpu'), "缺少 _decode_kv_to_cpu 方法"
    assert hasattr(attn, '_cpu_paged_attention'), "缺少 _cpu_paged_attention 方法"

    print("  ✓ Attention 类导入成功，所有异构成员/方法存在")
    return True


def test_init_cpu_kv_cache():
    """测试 init_cpu_kv_cache 正确分配 CPU pinned memory"""
    print("[Test 2] 验证 CPU KV Cache 分配...")
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        attn = Attention(
            num_heads=8, head_size=128, scale=128 ** -0.5,
            num_kv_heads=4, alibi_slopes=None,
            logits_soft_cap=0.0,
            attn_type="decoder",
        )

    # 模拟绑定 GPU KV Cache（Layout: [2, num_blocks, block_size, num_kv_heads, head_size]）
    num_blocks = 10
    block_size = 16
    num_kv_heads = 4
    head_size = 128

    gpu_kv_cache = torch.zeros(
        2, num_blocks, block_size, num_kv_heads, head_size,
        dtype=torch.float16, device="cuda:0"
    )
    attn.kv_cache = gpu_kv_cache

    # 初始化 CPU KV Cache
    attn.init_cpu_kv_cache()

    assert attn.cpu_kv_cache is not None, "cpu_kv_cache 未分配"
    assert attn.cpu_kv_cache.device.type == "cpu", "cpu_kv_cache 不在 CPU 上"
    assert attn.cpu_kv_cache.is_pinned(), "cpu_kv_cache 不是 pinned memory"

    # 检查 layout 是否为 [2, num_blocks, num_kv_heads, block_size, head_size]
    expected_shape = (2, num_blocks, num_kv_heads, block_size, head_size)
    assert attn.cpu_kv_cache.shape == expected_shape, \
        f"CPU KV Cache shape 错误: {attn.cpu_kv_cache.shape} != {expected_shape}"

    print(f"  ✓ CPU KV Cache 分配成功: shape={attn.cpu_kv_cache.shape}, dtype={attn.cpu_kv_cache.dtype}")
    return True


def test_forward_path_selection():
    """测试 forward() 在不同条件下正确选择路径"""
    print("[Test 3] 验证 forward() 路径选择...")
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        attn = Attention(
            num_heads=8, head_size=128, scale=128 ** -0.5,
            num_kv_heads=4, alibi_slopes=None,
            logits_soft_cap=0.0,
            attn_type="decoder",
        )

    # 绑定 GPU KV Cache
    num_blocks = 10
    block_size = 16
    num_kv_heads = 4
    head_size = 128
    gpu_kv_cache = torch.zeros(
        2, num_blocks, block_size, num_kv_heads, head_size,
        dtype=torch.float16, device="cuda:0"
    )
    attn.kv_cache = gpu_kv_cache
    attn.init_cpu_kv_cache()

    batch_size = 2
    seq_len = 32
    num_heads = 8
    query = torch.randn(batch_size * seq_len, num_heads, head_size, dtype=torch.float16, device="cuda:0")
    key = torch.randn(batch_size * seq_len, num_kv_heads, head_size, dtype=torch.float16, device="cuda:0")
    value = torch.randn(batch_size * seq_len, num_kv_heads, head_size, dtype=torch.float16, device="cuda:0")
    key_cache = torch.zeros(2, num_blocks, block_size, num_kv_heads, head_size, dtype=torch.float16, device="cuda:0")
    value_cache = torch.zeros(2, num_blocks, block_size, num_kv_heads, head_size, dtype=torch.float16, device="cuda:0")
    output = torch.empty_like(query)
    slot_mapping = torch.zeros(batch_size * seq_len, dtype=torch.long, device="cuda:0")
    sliding_window_blocks = 4096 // block_size

    # 构造 mock 的 attn_metadata（FlashAttentionMetadata 风格，无 num_prefill_tokens）
    mock_attn_meta = MagicMock()
    mock_attn_meta.query_start_loc = torch.tensor(
        [0, batch_size * seq_len], dtype=torch.int32, device="cuda:0")
    mock_attn_meta.seq_lens = torch.tensor(
        [batch_size * seq_len], dtype=torch.int32, device="cuda:0")
    mock_attn_meta.max_query_len = batch_size * seq_len
    mock_attn_meta.use_cascade = False
    mock_attn_meta.decode_cascade_uses_spec = False
    mock_attn_meta.slot_mapping = None

    # 构造 mock 的 forward context
    mock_forward_ctx = MagicMock()
    mock_forward_ctx.attn_metadata = mock_attn_meta
    mock_forward_ctx.slot_mapping = {"": None}  # slot_mapping 必须是 dict

    # 使用 patch 替换 get_forward_context
    from vllm.model_executor.layers.attention import attention as attn_module

    # 需要 mock 底层 KV cache update 和 attention ops，因为它们依赖完整的 runner 上下文
    # 直接 patch torch.ops.vllm 的 custom ops
    mock_kv_update = MagicMock(return_value=None)
    mock_attn_output = MagicMock(return_value=None)
    with patch.object(attn_module, 'get_forward_context', return_value=mock_forward_ctx):
        with patch('torch.ops.vllm.unified_kv_cache_update', mock_kv_update), \
             patch('torch.ops.vllm.unified_attention_with_output', mock_attn_output):

            # 分支 1：Graph Capture 阶段（强制 GPU 路径）
            with patch('torch.cuda.is_current_stream_capturing', return_value=True):
                try:
                    attn.forward(query, key, value)
                    print("  ✓ Graph Capture 阶段 forward 成功（GPU 路径）")
                except Exception as e:
                    print(f"  ✗ Graph Capture 阶段 forward 失败: {e}")
                    traceback.print_exc()
                    return False

            # 分支 2：非 Graph Capture + 有 prefill token（GPU 路径）
            mock_attn_meta.query_start_loc = torch.tensor(
                [0, batch_size * seq_len], dtype=torch.int32, device="cuda:0")
            mock_attn_meta.seq_lens = torch.tensor(
                [batch_size * seq_len], dtype=torch.int32, device="cuda:0")
            mock_attn_meta.max_query_len = batch_size * seq_len
            try:
                attn.forward(query, key, value)
                print("  ✓ Prefill 阶段 forward 成功（GPU 路径）")
            except Exception as e:
                print(f"  ✗ Prefill 阶段 forward 失败: {e}")
                traceback.print_exc()
                return False

        # 分支 3：非 Graph Capture + 无 prefill token（Decode CPU 路径）
        decode_tokens = batch_size
        decode_query = torch.randn(
            decode_tokens, num_heads, head_size,
            dtype=torch.float16, device="cuda:0")
        decode_key = torch.randn(
            decode_tokens, num_kv_heads, head_size,
            dtype=torch.float16, device="cuda:0")
        decode_value = torch.randn(
            decode_tokens, num_kv_heads, head_size,
            dtype=torch.float16, device="cuda:0")
        mock_attn_meta.query_start_loc = torch.tensor(
            [0, 1, 2], dtype=torch.int32, device="cuda:0")
        mock_attn_meta.seq_lens = torch.tensor(
            [seq_len + 1, seq_len + 1], dtype=torch.int32, device="cuda:0")
        mock_attn_meta.max_query_len = 1

        # 在 unified_attention_with_output custom op 内部会通过 get_attention_context
        # 获取 self，因此需要 mock 该函数返回真实的 attn 对象
        cpu_attn_called = [False]
        def mock_cpu_paged_attention(query_cpu, output_cpu, cpu_metadata):
            cpu_attn_called[0] = True
            if output_cpu.shape[0] > 0:
                output_cpu.zero_()

        # 同时 mock _decode_kv_to_cpu 避免真实的 KV 搬运（测试中不需要）
        def mock_decode_kv_to_cpu(key, value, slot_mapping):
            pass

        def mock_get_attention_context(layer_name):
            return mock_attn_meta, attn, attn.kv_cache, None

        # 确保 is_cudagraph_capturing() 返回 False，否则 CPU 路径会被跳过
        # 注意：attention.py 在模块级别导入了 is_cudagraph_capturing，需要 patch 本地引用
        with patch.object(attn_module, 'get_attention_context', side_effect=mock_get_attention_context):
            with patch.object(attn, '_cpu_paged_attention', side_effect=mock_cpu_paged_attention):
                with patch.object(attn, '_decode_kv_to_cpu', side_effect=mock_decode_kv_to_cpu):
                    with patch.object(attn_module, 'is_cudagraph_capturing', return_value=False):
                        try:
                            attn.forward(decode_query, decode_key, decode_value)
                            if cpu_attn_called[0]:
                                print("  ✓ Decode 阶段 forward 成功，且调用了 _cpu_paged_attention（CPU 路径）")
                            else:
                                print("  ✗ Decode 阶段 forward 执行成功，但未调用 _cpu_paged_attention（可能走了 GPU 路径）")
                                return False
                        except Exception as e:
                            print(f"  ✗ Decode 阶段 forward 失败: {e}")
                            traceback.print_exc()
                            return False

    return True


def run_all_tests():
    """运行所有模块级测试"""
    print("=" * 60)
    print("vLLM CPU-GPU 异构 Decode Attention — 模块级测试")
    print("=" * 60)

    results = []
    try:
        results.append(("导入与成员检查", test_attention_import()))
    except Exception as e:
        print(f"  ✗ 导入与成员检查失败: {e}")
        traceback.print_exc()
        results.append(("导入与成员检查", False))

    try:
        results.append(("CPU KV Cache 分配", test_init_cpu_kv_cache()))
    except Exception as e:
        print(f"  ✗ CPU KV Cache 分配失败: {e}")
        traceback.print_exc()
        results.append(("CPU KV Cache 分配", False))

    try:
        results.append(("forward() 路径选择", test_forward_path_selection()))
    except Exception as e:
        print(f"  ✗ forward() 路径选择失败: {e}")
        traceback.print_exc()
        results.append(("forward() 路径选择", False))

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    for name, ok in results:
        status = "✓ 通过" if ok else "✗ 失败"
        print(f"  {name}: {status}")

    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    print(f"\n总计: {passed}/{total} 通过")

    if passed == total:
        print("\n🎉 所有模块级测试通过！")
    else:
        print("\n⚠️ 部分测试失败，请检查输出日志。")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
