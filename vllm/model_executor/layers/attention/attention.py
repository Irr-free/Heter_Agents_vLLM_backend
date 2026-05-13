# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING, Any
import json
import os
import time

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import CacheConfig, get_current_vllm_config
from vllm.config.vllm import VllmConfig
from vllm.compilation.monitor import is_cudagraph_capturing
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionType,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
    get_kv_quant_mode,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention import MLAAttention

logger = init_logger(__name__)


def _heter_cpu_attention_enabled() -> bool:
    return os.environ.get("VLLM_HETER_DISABLE_CPU_ATTENTION", "0") != "1"


def _heter_profile_enabled() -> bool:
    return os.environ.get("VLLM_HETER_PROFILE", "0") == "1"


def _heter_profile_record(
    event: str,
    elapsed_s: float,
    layer_name: str | None = None,
    **fields: Any,
) -> None:
    if not _heter_profile_enabled():
        return
    path = os.environ.get(
        "VLLM_HETER_PROFILE_PATH",
        "/tmp/vllm_heter_profile_events.jsonl",
    )
    record = {
        "event": event,
        "elapsed_s": elapsed_s,
        "pid": os.getpid(),
        "ts": time.time(),
    }
    if layer_name is not None:
        record["layer"] = layer_name
    record.update(fields)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _heter_query_lens(attn_metadata: AttentionMetadata | None) -> torch.Tensor | None:
    """Return per-request query lengths when metadata exposes query_start_loc."""
    if attn_metadata is None:
        return None
    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    if not isinstance(query_start_loc, torch.Tensor) or query_start_loc.numel() < 2:
        return None
    return query_start_loc[1:] - query_start_loc[:-1]


def _heter_is_pure_decode(attn_metadata: AttentionMetadata | None) -> bool:
    """True only when every active request is a real single-token decode."""
    if attn_metadata is None:
        return False

    max_query_len = getattr(attn_metadata, "max_query_len", None)
    if max_query_len is not None and max_query_len != 1:
        return False

    query_lens = _heter_query_lens(attn_metadata)
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    if isinstance(query_lens, torch.Tensor) and isinstance(seq_lens, torch.Tensor):
        seq_lens = seq_lens[: query_lens.shape[0]]
        active = query_lens > 0
        if not bool(torch.any(active).item()):
            return False
        # Decode has a non-empty context before the newly scheduled token.
        return bool(
            torch.all((query_lens[active] == 1) &
                      (seq_lens[active] > query_lens[active])).item()
        )

    num_prefill_tokens = getattr(attn_metadata, "num_prefill_tokens", None)
    num_decode_tokens = getattr(attn_metadata, "num_decode_tokens", None)
    if num_prefill_tokens is not None and num_decode_tokens is not None:
        return num_prefill_tokens == 0 and num_decode_tokens > 0

    return False


def _heter_should_sync_full_kv_to_cpu(
    attn_metadata: AttentionMetadata | None,
) -> bool:
    """Sync full GPU KV cache after any non-pure-decode attention step."""
    if attn_metadata is None:
        return False
    query_lens = _heter_query_lens(attn_metadata)
    if isinstance(query_lens, torch.Tensor):
        return bool(torch.any(query_lens > 0).item()) and not _heter_is_pure_decode(
            attn_metadata
        )

    num_prefill_tokens = getattr(attn_metadata, "num_prefill_tokens", None)
    if num_prefill_tokens is not None:
        return num_prefill_tokens > 0
    return False


def validate_kv_sharing_target(
    current_layer_name, target_layer_name, static_forward_context
):
    error_msg = (
        f"Specified KV sharing target layer for {current_layer_name} "
        f"is not valid: target layer {target_layer_name} "
    )

    if current_layer_name == target_layer_name:
        raise ValueError(error_msg + "cannot be the same as the current layer.")

    if target_layer_name not in static_forward_context:
        from vllm.model_executor.models.utils import extract_layer_index

        # If target layer name is not in the static fwd context, it means either
        # a) the target layer does not come BEFORE the current layer, or
        # b) the target layer is not an Attention layer that exists in the model
        current_layer_idx = extract_layer_index(current_layer_name)
        target_layer_idx = extract_layer_index(target_layer_name)
        if current_layer_idx <= target_layer_idx:
            raise ValueError(error_msg + "must come before the current layer.")
        else:
            raise ValueError(error_msg + "is not a valid Attention layer in the model.")

    # Currently KV sharing is only supported between layers of the same type
    target_layer_attn_type = static_forward_context[target_layer_name].attn_type
    expected = static_forward_context[current_layer_name].attn_type
    if target_layer_attn_type != expected:
        raise ValueError(
            error_msg + f"must be the same type as the current layer ({expected})."
        )


def should_load_quant_weights(quant_method: QuantizeMethodBase | None) -> bool:
    """Returns whether the quantization method should load quantized weights."""
    return quant_method is not None and not isinstance(
        quant_method, UnquantizedLinearMethod
    )


def set_default_quant_scales(layer: nn.Module, register_buffer: bool = False) -> None:
    """Sets default quantization scales for the layer."""
    if register_buffer:
        layer.register_buffer("_k_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_v_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_q_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_prob_scale", torch.tensor(1.0, dtype=torch.float32))
    else:
        layer._k_scale.fill_(1.0)
        layer._v_scale.fill_(1.0)
        layer._q_scale.fill_(1.0)
        layer._prob_scale.fill_(1.0)

    # We also keep q/k/v_scale on host (cpu) memory for attention
    # backends that require the scales to be on host instead of on device.
    # e.g. Flashinfer
    layer._q_scale_float = 1.0
    layer._k_scale_float = 1.0
    layer._v_scale_float = 1.0
    layer._prob_scale_float = 1.0

    # Initialize q/k/v range constants used by calc_kv_scales
    layer.q_range = torch.tensor(envs.Q_SCALE_CONSTANT, dtype=torch.float32)
    layer.k_range = torch.tensor(envs.K_SCALE_CONSTANT, dtype=torch.float32)
    layer.v_range = torch.tensor(envs.V_SCALE_CONSTANT, dtype=torch.float32)


def _init_kv_cache_quant(
    layer: nn.Module,
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> None:
    """Initializes KV cache scaling factors and quantization method.

    This helper function sets up the KV cache quantization attributes that are
    shared between Attention and MLAAttention layers. It initializes scale
    tensors for query, key, value, and probability, and configures the
    quantization method if applicable.

    Args:
        layer: The attention layer instance to initialize.
        quant_config: Optional quantization configuration.
        prefix: Layer name prefix for quantization method lookup.
    """

    # Note [Register q/k/v/prob scales in state dict]
    # When calling model.to(device), only parameters/buffers in state dict are
    # moved. If not registering q/k/v/prob scales in state dict, there would
    # be an IMA error when a cuda kernel (e.g., quant_fp8) accesses the tensor
    # on cpu.
    # Registering in state dict means it interacts with weight loading. One edge
    # case is when quant_method is None, or quant_method is UnquantizedLinearMethod
    # (i.e., should_load_quant_weights(quant_method) == False).
    # In this case, the checkpoint does not have the scales. We need to
    # initialize the scales to 1.0 and update the scales after weight loading.
    # This is espectially important when we load dummy weights first (providing
    # wrong scales) and then load real weights (which misses scales and keeps the
    # wrong scales from dummy load).
    set_default_quant_scales(layer, register_buffer=True)

    # The output scale on host memory. This should be the input scale of
    # the quant op after this attention layer.
    layer._o_scale_float = None

    quant_method = (
        quant_config.get_quant_method(layer, prefix=prefix) if quant_config else None
    )

    # See [Note: Register q/k/v/prob scales in state dict]
    if should_load_quant_weights(quant_method):
        assert isinstance(quant_method, BaseKVCacheMethod)
        # TODO (mgoin): kv cache dtype should be specified in the FP8
        # checkpoint config and become the "auto" behavior
        if layer.kv_cache_dtype == "fp8_e5m2":
            raise ValueError("fp8_e5m2 kv-cache is not supported with fp8 checkpoints.")
        # If quantization is enabled, we make "k_scale" and "v_scale"
        # parameters so that it can be loaded from the model checkpoint.
        # The k/v_scale will then be converted back to native float32
        # values after weight loading.
        layer.quant_method = quant_method
        layer.quant_method.create_weights(layer)


# =============================================================================
# 异构系统：CPU Paged Attention Python Fallback（用于 GPU 构建环境快速验证）
# =============================================================================

def _cpu_attn_reshape_and_cache_fallback(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Python fallback for cpu_attn_reshape_and_cache.
    Layout: [num_blocks, num_kv_heads, block_size, head_size]
    """
    profile_start = time.perf_counter() if _heter_profile_enabled() else None
    block_size = key_cache.shape[2]
    for i, slot in enumerate(slot_mapping):
        slot_val = int(slot.item())
        if slot_val < 0:
            continue
        block_idx = slot_val // block_size
        block_offset = slot_val % block_size
        key_cache[block_idx, :, block_offset, :] = key[i]
        value_cache[block_idx, :, block_offset, :] = value[i]
    if profile_start is not None:
        _heter_profile_record(
            "cpu_decode_kv_cache_write_python",
            time.perf_counter() - profile_start,
            tokens=int(slot_mapping.numel()),
        )


# 全局标记文件路径，用于端到端测试验证 Decode CPU Attention 是否被执行
_CPU_ATTENTION_EXECUTED_FLAG = "/tmp/vllm_cpu_attention_executed.flag"


def _cpu_paged_attention_fallback(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    causal: bool,
    alibi_slopes: torch.Tensor,
    sliding_window_left: int,
    sliding_window_right: int,
    block_table: torch.Tensor,
    softcap: float,
    scheduler_metadata: torch.Tensor,
    s_aux: torch.Tensor,
) -> None:
    """Python fallback for cpu_attention_with_kv_cache.
    简化版 Paged Attention，使用 PyTorch 原生 ops 在 CPU 上执行。
    性能远低于 C++ kernel，但足以验证异构流程的正确性。
    """
    # 创建标记文件，供端到端测试验证 Decode 确实走了 CPU 路径
    try:
        with open(_CPU_ATTENTION_EXECUTED_FLAG, "w") as f:
            f.write("1")
    except Exception:
        pass

    profile_total = time.perf_counter() if _heter_profile_enabled() else None
    profile_collect = 0.0
    profile_repeat = 0.0
    profile_qk = 0.0
    profile_softmax = 0.0
    profile_pv = 0.0

    num_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    head_size = query.shape[2]
    block_size = key_cache.shape[2]
    num_queries_per_kv = num_heads // num_kv_heads
    num_seqs = len(seq_lens)

    for seq_idx in range(num_seqs):
        q_start = int(query_start_loc[seq_idx].item())
        q_end = int(query_start_loc[seq_idx + 1].item())
        seq_len = int(seq_lens[seq_idx].item())

        # 收集该序列的 block
        blocks = block_table[seq_idx]
        valid_blocks = []
        for b in blocks:
            b_val = int(b.item())
            if b_val >= 0:
                valid_blocks.append(b_val)
            else:
                break

        if not valid_blocks:
            continue

        phase_start = time.perf_counter() if profile_total is not None else None
        # 从 block cache 中拼接出该序列的 K/V
        k_list = [key_cache[b] for b in valid_blocks]   # each: [num_kv_heads, block_size, head_size]
        v_list = [value_cache[b] for b in valid_blocks]
        k_seq = torch.cat(k_list, dim=1)[:, :seq_len, :]  # [num_kv_heads, seq_len, head_size]
        v_seq = torch.cat(v_list, dim=1)[:, :seq_len, :]  # [num_kv_heads, seq_len, head_size]
        if phase_start is not None:
            profile_collect += time.perf_counter() - phase_start

        phase_start = time.perf_counter() if profile_total is not None else None
        # GQA: repeat_interleave K/V 到 num_heads
        k_seq = k_seq.repeat_interleave(num_queries_per_kv, dim=0)  # [num_heads, seq_len, head_size]
        v_seq = v_seq.repeat_interleave(num_queries_per_kv, dim=0)  # [num_heads, seq_len, head_size]
        if phase_start is not None:
            profile_repeat += time.perf_counter() - phase_start

        q_seq = query[q_start:q_end]  # [num_query_tokens, num_heads, head_size]

        phase_start = time.perf_counter() if profile_total is not None else None
        # Attention scores: [num_query_tokens, num_heads, seq_len]
        scores = torch.einsum('qhd,hkd->qhk', q_seq, k_seq) * scale
        if phase_start is not None:
            profile_qk += time.perf_counter() - phase_start

        # Causal mask
        # 注意：当 q_seq.shape[0] == 1 时为典型 Decode 阶段，query 是序列最后一个
        # token，应能看到全部历史，因此跳过 causal mask。
        if causal and q_seq.shape[0] > 1:
            global_q_start = q_start
            for qi in range(q_seq.shape[0]):
                global_q_pos = global_q_start + qi
                if global_q_pos < seq_len:
                    scores[qi, :, global_q_pos + 1:] = float('-inf')
                else:
                    scores[qi, :, :] = float('-inf')

        # Sliding window mask
        if sliding_window_left >= 0:
            for qi in range(q_seq.shape[0]):
                global_q_pos = q_start + qi
                left_bound = max(0, global_q_pos - sliding_window_left)
                if left_bound > 0:
                    scores[qi, :, :left_bound] = float('-inf')
        if sliding_window_right >= 0:
            for qi in range(q_seq.shape[0]):
                global_q_pos = q_start + qi
                right_bound = min(seq_len, global_q_pos + sliding_window_right + 1)
                if right_bound < seq_len:
                    scores[qi, :, right_bound:] = float('-inf')

        # Softcap
        if softcap != 0.0:
            scores = softcap * torch.tanh(scores / softcap)

        # ALiBi
        if alibi_slopes.numel() > 0:
            positions = torch.arange(seq_len, dtype=scores.dtype, device=scores.device)
            for h in range(num_heads):
                scores[:, h, :] += alibi_slopes[h].item() * positions.unsqueeze(0)

        phase_start = time.perf_counter() if profile_total is not None else None
        # Softmax
        scores = torch.softmax(scores, dim=-1)
        if phase_start is not None:
            profile_softmax += time.perf_counter() - phase_start

        phase_start = time.perf_counter() if profile_total is not None else None
        # Output: [num_query_tokens, num_heads, head_size]
        out_seq = torch.einsum('qhk,hkd->qhd', scores, v_seq)
        output[q_start:q_end] = out_seq
        if phase_start is not None:
            profile_pv += time.perf_counter() - phase_start

    if profile_total is not None:
        total = time.perf_counter() - profile_total
        _heter_profile_record(
            "cpu_attention_python_total",
            total,
            query_tokens=int(query.shape[0]),
            num_heads=int(num_heads),
            num_kv_heads=int(num_kv_heads),
            head_size=int(head_size),
            num_seqs=int(num_seqs),
        )
        _heter_profile_record("cpu_attention_collect_kv", profile_collect)
        _heter_profile_record("cpu_attention_repeat_kv", profile_repeat)
        _heter_profile_record("cpu_attention_qk", profile_qk)
        _heter_profile_record("cpu_attention_softmax", profile_softmax)
        _heter_profile_record("cpu_attention_pv", profile_pv)


def _cpu_paged_attention_fallback_fake(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    causal: bool,
    alibi_slopes: torch.Tensor,
    sliding_window_left: int,
    sliding_window_right: int,
    block_table: torch.Tensor,
    softcap: float,
    scheduler_metadata: torch.Tensor,
    s_aux: torch.Tensor,
) -> None:
    pass


direct_register_custom_op(
    "cpu_paged_attention_fallback",
    _cpu_paged_attention_fallback,
    mutates_args=["output"],
    fake_impl=_cpu_paged_attention_fallback_fake,
    dispatch_key="CPU",
)


class Attention(nn.Module, AttentionLayerBase):
    """Attention layer.

    This class takes query, key, and value tensors as input. The input tensors
    can either contain prompt tokens or generation tokens.
    The class does the following:

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the output tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        use_alibi_sqrt: bool | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        logits_soft_cap: float | None = None,
        per_layer_sliding_window: int | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        attn_backend: type[AttentionBackend] | None = None,
        head_size_v: int | None = None,
        **extra_impl_args,
    ) -> None:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.
        """
        super().__init__()
        sliding_window: int | None
        if per_layer_sliding_window is not None:
            # per-layer sliding window
            sliding_window = per_layer_sliding_window
        elif cache_config is not None:
            # model-level sliding window
            sliding_window = cache_config.sliding_window
        else:
            sliding_window = None

        vllm_config = get_current_vllm_config()
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            calculate_kv_scales = cache_config.calculate_kv_scales
        else:
            kv_cache_dtype = "auto"
            calculate_kv_scales = False

        # llm-compressor mdls need to set cache_dtype to "fp8" manually.
        kv_cache_scheme = getattr(quant_config, "kv_cache_scheme", None)
        if kv_cache_scheme is not None:
            kv_cache_dtype = "fp8"
            calculate_kv_scales = False
            if cache_config is not None:
                cache_config.cache_dtype = "fp8"
                cache_config.calculate_kv_scales = False

        # Check if per-head quant scales are required based on kv_cache_scheme
        use_per_head_quant_scales = (
            kv_cache_scheme is not None
            and kv_cache_scheme.get("strategy") == "attn_head"
        )

        # Skip quantization for specified layers
        if cache_config is not None and cache_config.kv_cache_dtype_skip_layers:
            from vllm.model_executor.models.utils import extract_layer_index

            skip = False
            # Check attention type
            if (
                sliding_window is not None
                and "sliding_window" in cache_config.kv_cache_dtype_skip_layers
            ):
                skip = True
            # Check layer index
            layer_idx = extract_layer_index(prefix)
            if str(layer_idx) in cache_config.kv_cache_dtype_skip_layers:
                skip = True
            if skip:
                kv_cache_dtype = "auto"
                calculate_kv_scales = False
            logger.debug(
                "Layer %s: kv_cache_dtype=%s, sliding_window=%s",
                prefix,
                kv_cache_dtype,
                sliding_window,
            )

        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            kv_cache_dtype, vllm_config.model_config
        )
        self.kv_cache_dtype = kv_cache_dtype
        self.calculate_kv_scales = calculate_kv_scales
        if num_kv_heads is None:
            num_kv_heads = num_heads
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) is not divisible by num_kv_heads ({num_kv_heads})"
        )
        self.quant_config = quant_config
        self.layer_name = prefix

        self.num_heads = num_heads
        self.head_size = head_size
        self.head_size_v = self.head_size if head_size_v is None else head_size_v
        self.num_kv_heads = num_kv_heads
        self.scale = scale
        self.sliding_window = sliding_window
        self.logits_soft_cap = logits_soft_cap
        self.sinks = extra_impl_args.get("sinks")
        self.has_sink = self.sinks is not None

        # NOTE: model_config may be None during certain tests
        model_config = vllm_config.model_config
        self.use_mm_prefix = model_config is not None and model_config.is_mm_prefix_lm

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()
        if attn_backend is None:
            self.attn_backend = get_attn_backend(
                head_size,
                dtype,
                kv_cache_dtype,
                use_mla=False,
                has_sink=self.has_sink,
                use_mm_prefix=self.use_mm_prefix,
                use_per_head_quant_scales=use_per_head_quant_scales,
                attn_type=attn_type,
            )
        else:
            self.attn_backend = attn_backend
        backend_supports_alibi_sqrt = self.attn_backend.supports_alibi_sqrt()
        use_alibi_sqrt = use_alibi_sqrt if use_alibi_sqrt else False
        if use_alibi_sqrt and not backend_supports_alibi_sqrt:
            raise ValueError(
                f"use_alibi_sqrt is not supported by backend "
                f"{self.attn_backend.get_name()}."
            )
        self.use_alibi_sqrt = bool(use_alibi_sqrt)
        if backend_supports_alibi_sqrt:
            extra_impl_args["use_alibi_sqrt"] = self.use_alibi_sqrt
        # prefix caching + batch invariance is currently not supported for
        # FLASHINFER and TRITON_MLA.
        if (
            cache_config is not None
            and cache_config.enable_prefix_caching
            and envs.VLLM_BATCH_INVARIANT
            and (
                self.attn_backend.get_name() == "FLASHINFER"
                or self.attn_backend.get_name() == "TRITON_MLA"
            )
        ):
            logger.warning_once(
                "Disabling prefix caching for FLASHINFER/TRITON_MLA "
                "with batch invariance, as it is not yet supported.",
            )
            cache_config.enable_prefix_caching = False

        if extra_impl_args.get("chunk_lookback", -1) > -1:
            assert self.attn_backend.get_name() == "TRITON_ATTN", (
                f"Chunked attention with lookback requires the Triton backend, "
                f"but got {self.attn_backend.get_name()}."
            )

        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(  # type: ignore[assignment]  # impl_cls always returns an AttentionImpl subclass
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **extra_impl_args,
        )
        self.backend = AttentionBackendEnum[self.attn_backend.get_name()]
        self.dtype = dtype

        # For cuda-alike (CUDA and ROCM) and cpu platforms, we control how
        # torch.compile works by registering the attention as one giant
        # opaque custom op. For other platforms, we directly call them
        # and let torch.compile handle them.
        self.use_direct_call = not current_platform.opaque_attention_op()

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        self.attn_type = attn_type

        if kv_sharing_target_layer_name is not None:
            validate_kv_sharing_target(
                prefix,
                kv_sharing_target_layer_name,
                compilation_config.static_forward_context,
            )
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        # use a placeholder kv cache tensor during init, which will be replaced
        # by bind_kv_cache
        # this variable will not be accessed if use_direct_call is True
        self.kv_cache = torch.tensor([])

        # Initialize KV cache quantization attributes
        _init_kv_cache_quant(self, quant_config, prefix)

        # Initialize TurboQuant buffers (Pi, S, centroids) if tq cache dtype
        if kv_cache_dtype.startswith("turboquant_"):
            self._init_turboquant_buffers(kv_cache_dtype, head_size, prefix)

        # for attn backends supporting query quantization
        self.query_quant = None
        if (
            self.impl.supports_quant_query_input
            and (
                self.kv_cache_dtype.startswith("fp8") or self.kv_cache_dtype == "nvfp4"
            )
            and not self.kv_cache_dtype.endswith("per_token_head")
        ):
            is_per_head = (
                hasattr(self, "q_scale") and self.q_scale.numel() == self.num_kv_heads
            )
            block_size = self.head_size * self.num_heads // self.num_kv_heads
            self.query_quant = QuantFP8(
                static=True,
                group_shape=GroupShape(-1, block_size)
                if is_per_head
                else GroupShape.PER_TENSOR,
            )

        # 异构系统：CPU KV Cache（主存）占位符，延迟初始化
        self.cpu_kv_cache: torch.Tensor | None = None
        self._d2h_stream: torch.cuda.Stream | None = None

    def _init_turboquant_buffers(
        self, cache_dtype: str, head_size: int, prefix: str
    ) -> None:
        """Initialize TurboQuant centroids for Lloyd-Max quantization."""
        from vllm.model_executor.layers.quantization.turboquant.centroids import (
            get_centroids,
        )
        from vllm.model_executor.layers.quantization.turboquant.config import (
            TurboQuantConfig,
        )

        tq_config = TurboQuantConfig.from_cache_dtype(cache_dtype, head_size)

        self.register_buffer(
            "_tq_centroids",
            get_centroids(head_size, tq_config.centroid_bits),
        )
        self._tq_config = tq_config

        # Pre-allocate decode intermediate buffers so model.to(device) moves
        # them to GPU *before* the memory profiler runs.  Without this the
        # profiler gives all free memory to KV cache blocks and the first
        # decode OOMs when these buffers are lazily allocated.
        _vllm_cfg = get_current_vllm_config()
        B = _vllm_cfg.scheduler_config.max_num_seqs
        Hq = self.num_heads
        S = _vllm_cfg.attention_config.tq_max_kv_splits_for_cuda_graph
        D = head_size
        self.register_buffer(
            "_tq_mid_o_buf",
            torch.empty(B, Hq, S, D + 1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_tq_output_buf",
            torch.empty(B, Hq, D, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_tq_lse_buf",
            torch.empty(B, Hq, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        # For some alternate attention backends like MLA the attention output
        # shape does not match the query shape, so we optionally let the model
        # definition specify the output tensor shape.
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        """
        The KV cache is stored inside this class and is accessed via
        `self.kv_cache`.

        Attention metadata (`attn_metadata`) is set using a context manager in
        the model runner's `execute_model` method. It is accessed via forward
        context using
        `vllm.forward_context.get_forward_context().attn_metadata`.
        """
        if self.calculate_kv_scales:
            torch.ops.vllm.maybe_calc_kv_scales(
                query, key, value, _encode_layer_name(self.layer_name)
            )
        output_dtype = query.dtype
        if self.query_quant is not None:
            # quantizing with a simple torch operation enables
            # torch.compile to fuse this into previous ops
            # which reduces overheads during decoding.
            # Otherwise queries are quantized using custom ops
            # which causes decoding overheads
            assert self.kv_cache_dtype in {"fp8", "fp8_e4m3", "nvfp4"}

            # check if query quantization is supported
            if self.impl.supports_quant_query_input:
                query, _ = self.query_quant(query, self._q_scale)

        if output_shape is None:
            # Handle both 2D [num_tokens, hidden] and
            # 3D [num_tokens, heads, head_dim] query
            num_tokens = query.shape[0]
            output_shape = torch.Size((num_tokens, self.num_heads * self.head_size_v))
        output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
        hidden_size = output_shape[-1]
        # Reshape the query, key, and value tensors.
        # NOTE(woosuk): We do this outside the custom op to minimize the
        # CPU overheads from the non-CUDA-graph regions.
        query = query.view(-1, self.num_heads, self.head_size)
        output = output.view(-1, self.num_heads, self.head_size_v)
        if key is not None:
            key = key.view(-1, self.num_kv_heads, self.head_size)
        if value is not None:
            value = value.view(-1, self.num_kv_heads, self.head_size_v)

        # ==================== 统一通过 custom op 调用 attention ====================
        kv_cache_dummy_dep = None
        if self.use_direct_call:
            if (
                not self.attn_backend.forward_includes_kv_cache_update
                and self.kv_sharing_target_layer_name is None
                and key is not None
                and value is not None
            ):
                kv_cache_dummy_dep = unified_kv_cache_update(
                    key, value, self.layer_name
                )
            unified_attention_with_output(
                query,
                key,
                value,
                output,
                self.layer_name,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )
        else:
            encoded = _encode_layer_name(self.layer_name)
            if (
                not self.attn_backend.forward_includes_kv_cache_update
                and self.kv_sharing_target_layer_name is None
                and key is not None
                and value is not None
            ):
                kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
                    key, value, encoded
                )
            torch.ops.vllm.unified_attention_with_output(
                query,
                key,
                value,
                output,
                encoded,
                kv_cache_dummy_dep=kv_cache_dummy_dep,
            )

        return output.view(-1, hidden_size)

    # =================================================================
    # 以下方法为 CPU-GPU 异构系统新增
    # =================================================================

    def init_cpu_kv_cache(self) -> None:
        """初始化 CPU 主存 KV Cache，由 bind_kv_cache 调用。"""
        if self.kv_cache is None:
            return
        gpu_cache = self.kv_cache  # GPU Tensor, layout: [2, num_blocks, block_size, num_kv_heads, head_size]
        num_blocks = gpu_cache.shape[1]
        block_size = gpu_cache.shape[2]
        # 如果已存在且 shape 匹配，直接复用（避免重复分配）
        if self.cpu_kv_cache is not None:
            if (self.cpu_kv_cache.shape[1] == num_blocks and
                    self.cpu_kv_cache.shape[3] == block_size):
                return
            # shape 不匹配（例如 profile run 后换成了真正的 KV cache），释放旧的
            self.cpu_kv_cache = None
            self._d2h_stream = None
        # CPU layout: [2, num_blocks, num_kv_heads, block_size, head_size]
        cpu_shape = (2, num_blocks, self.num_kv_heads, block_size, self.head_size)
        self.cpu_kv_cache = torch.empty(
            cpu_shape,
            dtype=gpu_cache.dtype,
            device='cpu',
            pin_memory=True,
        )
        self._d2h_stream = torch.cuda.Stream(device=gpu_cache.device)

    def _prefill_kv_to_cpu_async(self) -> None:
        """Prefill 阶段结束后，将该层 GPU KV Cache 异步搬运到 CPU Cache。"""
        if self.cpu_kv_cache is None or self._d2h_stream is None:
            return
        profile_start = time.perf_counter() if _heter_profile_enabled() else None
        gpu_k = self.kv_cache[0]  # [num_blocks, block_size, num_kv_heads, head_size]
        gpu_v = self.kv_cache[1]
        cpu_k = self.cpu_kv_cache[0]  # [num_blocks, num_kv_heads, block_size, head_size]
        cpu_v = self.cpu_kv_cache[1]
        with torch.cuda.stream(self._d2h_stream):
            # GPU layout -> CPU layout: 交换 block_size 和 num_kv_heads 维度
            cpu_k.copy_(gpu_k.permute(0, 2, 1, 3), non_blocking=True)
            cpu_v.copy_(gpu_v.permute(0, 2, 1, 3), non_blocking=True)
        if profile_start is not None:
            _heter_profile_record(
                "prefill_kv_d2h_enqueue",
                time.perf_counter() - profile_start,
                self.layer_name,
                num_blocks=int(gpu_k.shape[0]),
                block_size=int(gpu_k.shape[1]),
            )

    def _decode_kv_to_cpu(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Decode 阶段，将新生成的 K/V 写入 CPU Paged Cache。"""
        assert self.cpu_kv_cache is not None
        profile_start = time.perf_counter() if _heter_profile_enabled() else None
        key_cpu = key.to('cpu', non_blocking=False)
        value_cpu = value.to('cpu', non_blocking=False)
        slot_mapping_cpu = slot_mapping.to('cpu', non_blocking=False)
        if profile_start is not None:
            _heter_profile_record(
                "decode_kv_d2h",
                time.perf_counter() - profile_start,
                self.layer_name,
                tokens=int(key.shape[0]),
            )
            profile_start = time.perf_counter()
        cpu_k_cache, cpu_v_cache = self.cpu_kv_cache.unbind(0)
        # 调用 vLLM CPU 版本的 KV Cache 写入 kernel
        if hasattr(torch.ops._C, 'cpu_attn_reshape_and_cache'):
            torch.ops._C.cpu_attn_reshape_and_cache(
                key_cpu, value_cpu,
                cpu_k_cache, cpu_v_cache,
                slot_mapping_cpu,
                "vec",  # TODO: 根据 CPU 平台动态选择 isa
            )
        else:
            # Fallback：Python 直接写入（性能较低，但无需编译 CPU 扩展）
            logger.info("【异构系统】Decode KV to CPU — Python Fallback 执行中")
            _cpu_attn_reshape_and_cache_fallback(
                key_cpu, value_cpu, cpu_k_cache, cpu_v_cache, slot_mapping_cpu
            )
        if profile_start is not None:
            _heter_profile_record(
                "decode_kv_cache_write_total",
                time.perf_counter() - profile_start,
                self.layer_name,
                tokens=int(slot_mapping_cpu.numel()),
            )

    def _build_cpu_metadata(self, attn_metadata) -> dict:
        """将 GPU attn_metadata 中的关键 tensor 搬到 CPU，供 CPU Attention kernel 使用。"""
        profile_start = time.perf_counter() if _heter_profile_enabled() else None
        cpu_meta = {
            'query_start_loc': attn_metadata.query_start_loc.cpu(),
            'seq_lens': attn_metadata.seq_lens.cpu(),
            'block_table': attn_metadata.block_table.cpu(),
            'causal': attn_metadata.causal,
        }
        if hasattr(attn_metadata, 'alibi_slopes') and attn_metadata.alibi_slopes is not None:
            cpu_meta['alibi_slopes'] = attn_metadata.alibi_slopes.cpu()
        else:
            cpu_meta['alibi_slopes'] = None
        if hasattr(attn_metadata, 'scheduler_metadata') and attn_metadata.scheduler_metadata is not None:
            cpu_meta['scheduler_metadata'] = attn_metadata.scheduler_metadata.cpu()
        else:
            cpu_meta['scheduler_metadata'] = torch.empty(0)
        if profile_start is not None:
            _heter_profile_record(
                "decode_metadata_d2h",
                time.perf_counter() - profile_start,
                self.layer_name,
                block_table_shape=list(cpu_meta['block_table'].shape),
            )
        return cpu_meta

    def _cpu_paged_attention(
        self,
        query_cpu: torch.Tensor,
        output_cpu: torch.Tensor,
        cpu_metadata: dict,
    ) -> None:
        """调用 vLLM CPU 版本的 Paged Attention Kernel。"""
        profile_start = time.perf_counter() if _heter_profile_enabled() else None
        # 调试标记：只要进入此方法就创建标记文件
        try:
            with open(_CPU_ATTENTION_EXECUTED_FLAG, "w") as f:
                f.write("1")
        except Exception:
            pass
        cpu_k_cache, cpu_v_cache = self.cpu_kv_cache.unbind(0)
        alibi_slopes = cpu_metadata.get('alibi_slopes')
        scheduler_metadata = cpu_metadata.get('scheduler_metadata', torch.empty(0))
        if hasattr(torch.ops._C, 'cpu_attention_with_kv_cache'):
            torch.ops._C.cpu_attention_with_kv_cache(
                query_cpu,
                cpu_k_cache,
                cpu_v_cache,
                output_cpu,
                cpu_metadata['query_start_loc'],
                cpu_metadata['seq_lens'],
                self.scale,
                cpu_metadata['causal'],
                alibi_slopes,
                self.sliding_window[0] if isinstance(self.sliding_window, tuple) else -1,
                self.sliding_window[1] if isinstance(self.sliding_window, tuple) else -1,
                cpu_metadata['block_table'],
                self.logits_soft_cap if self.logits_soft_cap is not None else 0.0,
                scheduler_metadata,
                self.sinks,
            )
            if profile_start is not None:
                _heter_profile_record(
                    "cpu_attention_cpp_total",
                    time.perf_counter() - profile_start,
                    self.layer_name,
                    query_tokens=int(query_cpu.shape[0]),
                )
        else:
            # Fallback：Python 实现的 Paged Attention（性能较低，用于快速验证）
            logger.info(
                "【异构系统】Decode CPU Paged Attention — Python Fallback 执行中, "
                "layer=%s, query_shape=%s", self.layer_name, query_cpu.shape
            )
            torch.ops.vllm.cpu_paged_attention_fallback(
                query_cpu, cpu_k_cache, cpu_v_cache, output_cpu,
                cpu_metadata['query_start_loc'],
                cpu_metadata['seq_lens'],
                self.scale,
                cpu_metadata['causal'],
                alibi_slopes if alibi_slopes is not None else torch.empty(0),
                self.sliding_window[0] if isinstance(self.sliding_window, tuple) else -1,
                self.sliding_window[1] if isinstance(self.sliding_window, tuple) else -1,
                cpu_metadata['block_table'],
                self.logits_soft_cap if self.logits_soft_cap is not None else 0.0,
                scheduler_metadata,
                self.sinks if self.sinks is not None else torch.empty(0),
            )
            if profile_start is not None:
                _heter_profile_record(
                    "cpu_attention_python_wrapper",
                    time.perf_counter() - profile_start,
                    self.layer_name,
                    query_tokens=int(query_cpu.shape[0]),
                )

    def calc_kv_scales(self, query, key, value):
        self._q_scale.copy_(torch.abs(query).max() / self.q_range)
        self._k_scale.copy_(torch.abs(key).max() / self.k_range)
        self._v_scale.copy_(torch.abs(value).max() / self.v_range)
        self._q_scale_float = self._q_scale.item()
        self._k_scale_float = self._k_scale.item()
        self._v_scale_float = self._v_scale.item()
        # We only calculate the scales once
        self.calculate_kv_scales = False

    def extra_repr(self) -> str:
        s = f"head_size={self.impl.head_size}"  # type: ignore
        s += f", num_heads={self.impl.num_heads}"  # type: ignore
        s += f", num_kv_heads={self.impl.num_kv_heads}"  # type: ignore
        s += f", scale={self.impl.scale}"  # type: ignore
        s += f", backend={self.impl.__class__.__name__}"
        return s

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        self.impl.process_weights_after_loading(act_dtype)

        # If we should not load quant weights, we initialize the scales to 1.0
        # as the default value. See [Note: Register q/k/v/prob scales in state dict]
        # for more details.
        quant_method = (
            self.quant_config.get_quant_method(self, prefix=self.layer_name)
            if self.quant_config
            else None
        )
        if not should_load_quant_weights(quant_method):
            set_default_quant_scales(self, register_buffer=False)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
        # Block size may get updated after model loading, refresh it
        block_size = vllm_config.cache_config.block_size
        # Should not be called for enc-dec or encoder-only attention.
        assert self.attn_type == AttentionType.DECODER
        quant_mode = get_kv_quant_mode(self.kv_cache_dtype)
        if self.sliding_window is not None:
            assert not vllm_config.model_config.use_mla, (
                "MLA is not supported for slidingwindow"
            )
            return SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size_v,
                dtype=self.kv_cache_torch_dtype,
                kv_quant_mode=quant_mode,
                sliding_window=self.sliding_window,
            )
        elif self.kv_cache_dtype.startswith("turboquant_"):
            from vllm.model_executor.layers.quantization.turboquant.config import (
                TurboQuantConfig,
            )
            from vllm.v1.kv_cache_interface import TQFullAttentionSpec

            tq_config = TurboQuantConfig.from_cache_dtype(
                self.kv_cache_dtype, self.head_size
            )
            return TQFullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size,
                dtype=self.kv_cache_torch_dtype,
                tq_slot_size=tq_config.slot_size_aligned,
            )
        else:
            return FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_size,
                head_size_v=self.head_size_v,
                dtype=self.kv_cache_torch_dtype,
                kv_quant_mode=quant_mode,
            )


def maybe_calc_kv_scales(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]

    # Only calculate if the layer's calculate_kv_scales flag is True
    # This flag gets set to False after the first forward pass
    if not self.calculate_kv_scales:
        return

    self.calc_kv_scales(query, key, value)


def maybe_calc_kv_scales_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="maybe_calc_kv_scales",
    op_func=maybe_calc_kv_scales,
    mutates_args=["query", "key", "value"],
    fake_impl=maybe_calc_kv_scales_fake,
)


def get_attention_context(
    layer_name: str,
) -> tuple[Any, "Attention | MLAAttention", torch.Tensor, torch.Tensor]:
    """Extract attention context for a given layer.

    This helper function extracts the attention metadata, attention layer
    instance, KV cache tensor, and slot mapping for a specific layer.

    Args:
        layer_name: The name/identifier of the attention layer.

    Returns:
        A tuple containing:
        - attn_metadata: Attention metadata for this specific layer, or None if
            no metadata available
        - attn_layer: The attention layer instance (Attention or MLAAttention)
        - kv_cache: The KV cache tensor for current forward pass
        - slot_mapping: The slot mapping for this specific layer

        Note: attn_metadata may be None, but attn_layer and kv_cache are always
        extracted from the forward context.
    """
    forward_context: ForwardContext = get_forward_context()
    attn_metadata_raw = forward_context.attn_metadata
    attn_metadata: AttentionMetadata
    if isinstance(attn_metadata_raw, dict):
        attn_metadata = attn_metadata_raw[layer_name]
    elif isinstance(attn_metadata_raw, list):
        # list[dict[str, AttentionMetadata]]: used in speculative decoding
        # where [0] is the base-model (non-speculative) metadata dict.
        attn_metadata = attn_metadata_raw[0][layer_name]
    else:
        attn_metadata = attn_metadata_raw
    attn_layer: Attention | MLAAttention = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache
    slot_mapping = forward_context.slot_mapping
    assert isinstance(slot_mapping, dict), (
        f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
    )
    layer_slot_mapping = slot_mapping.get(layer_name)
    return attn_metadata, attn_layer, kv_cache, layer_slot_mapping


def unified_kv_cache_update(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    """
    Returns a dummy that is passed to unified_attention to signal a side effect and
    the data dependency between them to ensure torch.compile preserves ordering.
    """
    layer_name = _resolve_layer_name(layer_name)
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)
    if layer_slot_mapping is not None:
        assert hasattr(attn_layer.impl, "do_kv_cache_update"), (
            f"{attn_layer.impl.__class__.__name__} does not support kv cache update"
        )
        attn_layer.impl.do_kv_cache_update(  # type: ignore[attr-defined]
            attn_layer,
            key,
            value,
            kv_cache,
            layer_slot_mapping,
        )

    return torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)


def unified_kv_cache_update_fake(
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: LayerNameType,
) -> torch.Tensor:
    return torch.empty(0, device=key.device, dtype=key.dtype)


direct_register_custom_op(
    op_name="unified_kv_cache_update",
    op_func=unified_kv_cache_update,
    fake_impl=unified_kv_cache_update_fake,
    mutates_args=[],
)


@maybe_transfer_kv_layer
def unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    # kv_cache_dummy_dep is not used but accepting it creates a data dependency
    # that ensures torch.compile preserves ordering between KV cache update and
    # attention forward.
    del kv_cache_dummy_dep
    layer_name = _resolve_layer_name(layer_name)
    attn_metadata, self, kv_cache, _ = get_attention_context(layer_name)

    # ==================== 异构系统：Pure Decode CPU 路径 ====================
    # 该分支在 custom op 内部执行，完全绕过 torch.compile 的图捕获，
    # 因此可以安全地进行跨设备（GPU->CPU->GPU）操作。
    # 注意：使用 is_cudagraph_capturing() 而非 torch.cuda.is_current_stream_capturing()
    # 因为前者在 profile/warmup 期间也为 True，可以避免 dummy run 走 CPU 路径。
    if (
        _heter_cpu_attention_enabled()
        and (not is_cudagraph_capturing())
        and _heter_is_pure_decode(attn_metadata)
        and (query.numel() > 0)
        and (self.cpu_kv_cache is not None)
    ):
        profile_decode_start = time.perf_counter() if _heter_profile_enabled() else None
        # 1. 等待该层之前 Prefill 的 async D2H 完成
        if self._d2h_stream is not None:
            profile_start = time.perf_counter() if profile_decode_start is not None else None
            torch.cuda.current_stream().wait_stream(self._d2h_stream)
            if profile_start is not None:
                _heter_profile_record(
                    "decode_wait_prefill_d2h",
                    time.perf_counter() - profile_start,
                    self.layer_name,
                )

        # 2. K/V 搬到 CPU，写入 CPU Paged Cache
        if key is not None and value is not None and attn_metadata is not None:
            slot_mapping = getattr(attn_metadata, 'slot_mapping', None)
            if slot_mapping is not None:
                self._decode_kv_to_cpu(key, value, slot_mapping)

        # 3. Q 搬到 CPU，分配 CPU output
        profile_start = time.perf_counter() if profile_decode_start is not None else None
        query_cpu = query.to('cpu', non_blocking=False)
        if profile_start is not None:
            _heter_profile_record(
                "decode_query_d2h",
                time.perf_counter() - profile_start,
                self.layer_name,
                query_tokens=int(query.shape[0]),
            )
            profile_start = time.perf_counter()
        output_cpu = torch.empty(
            (query.shape[0], self.num_heads, self.head_size_v),
            dtype=output.dtype,
            device='cpu',
            pin_memory=True,
        )
        if profile_start is not None:
            _heter_profile_record(
                "decode_output_cpu_alloc",
                time.perf_counter() - profile_start,
                self.layer_name,
                query_tokens=int(query.shape[0]),
            )

        # 4. 构造 CPU metadata 并执行 CPU Paged Attention
        if attn_metadata is not None:
            cpu_metadata = self._build_cpu_metadata(attn_metadata)
            self._cpu_paged_attention(query_cpu, output_cpu, cpu_metadata)

        # 5. Output 搬回 GPU
        profile_start = time.perf_counter() if profile_decode_start is not None else None
        output.copy_(output_cpu.to(query.device, non_blocking=False))
        if profile_start is not None:
            _heter_profile_record(
                "decode_output_h2d",
                time.perf_counter() - profile_start,
                self.layer_name,
                query_tokens=int(query.shape[0]),
            )
            _heter_profile_record(
                "decode_cpu_path_layer_total",
                time.perf_counter() - profile_decode_start,
                self.layer_name,
                query_tokens=int(query.shape[0]),
            )
        return

    # ==================== GPU 路径（原有逻辑）====================
    profile_gpu_start = (
        time.perf_counter()
        if _heter_profile_enabled() and not is_cudagraph_capturing()
        else None
    )
    self.impl.forward(
        self,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
    )
    if profile_gpu_start is not None:
        _heter_profile_record(
            "gpu_attention_forward",
            time.perf_counter() - profile_gpu_start,
            self.layer_name,
            pure_decode=bool(_heter_is_pure_decode(attn_metadata)),
            query_tokens=int(query.shape[0]) if query is not None else 0,
        )
    if (_heter_cpu_attention_enabled()
            and not is_cudagraph_capturing()
            and _heter_should_sync_full_kv_to_cpu(attn_metadata)):
        self._prefill_kv_to_cpu_async()


def unified_attention_with_output_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    return


direct_register_custom_op(
    op_name="unified_attention_with_output",
    op_func=unified_attention_with_output,
    mutates_args=["output", "output_block_scale"],
    fake_impl=unified_attention_with_output_fake,
)
