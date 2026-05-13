#include "core/registration.h"

#include <torch/extension.h>
#include <torch/library.h>

torch::Tensor get_scheduler_metadata(
    const int64_t num_req, const int64_t num_heads_q,
    const int64_t num_heads_kv, const int64_t head_dim,
    const torch::Tensor& seq_lens, at::ScalarType dtype,
    const torch::Tensor& query_start_loc, const bool casual,
    const int64_t window_size, const std::string& isa_hint,
    const bool enable_kv_split);

void cpu_attn_reshape_and_cache(const torch::Tensor& key,
                                const torch::Tensor& value,
                                torch::Tensor& key_cache,
                                torch::Tensor& value_cache,
                                const torch::Tensor& slot_mapping,
                                const std::string& isa);

void cpu_attention_with_kv_cache(
    const torch::Tensor& query, const torch::Tensor& key_cache,
    const torch::Tensor& value_cache, torch::Tensor& output,
    const torch::Tensor& query_start_loc, const torch::Tensor& seq_lens,
    const double scale, const bool causal,
    const std::optional<torch::Tensor>& alibi_slopes,
    const int64_t sliding_window_left, const int64_t sliding_window_right,
    const torch::Tensor& block_table, const double softcap,
    const torch::Tensor& scheduler_metadata,
    const std::optional<torch::Tensor>& s_aux);

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "get_scheduler_metadata(int num_req, int num_heads_q, int num_heads_kv, "
      "int head_dim, Tensor seq_lens, ScalarType dtype, Tensor "
      "query_start_loc, bool casual, int window_size, str isa_hint, bool "
      "enable_kv_split) -> Tensor",
      &get_scheduler_metadata);
  ops.def(
      "reshape_and_cache(Tensor key, Tensor value, Tensor(a2!) key_cache, "
      "Tensor(a3!) value_cache, Tensor slot_mapping, str isa) -> ()",
      &cpu_attn_reshape_and_cache);
  ops.def(
      "attention_with_kv_cache(Tensor query, Tensor key_cache, Tensor "
      "value_cache, Tensor(a3!) output, Tensor query_start_loc, Tensor "
      "seq_lens, float scale, bool causal, Tensor? alibi_slopes, SymInt "
      "sliding_window_left, SymInt sliding_window_right, Tensor block_table, "
      "float softcap, Tensor scheduler_metadata, Tensor? s_aux) -> ()",
      &cpu_attention_with_kv_cache);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
