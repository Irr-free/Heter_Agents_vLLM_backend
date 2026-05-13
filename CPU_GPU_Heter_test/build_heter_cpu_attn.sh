#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/zb/tool/miniconda3/envs/vLLM_GPU/bin/python}"
CXX_BIN="${CXX:-c++}"
OUT_DIR="${REPO_ROOT}/vllm/libs"
OUT_LIB="${OUT_DIR}/libvllm_heter_cpu_attn.so"
BUILD_DIR="${REPO_ROOT}/CPU_GPU_Heter_test/build"

mkdir -p "${OUT_DIR}" "${BUILD_DIR}"

"${PYTHON_BIN}" "${REPO_ROOT}/csrc/cpu/generate_cpu_attn_dispatch.py"

TORCH_FLAGS="$("${PYTHON_BIN}" - <<'PY'
from torch.utils.cpp_extension import include_paths, library_paths
import sysconfig

parts = []
parts.append(f"-I{sysconfig.get_paths()['include']}")
for path in include_paths("cpu"):
    parts.append(f"-I{path}")
for path in library_paths("cpu"):
    parts.append(f"-L{path}")
print(" ".join(parts))
PY
)"
TORCH_CXX11_ABI="$("${PYTHON_BIN}" - <<'PY'
import torch
print(1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0)
PY
)"

"${CXX_BIN}" -std=gnu++17 -O3 -fPIC -shared \
  -I"${REPO_ROOT}/csrc" \
  ${TORCH_FLAGS} \
  -D_GLIBCXX_USE_CXX11_ABI="${TORCH_CXX11_ABI}" \
  -DTORCH_EXTENSION_NAME=_heter_cpu_attn_C \
  -DVLLM_CPU_EXTENSION \
  -DVLLM_NUMA_DISABLED \
  -fopenmp -mf16c -mavx2 \
  "${REPO_ROOT}/csrc/cpu/utils.cpp" \
  "${REPO_ROOT}/csrc/cpu/cpu_attn.cpp" \
  "${REPO_ROOT}/csrc/cpu/heter_cpu_attn_bindings.cpp" \
  -ltorch -ltorch_cpu -lc10 -lgomp \
  -Wl,-rpath,'$ORIGIN' \
  -Wl,-rpath,"$("${PYTHON_BIN}" - <<'PY'
from torch.utils.cpp_extension import library_paths
print(library_paths("cpu")[0])
PY
)" \
  -o "${OUT_LIB}"

echo "${OUT_LIB}"
