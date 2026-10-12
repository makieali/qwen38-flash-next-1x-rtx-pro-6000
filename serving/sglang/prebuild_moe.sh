#!/bin/bash
# Build FlashInfer 0.6.17 SM120 CUTLASS fused-MoE JIT module with the same env as the SGLang launcher.
CU=/srv/ai/sglang/cuda133/nvidia/cu13; CACHE=/srv/ai/sglang/cache; R=/srv/ai/sglang/pennyroyal
export CUDA_HOME=$CU CUDACXX=$CU/bin/nvcc PATH=$R/.venv/bin:$CU/bin:$PATH LD_LIBRARY_PATH=$CU/lib
export CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 CUDAHOSTCXX=/usr/bin/g++-15 TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=16 FLASHINFER_NINJA_JOBS=16 FLASHINFER_NVCC_THREADS=1 XDG_CACHE_HOME=$CACHE FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer
exec $R/.venv/bin/python -c "
import time; t=time.time()
from flashinfer.fused_moe.core import get_cutlass_fused_moe_module
m = get_cutlass_fused_moe_module(\"120\")
print(\"fused_moe_120 built in %.0fs:\" % (time.time()-t), type(m).__name__, flush=True)"
