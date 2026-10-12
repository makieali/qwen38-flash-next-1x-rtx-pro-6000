#!/bin/bash
# Flash-Next (our uncensored NVFP4) on SGLang Pennyroyal v2.5.3, adapted from configs/pennyroyal/serve-flash-next-frspec.sh:
# no NIXL/HiCache storage, native 256k context (no YaRN x2), RAM-backed PLE, FR-Spec + NEXTN MTP, FP8 KV.
# Usage: serve-sglang.sh [online_mxfp8: true|false]
set -euo pipefail
R=/srv/ai/sglang/pennyroyal; C=$R/configs/pennyroyal
MODEL=/srv/ai/models/flash-next-heretic-nvfp4-sgl
CACHE=/srv/ai/sglang/cache; mkdir -p $CACHE/{huggingface,torch,torchinductor,triton,cuda,flashinfer,sglang/jit}
CU=/srv/ai/sglang/cuda133/nvidia/cu13   # CUDA 13.3 pip toolkit (13.0 headers clash with Ubuntu 26.04 glibc rsqrt)
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 CUDA_HOME=$CU CUDACXX=$CU/bin/nvcc
export PATH=$R/.venv/bin:$CU/bin:$PATH LD_LIBRARY_PATH=$CU/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CC=/usr/bin/gcc-15 CXX=/usr/bin/g++-15 CUDAHOSTCXX=/usr/bin/g++-15 TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=8 FLASHINFER_NINJA_JOBS=8 FLASHINFER_NVCC_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=8
export HF_HOME=$CACHE/huggingface XDG_CACHE_HOME=$CACHE TORCH_HOME=$CACHE/torch TORCHINDUCTOR_CACHE_DIR=$CACHE/torchinductor
export TRITON_CACHE_DIR=$CACHE/triton CUDA_CACHE_PATH=$CACHE/cuda FLASHINFER_WORKSPACE_BASE=$CACHE/flashinfer
export SGLANG_CACHE_DIR=$CACHE/sglang SGLANG_JIT_CACHE_DIR=$CACHE/sglang/jit
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SGLANG_NUMA_BIND_V2=false SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_MAMBA_CONV_DTYPE=bfloat16 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export SGLANG_MM_PREPROCESS_DEVICE=cpu SGLANG_FORWARD_UNKNOWN_TOOLS=true
export SGLANG_SM120_ONLINE_MXFP8=${1:-false}
exec $R/.venv/bin/sglang serve \
  --model-path $MODEL --load-format safetensors --served-model-name qwen3.8-flash-next-uncensored \
  --host 127.0.0.1 --port 8080 --tp 1 \
  --dtype bfloat16 --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.95 --warmups=structured_output \
  --context-length 262144 --page-size 64 --max-running-requests 4 --sleep-on-idle \
  --chunked-prefill-size 4096 \
  --mamba-radix-cache-strategy extra_buffer --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 24 --gdn-mtp-cache-mode none \
  --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer --mamba-track-interval 64 \
  --ple-offload-embedding --trust-remote-code \
  --chat-template $C/templates/froggeric-v22.5.jinja --image-processor-backend pil \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-request-time-stats-logging --enable-metrics \
  --preferred-sampling-params "{\"temperature\": 0.7, \"top_p\": 0.8, \"top_k\": 20, \"presence_penalty\": 1.5}" \
  --default-chat-template-kwargs "{\"enable_thinking\": false}" \
  --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --speculative-draft-model-quantization unquant --speculative-token-map $C/frspec/flash-next-64k.pt --watchdog-timeout 1800
