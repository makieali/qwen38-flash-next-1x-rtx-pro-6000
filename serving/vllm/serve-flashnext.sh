#!/bin/bash
# Qwen3.8-Flash-Next heretic (uncensored), our NVFP4 build, on the RTX PRO 6000 (SM120).
# Usage: serve-flashnext.sh [moe-backend]   default flashinfer_b12x (native FP4 tensor cores, FlashInfer 0.7.1rc4)
BACKEND=${1:-flashinfer_b12x}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_HOME=/srv/ai/vllm/venv/lib/python3.14/site-packages/nvidia/cu13 PATH=/srv/ai/vllm/venv/bin:/srv/ai/vllm/venv/lib/python3.14/site-packages/nvidia/cu13/bin:$PATH
export B12X_FORCE_BACKEND=dynamic  # static path pads n=640 to 768 and keeps a per-layer weight copy (OOM)
export VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0
exec /srv/ai/vllm/venv/bin/vllm serve /srv/ai/models/flash-next-heretic-nvfp4 \
  --served-model-name qwen3.8-flash-next-uncensored \
  --host 127.0.0.1 --port 8080 \
  --quantization modelopt \
  --moe-backend "$BACKEND" \
  --distributed-executor-backend mp \
  --engram-config "{\"cpu_offload\": true, \"use_thp\": true}" \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 4 --max-num-batched-tokens 4096 \
  --enable-prefix-caching \
  --limit-mm-per-prompt "{\"image\": 2, \"video\": 0}" \
  --speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": 3, \"moe_backend\": \"triton\"}" \
  --compilation-config "{\"cudagraph_mode\": \"FULL_DECODE_ONLY\"}" \
  --override-generation-config "{\"temperature\": 0.7, \"top_p\": 0.8, \"top_k\": 20, \"presence_penalty\": 1.5}" \
  --default-chat-template-kwargs "{\"enable_thinking\": false}" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
