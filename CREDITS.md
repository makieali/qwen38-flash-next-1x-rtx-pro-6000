# Credits

This repository contains our own scripts, patches, measurements and docs. It contains **no model weights** and no
copies of the projects below; it links to them, and two small patches modify installed copies of vLLM/FlashInfer at
install time. Check each project's licence before you use it.

| Project | What we used it for |
|---|---|
| Qwen team: Qwen3.8 Flash-Next | The model architecture and original weights (see the model card for its licence) |
| trohrbaugh: Qwen3.8-Flash-Next-heretic | The BF16 uncensored variant we converted (licence per its model card) |
| RadixArk: NVFP4 release of Qwen3.8 Flash-Next | Reference layout, calibrated activation scales (`input_scale`), `tokenizer.json` |
| [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) (SGLang Pennyroyal) | Production engine, launcher we adapted, chat template, FR-Spec token map, YaRN reference config |
| [SGLang](https://github.com/sgl-project/sglang) | Serving engine (Apache-2.0) |
| [vLLM](https://github.com/vllm-project/vllm) | Earlier serving path (Apache-2.0) |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | FP4 MoE and Gated DeltaNet kernels (Apache-2.0) |
| b12x | SM12x FP4 MoE backend used by vLLM |
| [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) | NVFP4 checkpoint format |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | Our starting point (MIT) |
| [SirTificate/qwen38-flash-next-2x-rtx6000-sm120](https://github.com/SirTificate/qwen38-flash-next-2x-rtx6000-sm120) | Two-card numbers we quote for comparison |
| NVIDIA NVML / pynvml | Fan and power control in `tools/gpuctl.py` |

Thanks to all of them. Corrections welcome: open an issue.
