"""Local vLLM patch: share FlashInfer b12x MoE workspaces across identically shaped layers."""
import sys
p = sys.argv[1]
s = open(p).read()
if "_SHARED_B12X_WRAPPERS" in s:
    print("already patched"); sys.exit()
old = """        self._wrapper = B12xMoEWrapper(
            num_experts=self.global_num_experts,
            top_k=self.topk,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_size_per_partition,
            use_cuda_graph=True,
            max_num_tokens=self.max_num_tokens,
            num_local_experts=self.num_local_experts,
            activation=self._activation_str,
        )
"""
new = """        # Local patch (AI box, 2026-10-07): MoE layers run strictly one after another, so every identically
        # shaped wrapper reuses the first one's CUDA-graph workspaces (FlashInfer supports this via
        # shared_*). Without it each of the 48 layers allocates its own (~14 GB total on SM120).
        key = (self.global_num_experts, self.topk, self.hidden_dim, self.intermediate_size_per_partition,
               self.max_num_tokens, self.num_local_experts, self._activation_str)
        first = _SHARED_B12X_WRAPPERS.get(key)
        shared = {} if first is None else {
            "shared_static_workspace": first._static_workspace,
            "shared_dynamic_workspace": first._dynamic_workspace,
            "shared_output": first._moe_output,
        }
        self._wrapper = B12xMoEWrapper(
            num_experts=self.global_num_experts,
            top_k=self.topk,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_size_per_partition,
            use_cuda_graph=True,
            max_num_tokens=self.max_num_tokens,
            num_local_experts=self.num_local_experts,
            activation=self._activation_str,
            **shared,
        )
        _SHARED_B12X_WRAPPERS.setdefault(key, self._wrapper)
"""
assert s.count(old) == 1, "anchor not found"
s = s.replace(old, new)
i = s.index("\nclass ")
s = s[:i] + "\n\n_SHARED_B12X_WRAPPERS: dict = {}  # local patch: one workspace set per wrapper shape\n" + s[i:]
open(p, "w").write(s)
print("patched")
