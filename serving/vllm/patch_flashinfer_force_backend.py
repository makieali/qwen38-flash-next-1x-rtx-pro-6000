"""Make FlashInfer's SM12x b12x MoE backend choice settable from the environment (B12X_FORCE_BACKEND).

FlashInfer has a test hook `_FORCED_BACKEND` in fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py. Its "static"
path pads the expert intermediate size 640 -> 768 and keeps a padded copy of every layer's weights, which runs
out of memory for this model on one 96 GB card. Forcing "dynamic" (multiples of 128) avoids that.

Usage: python patch_flashinfer_force_backend.py <venv site-packages dir>
Then:  export B12X_FORCE_BACKEND=dynamic
"""
import sys
from pathlib import Path

target = Path(sys.argv[1]) / "flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py"
src = target.read_text()
old = "_FORCED_BACKEND: str | None = None"
new = '_FORCED_BACKEND: str | None = os.environ.get("B12X_FORCE_BACKEND")  # local patch: env override (default unchanged)'
if new in src:
    print("already patched")
elif old not in src:
    sys.exit(f"pattern not found in {target}; FlashInfer version differs, patch by hand")
else:
    if "\nimport os\n" not in src:
        src = "import os\n" + src
    target.write_text(src.replace(old, new, 1))
    print(f"patched {target}")
