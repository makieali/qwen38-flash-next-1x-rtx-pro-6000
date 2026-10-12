"""Acceptance test after any vLLM change: speed + tools x10 + vision + uncensored. Runs on the AI box."""
from pathlib import Path
import base64, glob, json, subprocess, sys
import eval_models as ev  # same folder
base, model = ev.TARGETS["flashnext"]
print(subprocess.run([sys.executable, str(Path(__file__).with_name("bench.py"))], capture_output=True, text=True).stdout.strip())
tools_ok = 0; again = 0
for _ in range(10):
    r = ev.test_tools(base, model); tools_ok += bool(r.get("used_result")); again += r.get("called", False) and not r.get("used_result")
print(f"tools: {tools_ok}/10 answered with the tool result ({again} asked for the tool again instead)")
print("uncensored:", json.dumps(ev.test_uncensored(base, model)))
img = sorted(glob.glob("/usr/share/backgrounds/**/*.jpg", recursive=True) + glob.glob("/usr/share/backgrounds/**/*.png", recursive=True))
if img:
    b64 = base64.b64encode(open(img[0], "rb").read()).decode(); mime = "png" if img[0].endswith("png") else "jpeg"
    r, s = ev.chat(base, model, [{"role": "user", "content": [{"type": "text", "text": "Describe this image in one sentence."},
                   {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{b64}"}}]}], max_tokens=200)
    print(f"vision ({img[0].split('/')[-1]}, {s:.1f}s):", ev.text_of(r)[:200])
