"""Quick serving benchmark for the local OpenAI-compatible endpoint (run on the AI box)."""
import concurrent.futures as cf, json, sys, time, urllib.request
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "qwen3.8-flash-next-uncensored"

def stream(messages, max_tokens=600):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.7,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time(); first = None; text = []; usage = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]": continue
            d = json.loads(line[5:])
            if d.get("usage"): usage = d["usage"]
            for ch in d.get("choices", []):
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if first is None: first = time.time()
                    text.append(piece)
    end = time.time(); out = usage["completion_tokens"] if usage else 0
    return {"ttft": first - t0 if first else None, "decode_tps": (out - 1) / (end - first) if first and out > 1 else 0,
            "out": out, "prompt": usage["prompt_tokens"] if usage else 0, "total_s": end - t0, "text": "".join(text)}

story = [{"role": "user", "content": "Write a detailed 450-word story about a lighthouse keeper who finds a message in a bottle."}]
code = [{"role": "user", "content": "Write a Python function that parses an ISO-8601 duration like P3DT4H5M into seconds, with tests."}]
print("warm-up:", round(stream(story, 64)["total_s"], 1), "s")
for name, msg in (("story", story), ("code", code)):
    rs = [stream(msg) for _ in range(2)]
    print(f"1 user  {name:5s}: decode {sum(r['decode_tps'] for r in rs)/2:6.1f} tok/s | first token {sum(r['ttft'] for r in rs)/2:.2f} s | {rs[0]['out']} tokens")
with cf.ThreadPoolExecutor(4) as ex:
    t0 = time.time(); rs = list(ex.map(lambda m: stream(m), [story, code, story, code])); wall = time.time() - t0
print(f"4 users      : aggregate {sum(r['out'] for r in rs)/wall:6.1f} tok/s | per-user decode {sum(r['decode_tps'] for r in rs)/4:.1f} tok/s")
long = [{"role": "user", "content": ("The quick brown fox jumps over the lazy dog. " * 3000) + "\nHow many times does the word fox appear above? Answer with a number only."}]
r = stream(long, 16)
print(f"long prompt  : {r['prompt']} tokens read in {r['ttft']:.2f} s = {r['prompt']/r['ttft']:.0f} tok/s prefill")
check = stream([{"role": "user", "content": "What is 17 * 23? Reply with just the number."}], 20)["text"].strip()
print("sanity 17*23 =", repr(check[:40]))
