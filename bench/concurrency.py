"""Concurrency benchmark: N parallel streaming chats with unique prompts; total and per-user tok/s, time to first
token, peak running/queued requests (from SGLang /metrics). Run on the serving machine. First pass = warm-up."""
import json, threading, time, urllib.request, uuid
URL, MODEL = "http://127.0.0.1:8080", "qwen3.8-flash-next-uncensored"
TOPICS = ["a lighthouse keeper", "a robot chef", "a desert caravan", "a submarine crew", "a mountain village", "a jazz club",
          "a space station garden", "a night market", "an old library", "a racing team"]
def one(i, res):
    msg = f"[{uuid.uuid4()}] Write a vivid 500-word story about {TOPICS[i % len(TOPICS)]}."
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": msg}], "max_tokens": 700, "temperature": 0.7,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    t0 = time.time(); first = None; toks = 0
    with urllib.request.urlopen(urllib.request.Request(URL + "/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]": continue
            d = json.loads(line[6:])
            if d.get("choices") and d["choices"][0].get("delta", {}).get("content") and first is None: first = time.time()
            if d.get("usage"): toks = d["usage"]["completion_tokens"]
    res.append((toks, first - t0 if first else None, time.time() - (first or t0)))
peak = [0]; stop = [False]
def watch():
    while not stop[0]:
        m = urllib.request.urlopen(URL + "/metrics", timeout=5).read().decode()
        for l in m.splitlines():
            if l.startswith("sglang:num_running_reqs"): peak[0] = max(peak[0], float(l.split()[-1]))
            if l.startswith("sglang:num_queue_reqs") and float(l.split()[-1]) > 0: peak.append(("queued", float(l.split()[-1])))
        time.sleep(0.5)
for n in (1, 4, 8):
    res, peak[:] = [], [0]; stop[0] = False; w = threading.Thread(target=watch); w.start()
    t = time.time(); th = [threading.Thread(target=one, args=(i, res)) for i in range(n)]; [x.start() for x in th]; [x.join() for x in th]
    wall = time.time() - t; stop[0] = True; w.join()
    tot = sum(r[0] for r in res); per = [r[0] / r[2] for r in res if r[2] > 0]
    q = max([p[1] for p in peak[1:]], default=0)
    print(f"{n} users: total {tot/wall:6.0f} tok/s | per user {min(per):5.0f}-{max(per):5.0f} tok/s | first word {min(r[1] for r in res):.2f}-{max(r[1] for r in res):.2f} s | running peak {peak[0]:.0f}, queued peak {q:.0f}")
