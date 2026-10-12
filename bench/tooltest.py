"""Tool-call reliability under different sampling settings (20 two-step runs each, 4 in parallel)."""
import json, sys, concurrent.futures as cf
import eval_models as ev  # same folder
base, model = ev.TARGETS["flashnext"]
TOOLS = [{"type": "function", "function": {"name": "get_exchange_rate", "description": "Get the current exchange rate between two currencies",
          "parameters": {"type": "object", "properties": {"base": {"type": "string"}, "quote": {"type": "string"}}, "required": ["base", "quote"]}}}]

def run(extra):
    msgs = [{"role": "user", "content": "How many Pakistani rupees is 250 US dollars right now? Use the tool."}]
    r1, _ = ev.chat(base, model, msgs, tools=TOOLS, max_tokens=3000, **extra)
    m = r1["choices"][0]["message"]; calls = m.get("tool_calls") or []
    if not calls:
        return "nocall"
    msgs += [m, {"role": "tool", "tool_call_id": calls[0].get("id", "call_0"), "content": json.dumps({"rate": 280.5})}]
    r2, _ = ev.chat(base, model, msgs, tools=TOOLS, max_tokens=3000, **extra)
    t = ev.text_of(r2)
    if "70,125" in t or "70125" in t:
        return "ok"
    return "recall" if r2["choices"][0]["message"].get("tool_calls") else "wrong"

CONFIGS = {
    "presence 1.5 (now)": {},
    "presence 0": {"presence_penalty": 0.0},
    "presence 0, temp 0.3": {"presence_penalty": 0.0, "temperature": 0.3},
    "presence 0.5": {"presence_penalty": 0.5},
    "presence 1.0": {"presence_penalty": 1.0},
    "thinking on, p0.5": {"presence_penalty": 0.5, "chat_template_kwargs": {"enable_thinking": True}},
}
for name in (sys.argv[1:] or CONFIGS):
    extra = CONFIGS[name]
    with cf.ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda _: run(extra), range(40)))
    print(f"{name:22s}: ok {res.count('ok')}/40 | called tool again {res.count('recall')} | "
          f"no first call {res.count('nocall')} | wrong {res.count('wrong')}", flush=True)
