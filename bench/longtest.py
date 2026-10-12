import random
from pathlib import Path
src = Path(__file__).with_name("bench.py").read_text().split("story = [")[0]
exec(src)
words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango".split()
for i in range(3):
    rnd = random.Random()
    text = " ".join(rnd.choice(words) + str(rnd.randint(0, 999)) for _ in range(9000))
    msg = [{"role": "user", "content": text + "\nHow many words above start with alpha? Answer with a number only."}]
    r = stream(msg, 16)
    print(f"run {i+1}: {r['prompt']} unique tokens read in {r['ttft']:.2f} s = {r['prompt']/r['ttft']:.0f} tok/s")
