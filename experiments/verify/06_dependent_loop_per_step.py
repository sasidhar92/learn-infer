"""A real decode loop: each step's input is the previous step's argmax (.item() round trip).
Per-step times recorded; printed as 10 blocks of 20. Prompt prefilled first. bf16 and 4-bit."""
import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

for name in ["mlx-community/Qwen3-4B-bf16", "mlx-community/Qwen3-4B-4bit"]:
    model, tok = load(name)
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array(tok.encode("Explain why the sky is blue."))[None], cache=cache))
    x = mx.array([[42]])
    out = model(x, cache=cache); mx.eval(out)       # warm-up
    times = []
    for _ in range(200):
        tic = time.perf_counter()
        out = model(x, cache=cache)
        t = mx.argmax(out[:, -1, :], axis=-1).item()
        x = mx.array([[t]])
        times.append((time.perf_counter() - tic) * 1000)
    print(name.split("/")[-1])
    for i in range(0, 200, 20):
        chunk = times[i:i+20]
        print(f"   steps {i:3d}-{i+19:3d}: {sum(chunk)/20:5.1f} ms")
    print(f"   mean {sum(times)/200:5.1f} ms -> {1000/(sum(times)/200):5.1f} tok/s")
    del model, cache; mx.clear_cache()
