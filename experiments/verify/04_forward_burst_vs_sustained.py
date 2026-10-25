"""Same forward pass timed as a 20-step burst and as 200 sustained steps. bf16 and 4-bit.
Prompt is prefilled first. Input token is fixed (42) so only run length differs."""
import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

def timed_steps(model, cache, x, n):
    tic = time.perf_counter()
    for _ in range(n):
        mx.eval(model(x, cache=cache))
    return (time.perf_counter() - tic) / n * 1000

for name in ["mlx-community/Qwen3-4B-bf16", "mlx-community/Qwen3-4B-4bit"]:
    model, tok = load(name)
    cache = make_prompt_cache(model)
    mx.eval(model(mx.array(tok.encode("Explain why the sky is blue."))[None], cache=cache))
    x = mx.array([[42]])
    mx.eval(model(x, cache=cache))                     # warm-up
    burst = timed_steps(model, cache, x, 20)
    sustained = timed_steps(model, cache, x, 200)
    print(f"{name.split('/')[-1]:<14} burst(20) {burst:5.1f} ms | sustained(200) {sustained:5.1f} ms")
    del model, cache; mx.clear_cache()
