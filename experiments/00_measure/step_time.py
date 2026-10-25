"""Where does TPOT go? Time one raw decode forward pass (1 token, with KV cache)
vs. the full generation-loop TPOT. Also time the lm_head alone.

    .venv/bin/python experiments/00_measure/step_time.py
"""
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

MODELS = ["mlx-community/Qwen3-4B-bf16", "mlx-community/Qwen3-4B-4bit"]


def timeit(fn, repeats=50):
    fn()
    mx.synchronize()
    tic = time.perf_counter()
    for _ in range(repeats):
        fn()
    mx.synchronize()
    return 1000 * (time.perf_counter() - tic) / repeats


for model_id in MODELS:
    model, tok = load(model_id)
    cache = make_prompt_cache(model)
    prompt = mx.array(tok.encode("The quick brown fox jumps over the lazy dog. " * 10))[None]
    mx.eval(model(prompt, cache=cache))  # prefill ~100 tokens so the cache is realistic
    x = mx.array([[1]])

    def full_step():
        mx.eval(model(x, cache=cache))
        cache[0].offset  # keep cache growing like real decode

    def head_only():
        h = mx.zeros((1, 1, model.args.hidden_size), dtype=mx.bfloat16)
        head = model.lm_head if hasattr(model, "lm_head") and model.lm_head is not None else model.model.embed_tokens.as_linear
        mx.eval(head(h))

    step_ms = timeit(full_step)
    head_ms = timeit(head_only)
    print(f"{model_id.split('/')[-1]:<16} raw forward step {step_ms:6.2f} ms | lm_head alone {head_ms:5.2f} ms")
    del model, cache
    mx.clear_cache()
