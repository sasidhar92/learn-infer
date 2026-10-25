"""Cost of picking the token: argmax over 151,936 logits, 20 reps."""
import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model, tok = load("mlx-community/Qwen3-4B-bf16")
cache = make_prompt_cache(model)
out = model(mx.array([[42]]), cache=cache); mx.eval(out)
mx.eval(mx.argmax(out[:, -1, :], axis=-1))
tic = time.perf_counter()
for _ in range(20):
    mx.eval(mx.argmax(out[:, -1, :], axis=-1))
print(f"argmax alone: {(time.perf_counter()-tic)/20*1000:.2f} ms  (logits shape {out.shape})")
