import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model, tokenizer = load("mlx-community/Qwen3-4B-bf16")
cache = make_prompt_cache(model)
ids = tokenizer.encode("Explain why the sky is blue.")
x = mx.array(ids)[None]
out = model(x, cache=cache)
mx.eval(out)
x = mx.array([[42]])


times = []

for _ in range(200):
  tic = time.perf_counter()
  out = model(x, cache= cache)
  tok = mx.argmax(out[:,-1, :], axis =-1).item()
  x = mx.array([[tok]])
  times.append((time.perf_counter()-tic)*1000)


for i in range(0,200,20
              ):
  chunk = times[i:i+20]
  print(f"steps {i:3d} - {i +19:3d} : {sum(chunk)/ 20:5.1f} ms")
    

  