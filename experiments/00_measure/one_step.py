import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model, tokenizer = load("mlx-community/Qwen3-4B-bf16")
#model, tokenizer = load("mlx-community/Qwen3-4B-4bit")
print(type(model).__name__)
print(len(model.layers), "layers")


ids= tokenizer.encode("Explain why the sky is blue.")
print(ids)
print(len(ids), "tokens")

cache = make_prompt_cache(model)
prompt = mx.array(ids)[None]
out = model(prompt, cache= cache)
mx.eval(out)
print(out.shape)


x= mx.array([[42]])
out = model(x, cache= cache)
mx.eval(out)
print(out.shape)


tic = time.perf_counter()
for _ in range(20):
  #mx.eval(model(x, cache=cache))
  mx.eval(mx.argmax(out[:,-1,:], axis =-1))

secs = (time.perf_counter()- tic)/20

print(f"argmax alone: {secs*1000:.2f} ms")
#print(f"one forward pass: {secs*1000:.1f} ms")

#print(f"that alone would allow {1/ secs:.0f} tok/s")






