"""Tokenize the prompt, show logits shapes after prefill and after one decode token."""
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

model, tokenizer = load("mlx-community/Qwen3-4B-bf16")
print(type(model).__name__, "|", len(model.layers), "layers")
ids = tokenizer.encode("Explain why the sky is blue.")
print("ids   :", ids)
print("tokens:", len(ids), "->", [tokenizer.decode([i]) for i in ids])
cache = make_prompt_cache(model)
out = model(mx.array(ids)[None], cache=cache); mx.eval(out)
print("prefill logits shape:", out.shape)
scores = out[0, -1]
top = mx.argsort(scores)[-5:][::-1].tolist()
print("top-5 next tokens after the prompt:")
for i in top: print(f"   {scores[i].item():7.2f}  {tokenizer.decode([i])!r}")
out = model(mx.array([[42]]), cache=cache); mx.eval(out)
print("one-token logits shape:", out.shape)
