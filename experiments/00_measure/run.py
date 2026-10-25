"""Module 0 / Experiment 0.1 — the roofline sanity check.

For each model: weight bytes, prefill tok/s, decode TPOT (warmed up, prompt excluded),
and decode speed as a fraction of the memory-bandwidth ceiling.

    .venv/bin/python experiments/00_measure/run.py
"""
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import bench, report  # noqa: E402

MODELS = ["mlx-community/Qwen3-4B-bf16", "mlx-community/Qwen3-4B-4bit"]
MAX_TOKENS = 256
PROMPTS = [
    "How many positive whole-number divisors does 196 have?",
    "Explain why the sky is blue in three short paragraphs.",
    "Write a Python function that checks whether a string is a palindrome, with tests.",
]


def measure_bandwidth(gb=2.0, repeats=5):
    """Achievable read bandwidth: stream a large array through a reduction."""
    n = int(gb * 1e9 / 2)  # bf16 elements
    x = mx.ones((n,), dtype=mx.bfloat16)
    mx.eval(x)
    mx.eval(mx.sum(x))  # warmup
    tic = time.perf_counter()
    for _ in range(repeats):
        mx.eval(mx.sum(x))
    secs = (time.perf_counter() - tic) / repeats
    del x
    return x_bytes(n) / secs


def x_bytes(n_bf16):
    return n_bf16 * 2


def weight_bytes(model):
    return sum(p.nbytes for _, p in tree_flatten(model.parameters()))


def main():
    bw = measure_bandwidth()
    print(f"Achievable memory bandwidth (2 GB reduction): {bw / 1e9:.0f} GB/s  (spec ~546)\n")

    all_stats = []
    for model_id in MODELS:
        model, tokenizer = load(model_id)
        wb = weight_bytes(model)
        ceiling = bw / wb
        print(f"{model_id}: weights {wb / 1e9:.2f} GB -> decode ceiling {ceiling:.0f} tok/s")
        sampler = make_sampler(temp=0.0)

        def run(prompt):
            msgs = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            n, gen_tps, prompt_tps = 0, 0.0, 0.0
            for r in stream_generate(model, tokenizer, text, max_tokens=MAX_TOKENS, sampler=sampler):
                n += 1
                gen_tps, prompt_tps = r.generation_tps, r.prompt_tps
            return n, n / gen_tps, {"prefill_tps": prompt_tps}

        name = model_id.split("/")[-1]
        s = bench(name, run, PROMPTS, warmup=1, repeats=2)
        s.extra["ceiling_tps"] = ceiling
        all_stats.append(s)
        pf = sum(s.extra["prefill_tps"]) / len(s.extra["prefill_tps"])
        print(f"  prefill {pf:.0f} tok/s | decode {s.tps:.1f} tok/s = {100 * s.tps / ceiling:.0f}% of ceiling\n")
        del model
        mx.clear_cache()

    report(all_stats)


if __name__ == "__main__":
    main()
