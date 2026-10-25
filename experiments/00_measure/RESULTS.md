# Smoke test — Qwen3-4B, DFlash vs autoregressive (2026-10-18)

Hardware: M4 Max, 128 GB. MLX 0.32, mlx-lm 0.31.3. Draft: z-lab/Qwen3-4B-DFlash-b16 (5 layers, block 16, bf16).
Command: `dflash benchmark mlx --dataset gsm8k --max-samples 5 --max-new-tokens 256 [--reasoning off] [--block-size 5]`
Greedy. 5 prompts only — directional, not final numbers. Raw output in smoke_test.txt.

| target | block | thinking | baseline tok/s | DFlash tok/s | speedup | τ (acc. len) |
|---|---|---|---|---|---|---|
| bf16 | 16 | on (default) | 22.0 | 24.7 | 1.12× | 3.06 |
| 4-bit | 5 | on (default) | 70.5 | 62.8 | 0.89× | 2.59 |
| bf16 | 16 | **off** | 23.8 | 55.6 | **2.34×** | **6.74** |
| 4-bit | 5 | **off** | 74.1 | 88.9 | 1.20× | 3.50 |

Paper, same model, greedy, H100-class GPU, Transformers: τ 6.54, 4.91×.

## What this already teaches (hooks for Modules 0–5)
1. **Decoding mode must match the drafter's training data.** Qwen3's template defaults to
   thinking on; this drafter was trained non-thinking. τ 3.06 → 6.74 from one flag.
2. **τ reproduces, speedup does not.** τ 6.74 ≈ paper's 6.54, so the drafter is as good here as
   on an H100. The 2.3× vs 4.9× gap is entirely `T_draft + T_verify` on this hardware
   (Module 1: how flat is the k-token forward-pass curve on MLX bf16?).
3. **Quantized target caps the block.** With block 5, 47% of steps hit the cap (histogram
   last bin), so τ is artificially truncated at 3.5 and speedup is only 1.2×. The README's
   "block_size <= 5 for quantized" is a kernel limitation worth measuring directly.
4. **bf16 baseline is at ~35% of the bandwidth roofline** (23.8 vs ~68 tok/s ceiling);
   4-bit is at ~32% (74 vs ~235). Module 0 question: where does the rest go?

---

# Experiment 0.1 — roofline sanity check (2026-10-18)

Scripts: `run.py` (harness-based, 3 prompts × 2 repeats, warmup 1, greedy, thinking off,
256 new tokens), `step_time.py` (raw single-token forward, 50 reps, synchronous).
Raw outputs: `run_output.txt`, `step_time_output.txt`.

Achievable bandwidth (2 GB bf16 reduction): **425 GB/s** (spec 546 → 78%).

| model | weights | ceiling tok/s | prefill tok/s | decode tok/s | TPOT ms | % of ceiling | raw fwd ms | lm_head ms |
|---|---|---|---|---|---|---|---|---|
| Qwen3-4B bf16 | 8.04 GB | 53 | 157 | 21.9 | 45.7 ± 0.9 | 41% | 39.1 | 5.0 |
| Qwen3-4B 4-bit | 2.26 GB | 188 | 433 | 70.0 | 14.3 ± 0.6 | 37% | 17.3 | 1.0 |

## Interpretation
1. **The roofline holds as a shape, not as a number.** 4-bit has 3.56× fewer bytes and is
   3.2× faster. Decode really is bandwidth-bound; the ratio is almost exactly bytes.
2. **Both reach only ~40% of the ceiling.** For bf16 the raw forward pass alone (39 ms)
   is 2× the ideal 19 ms, so the loss is *inside the kernels*: MLX's bf16 matrix-vector
   kernels do not stream weights at full bandwidth. The Python loop adds only ~6 ms (14%).
3. **The synchronous measurement is slower than the real loop for 4-bit** (17.3 vs 14.3 ms).
   MLX is lazy: each step first builds a graph in Python, then runs it on the GPU. The
   generation loop uses `async_eval` to overlap building step N+1 with running step N.
   A sync benchmark serializes them and over-reports. Lesson: *how* you time changes the answer.
4. **lm_head is 11% of the bf16 step** (152k-vocab × 2560). This matters for DFlash: the
   drafter shares the target's head and must run it over a 16-token block (Module 4).
5. **Prefill is 7× faster per token than decode** (157 vs 21.9) even on a 22-token prompt:
   the same weights are read once for all tokens. This is the Module 1 "free lunch".

## Measurement rules learned
- Warm up at least one full generation before timing; the first call includes compilation.
- Exclude prefill from TPOT; report it separately.
- Report ± over repeats; here the spread is <5%, so 3 prompts × 2 repeats is enough for 4B.
- State the decoding mode (`enable_thinking=False`), sampler (greedy), and max tokens.
- Measure bandwidth on the machine; never plug in the spec sheet.

> **Revised 2026-10-20.** The "~40% of ceiling" finding above was environmental. The same
> `mlx_lm.generate` command gave 55.0 tok/s (99% of the bf16 ceiling) on Oct 20 on AC power.
> Interpretation points 2 and 3 above are withdrawn. See book/index.html Chapter 5.
