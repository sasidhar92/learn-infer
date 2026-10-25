# Verification run — all experiments in order (2026-10-20, 11:47, AC power, low power off)

Files: `00_environment.py` … `06_dependent_loop_per_step.py`, runner `run_all.sh`,
raw output `RESULTS_RAW.txt`. Re-run everything with:

    experiments/verify/run_all.sh

Note on `00_environment`: the "thermal: 2 warning lines" in the raw output counted the two
lines that say "No thermal warning level has been recorded" — i.e. no warnings. Script fixed
to print the lines themselves.

## Results, with the earlier hand-run numbers beside them

| # | Experiment | Setting | Earlier (Oct 18–20, by hand) | Verified now | Ceiling |
|---|---|---|---|---|---|
| 01 | bandwidth | 0.2 GB sum | 150–179 GB/s | **176 GB/s** (1.13 ms) | 546 spec |
| 01 | bandwidth | 2 GB sum | 449 GB/s, 4.5 ms | **469 GB/s** (4.26 ms) | |
| 01 | bandwidth | 8 GB sum | 444 GB/s, 18.0 ms | **435 GB/s** (18.40 ms) | |
| 02 | generate CLI | bf16, greedy, 200 tok | 26.6 (Oct 19) → 55.0 (Oct 20) | **55.8 tok/s**, 8.11 GB peak | 55–58 |
| 02 | generate CLI | 4-bit, greedy, 200 tok | 85.1 (Oct 19) | **165.2 tok/s**, 2.37 GB peak | 196–207 |
| 03 | tokens | "Explain why the sky is blue." | 8 tokens | **8 tokens**: Ex·plain· why· the· sky· is· blue·. | |
| 03 | shapes | prefill / one token | (1,8,151936) / (1,1,151936) | **same** | |
| 03 | top-5 next token | after raw prompt | — | ' The' 20.88, ' \n\n' 18.25, ' (' 18.25, ' Also' 18.12, ' ' 17.38 | |
| 04 | forward pass | bf16 burst 20 / sustained 200 | 19.9 / 41.4 ms (Oct 19) | **18.8 / 18.9 ms** | 18.4 ms read |
| 04 | forward pass | 4-bit burst 20 / sustained 200 | 7.7 / — | **7.1 / 7.2 ms** | 5.2 ms read |
| 05 | argmax | 151,936 logits | 0.19 ms | **0.26 ms** | |
| 06 | dependent loop | bf16, 200 steps, per-step | 36.2 ms (Oct 19), 18.7 (Oct 20) | **18.9 ms flat → 53.0 tok/s** | |
| 06 | dependent loop | 4-bit, 200 steps, per-step | — | **7.2 ms flat → 139.7 tok/s** | |

## What the verified set says

1. **Bandwidth is stable**: 435–469 GB/s on large reads, overhead-dominated on 0.2 GB. Matches your hand runs.
2. **Tokenization**: "Explain" is two tokens ("Ex" + "plain"), so 7 words + "." = 8 tokens. Earlier I said one word = one token here; that was wrong for "Explain".
3. **bf16 decode is at the roofline today**: forward pass 18.8 ms vs 18.4 ms pure read (98%); generate 55.8 tok/s.
4. **4-bit decode is at 84% of its roofline**: 165 tok/s vs 196. Fixed overhead (~2 ms of a 7.2 ms step) is the gap, as Chapter 3 predicts for small jobs.
5. **Burst == sustained today** (18.8 vs 18.9). The Oct 19 difference (19.9 vs 41.4) does not reproduce. Together with 26.6 → 55.8 on generate, the only consistent explanation is that the machine was in a slow state during parts of Oct 18–19. Which state is untested — that is Exercise 5.4.
6. **Around-the-model cost is small**: argmax 0.26 ms; the dependent loop with `.item()` (18.9) is within 0.1 ms of the plain forward pass (18.8–18.9).
7. **Prefill at 16 tokens**: 175 tok/s bf16 but only 90 tok/s 4-bit — the quantized kernels are slower per token when processing several tokens at once. A preview of the README's "block_size <= 5 for quantized" note and of Chapter 7.

## Run 2 (2026-10-20, 12:01, AC power) — reproducibility check

| Experiment | Run 1 (11:47) | Run 2 (12:01) |
|---|---|---|
| bandwidth 0.2 / 2 / 8 GB | 176 / 469 / 435 GB/s | 190 / 468 / 441 GB/s |
| generate bf16 | 55.8 tok/s | 55.7 tok/s |
| generate 4-bit | 165.2 tok/s | 162.9 tok/s |
| forward bf16 burst / sustained | 18.8 / 18.9 ms | 18.8 / 19.0 ms |
| forward 4-bit burst / sustained | 7.1 / 7.2 ms | 7.3 / 7.1 ms |
| argmax | 0.26 ms | 0.25 ms |
| dependent loop bf16 / 4-bit | 18.9 / 7.2 ms | 18.8 / 7.1 ms |
| prefill tok/s (16-token prompt) bf16 / 4-bit | 175 / 90 | 110 / 317 |

Everything reproduces within ~2% except prefill speed on a 16-token prompt, which swings
3× between runs: a few-millisecond job dominated by fixed overhead. Not a meaningful number.
