# Inference Engineering through DFlash — a hands-on curriculum

Target: understand *DFlash: Block Diffusion for Flash Speculative Decoding*
(Chen, Liang, Liu — arXiv 2602.06036, ICML 2026) inside and out, and come out
able to measure "before vs after" for any inference technique on this Mac.

Hardware: MacBook Pro, Apple M4 Max, 128 GB unified memory (~546 GB/s).
Stack: Python 3.12 venv at `.venv/`, MLX 0.32 + mlx-lm 0.31.3, `dflash` installed
editable from `./dflash`. Models live in the HF cache (`~/.cache/huggingface`).

Sources you will keep coming back to:
- Paper (HTML): https://arxiv.org/html/2602.06036v2
- Talk by Jian Chen (53 min): https://www.youtube.com/watch?v=TS_W8q_zXA8
- Code: `dflash/dflash/model_mlx.py` (the one you'll read most), `benchmark.py`
- DFlash blog: https://z-lab.ai/projects/dflash/  ·  DFlash 2 blog: https://inco.ai/blog/dflash2/
- SGLang Spec V2 blog: https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2

Talk chapters (so you can watch the slice that matches each module):
0:00 intro · 0:33 background · 1:52 spec-decoding challenges · 2:58 block diffusion drafting ·
5:32 injecting target features · 6:34 implementation · 11:13 training & inference ·
17:01 evaluation · 20:19 adoption · 22:04 DFlash 2 · 23:40 path selection ·
29:11 convolution for suffix decay · 33:56 DFlash 2 results · 40:10 Splash engine · 45:43 Q&A

How each module works: **Concept → Why → Read (paper/code) → Experiment → Write-up**.
Every experiment lives in `experiments/NN_name/` with a `run.py` and a `RESULTS.md`
that records the command, hardware, numbers and one paragraph of interpretation.
That folder is the habit you're building.

---

## Module 0 — Measurement first: what does "tokens/sec" even mean?

**Concept.** Decoding an LLM has two phases. *Prefill* processes the prompt in one
big parallel pass (compute-bound). *Decode* produces one token per forward pass
(memory-bound). The number people quote, generation tok/s, is `1 / TPOT`
(time per output token). Peak memory matters too.

**Why.** Every speedup claim in the paper is "baseline TPOT / DFlash TPOT". If you
can't produce a clean, repeatable baseline number, you cannot evaluate anything.

**Read.** `dflash/benchmark.py:188-216` (`_make_decode_metrics`, `_print_decode_summary`):
it is tiny, and it is literally the README's before/after. Note: it averages TPOT
per request, not tokens, and prints an acceptance-length histogram.

**Experiment 0.1 — the roofline sanity check.**
Run plain `mlx_lm.generate` on Qwen3-4B bf16 and 4-bit. Record tok/s.
Then compute the bandwidth bound: `bytes_of_weights / 546e9`. For 4B params in bf16
that's ~8 GB → ~68 tok/s ceiling; 4-bit ~2.3 GB → ~235 tok/s ceiling.
Question to answer in RESULTS.md: what fraction of the ceiling do you hit, and why
is 4-bit not 3.5× faster than bf16 in practice?

**Experiment 0.2 — a reusable harness.** Build `experiments/harness.py`: given a
model id, a list of prompts, and a *generate function*, it does warmup, N runs,
and reports mean/std of TPOT, tok/s, peak memory, with a fixed seed. You will plug
AR decoding, classic speculative decoding and DFlash into the same harness later.
This is the artifact that makes you "capable of doing it for other implementations".

Checkpoint: you can state your Mac's bf16 and 4-bit decode ceilings from memory.

---

## Module 1 — The free lunch: a forward pass over k tokens costs about the same as over 1

**Concept.** In decode, each step reads *all* the weights from memory to produce
one token. The GPU is idle most of the time. If you feed 8 tokens instead of 1,
you read the weights once and do 8× the math, which is nearly free until you
become compute-bound.

**Why.** This single fact is the reason speculative decoding exists at all. DFlash's
whole design (bigger drafter, 16-token blocks) is an argument about *where* on this
curve you want to sit.

**Read.** Paper §1 and the latency model `L = (T_draft + T_verify) / τ` (§3).
Talk 0:33–2:58.

**Experiment 1.1 — the flat curve.** With the Qwen3-4B target loaded in MLX, time a
single forward pass with a KV cache for input lengths 1, 2, 4, 8, 16, 32, 64, 128.
Plot latency vs k (bf16 and 4-bit). Find the knee. Note what the README says:
on quantized models MLX's matmul kernel is inefficient at verify widths > 5,
so expect the 4-bit knee to appear early. That's a *kernel* fact, not a model fact.

Checkpoint: you can say "on my Mac, verifying a block of B tokens costs X× one token"
for bf16 and 4-bit.

---

## Module 2 — Speculative decoding from scratch (draft → verify → accept)

**Concept.** A cheap *draft* model proposes k tokens; the *target* model verifies all
k+1 positions in one forward pass. Greedy verification: accept the longest prefix
that matches the target's argmax, then take the target's token at the first
mismatch as a free "bonus" token. Acceptance length τ = accepted + bonus.
Lossless: output is exactly what the target would have produced.

**Why.** τ is the numerator of every speedup. Understanding exactly how acceptance
and rollback work is the foundation for reading DFlash's loop.

**Read.** `model_mlx.py:726-907` (`_stream_generate`) — skip the DFlash-specific
parts for now, follow: build block → draft → verify → compute `accepted` →
`new_tokens = d_list[:accepted] + [bonus]` → trim the KV cache by `bs - accepted - 1`.

**Experiment 2.1 — classic SD with a real small drafter.** mlx-lm already supports
speculative decoding with a separate draft model (`--draft-model`,
`--num-draft-tokens`). Use Qwen3-0.6B as the draft for Qwen3-4B. Measure τ and
speedup on 10 GSM8K prompts. This is the "before" that DFlash claims to beat.

**Experiment 2.2 — write your own.** ~80 lines in MLX: implement greedy
draft-verify-rollback yourself with mlx-lm's `KVCache` trimming. Assert token-for-token
equality with plain greedy decoding. (Until it's bit-identical, it's not speculative
decoding, it's a different model.)

**Experiment 2.3 — lossless sampling.** Implement rejection sampling
(`model_mlx.py:68-107` `_rejection_sample`) and verify empirically: for a fixed
prompt, sample 2,000 next tokens with and without SD at temperature 1 and compare
the two histograms.

Checkpoint: you can explain why the bonus token is free and why τ ≤ k+1.

---

## Module 3 — Why autoregressive drafters hit a ceiling (the EAGLE-3 problem)

**Concept.** EAGLE/EAGLE-3/Medusa are state of the art *autoregressive* drafters:
a 1-layer head on top of the target's features, drafting one token per step. Drafting
k tokens costs k sequential draft passes. To keep that cheap they must stay
tiny (1 layer), which caps draft quality; τ stalls around 3–4 and speedups at 2–3×.
Trees of candidates (16–60 nodes) help a bit and cost more verify compute.

**Why.** This is the gap DFlash opens. You need to feel the cost curve, not just read it.

**Read.** Paper §2 (related work), Table 1 EAGLE-3 rows. Talk 1:52–2:58.

**Experiment 3.1 — the drafting-cost cliff.** With your Module 2 classic SD,
sweep `num_draft_tokens` k ∈ {1,2,3,4,6,8,12,16}. Plot τ and end-to-end speedup
vs k. τ keeps rising slowly; speedup peaks and falls because draft time grows
linearly in k. Mark the peak. That's the ceiling the paper is talking about.

Checkpoint: one sentence on why "a deeper AR drafter" is not a fix.

---

## Module 4 — Block diffusion drafting: one forward pass, a whole block

**Concept.** A *masked diffusion* model takes `[anchor, MASK, MASK, …, MASK]` and
predicts all masked positions at once (bidirectional attention inside the block,
causal attention to the context). DFlash uses it as a *single-step* denoiser:
one pass → 15 draft tokens. Drafting cost no longer grows with block size, so
you can afford a 5-layer (or 8-layer) drafter instead of 1 layer.

**Why.** "Deeper model, more tokens, less time." This flips the design constraint
of Module 3.

**Read.** Paper §3.1–3.2, Figure 1. Talk 2:58–5:32 and 6:34–11:13.
Code: `model_mlx.py` `DFlashDraftModel.hidden_states` and `DFlashAttention.__call__`
(mask: block tokens see each other + the cached context). Note `mask_token_id`
in the config and the `block = [tokens[-1]] + [mask_id]*(bs-1)` line in the loop.

**Experiment 4.1 — first DFlash run (the README before/after).**
```
dflash benchmark mlx --model mlx-community/Qwen3-4B-bf16 --draft z-lab/Qwen3-4B-DFlash-b16 \
    --dataset gsm8k --max-samples 20
```
Record baseline tok/s, DFlash tok/s, speedup, τ, histogram. Then the 4-bit target with
`--block-size 5`. Compare with the paper's Qwen3-4B numbers (τ ≈ 6.5, 4.9× on H100-class)
and explain every difference you see in terms of Modules 0–1.

**Experiment 4.2 — draft time is flat.** Instrument the loop (or copy it into
`experiments/`) to time draft pass and verify pass separately per step. Sweep
`--block-size` 4, 8, 12, 16. Draft time should be ~flat; verify time follows
your Module 1 curve; τ rises with block size. Compute predicted speedup from
`L = (T_draft+T_verify)/τ` and check it against the measured one.

Checkpoint: you can draw the per-step timeline for AR, classic SD, and DFlash.

---

## Module 5 — "The target knows best": feature fusion and KV injection

**Concept.** The target model's hidden states already encode a lot about the next
several tokens. DFlash grabs hidden states from 5 target layers (spread from layer 2
to the third-to-last), concatenates them, projects to one *context feature*
(`fc` + RMSNorm), and injects it as extra Key/Value entries into **every** draft layer's
attention — not into the input embedding like EAGLE-3. Embedding and LM head are
shared with the target and frozen.

**Why.** Without target features a 5-layer block-diffusion drafter only reaches
~2.3–3.7× (paper Table 10 / blog). KV injection beats input fusion because the
signal doesn't dilute with depth (Table 9: 2.9× → 3.3× on GSM8K).

**Read.** Paper §3.2 and Tables 7, 9, 10. Talk 5:32–6:34.
Code: `_LayerHook`/`_patch_model` (how hidden states are captured from the target
during the *verify* pass — notice they come for free), `DFlashDraftModel.__init__`
(`fc`: `5*hidden → hidden`), `DFlashAttention.__call__` (where `x_ctx` turns into
K and V), `build_target_layer_ids` in `model.py:25`.

**Experiment 5.1 — break the conditioning.** Monkey-patch the drafter so the context
feature is zeros (or hidden states from the wrong layers). Measure τ. This is a
one-line ablation that reproduces the paper's central claim on your laptop.

**Experiment 5.2 — which layers matter?** Replace one of the 5 target layers at a
time with a neighbour and watch τ. (Small effects expected; the point is learning to
run a controlled ablation.)

Checkpoint: explain in two sentences why conditioning on the target is "free" at
inference time.

---

## Module 6 — Losslessness in practice: cache rollback and hybrid models

**Concept.** After verification you must undo the rejected tokens: trim the target's
KV cache and the draft's cache by `bs - accepted - 1`. For plain Transformers that's
a slice. Qwen3.5/3.6/Gemma-4 mix attention with linear-recurrent layers
(Gated DeltaNet), whose state can't be sliced — `_GDNStateCapture` snapshots and
rolls back the recurrent state instead.

**Why.** Correctness. A speculative decoder that leaks a rejected token into state is
silently a different model. This is where production bugs hide.

**Read.** `model_mlx.py` `_trim_recent_cache`, `_GDNStateCapture`, `can_trim_prompt_cache`.

**Experiment 6.1 — prove losslessness.** Greedy: assert DFlash output ==
baseline output for 50 prompts (Qwen3-4B). Then do the same for Qwen3.5-4B with
`z-lab/Qwen3.5-4B-DFlash` (hybrid model → rollback path). Any mismatch is a bug or
a float-nondeterminism story — investigate which.

Checkpoint: you know what state a decoder has, and how each kind is rolled back.

---

## Module 7 — Training the drafter (the capstone)

**Concept.** Data: ~800K prompts whose responses were *regenerated by the target*
(so the drafter learns the target's distribution, not the dataset's). Each sequence
gets random *anchor* positions; from each anchor a block of masked tokens is trained
in one pass with a sparse attention mask. Loss: cross-entropy with exponential
position decay `w_k = exp(-(k-1)/γ)` (later positions are harder and noisier).
Only the 5 draft layers + `fc` train; AdamW 6e-4, 6 epochs, cosine, seq len 3072.

**Why.** Anchor sampling matches inference (you always start from a clean bonus
token): Table 13 shows 4.13× → 4.69× from this trick alone. Regenerated responses
are what make the drafter a model *of the target*.

**Read.** Paper §3.3 + Appendix (anchor sampling, loss decay figure). Talk 11:13–17:01.

**Experiment 7.1 — train a toy DFlash in MLX.** Target: Qwen3-0.6B (or 1.7B).
Drafter: 2 layers, block 8, γ=4. Data: 5–20K GSM8K/Alpaca prompts with responses
generated by the target (that's a Module 0 harness job). Train on-device with
features computed online. Plot τ on held-out prompts vs training steps.
You will not match z-lab's numbers; you *will* understand every design decision.

**Experiment 7.2 — ablate what you built.** Uniform vs anchor masking; flat vs
decayed loss; input fusion vs KV injection. Three small runs, three plots.

Checkpoint: you can read `z-lab/Qwen3-4B-DFlash-b16/config.json` and explain every field.

---

## Module 8 — Measuring like the paper does (and spotting bad benchmarks)

**Concept.** Report acceptance length *and* end-to-end speedup, always: two drafters
with equal τ can differ 1.5× in speed. Results vary a lot by task (math ≫ chat:
Qwen3-8B 6.2× on MATH-500 vs 2.3× on Alpaca). Speedups shrink with concurrency
(verification becomes compute-bound). Quantization changes the verify-width knee.
Specify hardware, precision, decoding mode, max tokens, warmup, seeds.

**Why.** This is the module that turns you into someone who can evaluate the *next*
paper's README.

**Read.** Paper §4 Tables 1–5, Table 8 (block size mismatch), LMSYS blog "lessons".

**Experiment 8.1 — the task sweep.** `dflash benchmark mlx` on gsm8k, humaneval,
mt-bench, 20 samples each, bf16 and 4-bit. Make the table the README doesn't show.
**Experiment 8.2 — block-size mismatch.** Drafter trained at 16, run at 8, 5, 4.
Reproduce the shape of Table 8 on your hardware.
**Experiment 8.3 — a benchmark checklist.** Write `BENCHMARK_CHECKLIST.md`: the
10 things a before/after must state. Use it on every future experiment.

Checkpoint: given someone else's speedup table, you can list what's missing.

---

## Module 9 — DFlash 2: fixing what the block gets wrong

**Concept.** DFlash 1 picks the top-1 token at each block position *independently*,
so adjacent picks can be locally incoherent (repeated words), and accuracy decays
toward the end of the block ("suffix decay": recall@1 ~85% at position 0 → ~73% at
position 6). DFlash 2 adds (a) a *candidate selector*: keep top-16 per position,
score adjacent pairs with a low-rank bilinear term, walk one coherent path; and
(b) a *two-tap dynamic depthwise convolution* around each sublayer so each position
mixes in its predecessor. Cost ≈ +1.3% cycle latency; gain ≈ +21% τ.

**Why.** An oracle choosing among the top-16 would lift τ from 4.27 to 6.79 — the
headroom was in *selection*, not in a bigger model.

**Read.** DFlash 2 blog; talk 22:04–40:10. Code: `CandidateSelector.select`,
`GroupedDynamicCausalConv`, `DFlash2DecoderLayer` in `model_mlx.py`.

**Experiment 9.1 — see suffix decay yourself.** Using DFlash 1 on Qwen3-4B, log
per-position accept/reject across 50 prompts → recall-by-position curve. Also
compute "oracle" recall@16 by checking whether the target's token is in the drafter's
top-16 at each position. You will reproduce the blog's motivating figure.
**Experiment 9.2 — run DFlash 2.** `mlx-community/Qwen3.8-27B-4bit` +
`z-lab/Qwen3.8-27B-DFlash2`, `--block-size 5 --draft-bits 4`. Compare histogram shape
against DFlash 1.

Checkpoint: explain the selector's scoring function in one line.

---

## Module 10 — From laptop to serving (reading module)

Spec V2 in SGLang, overlap scheduling, sharing the draft KV with the radix cache,
why speedups fall at concurrency 32, MTP vs DFlash, vLLM integration, llama.cpp PR.
You can't run B200 numbers here, but you can run the llama.cpp DFlash PR on the Mac
as a second engine and compare against MLX with your Module 0 harness.

---

## Suggested pace

| Week | Modules | Output |
|---|---|---|
| 1 | 0, 1 | harness.py, roofline numbers, flat-curve plot |
| 2 | 2, 3 | your own greedy SD, rejection sampling, k-sweep |
| 3 | 4, 5 | first DFlash before/after, draft-vs-verify timing, zero-feature ablation |
| 4 | 6, 8 | losslessness proof, task sweep table, checklist |
| 5–6 | 7 | toy DFlash drafter trained on the Mac |
| 7 | 9, 10 | suffix-decay curve, DFlash 2 run, serving notes |

Rule for every module: no RESULTS.md, not done.
