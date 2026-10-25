# learn-infer — task tracker

## Session 1 (2026-10-18): orientation + setup
- [x] Read DFlash paper (arXiv 2602.06036 v2), project blog, DFlash 2 blog, LMSYS Spec V2 blog
- [x] Clone z-lab/dflash into ./dflash and map the code (4 files, ~2.2k LOC)
- [x] Write module-by-module curriculum -> CURRICULUM.md
- [x] Python 3.12 venv with mlx + mlx-lm + dflash[local]
- [x] Download Qwen3-4B (bf16 + 4-bit) and z-lab/Qwen3-4B-DFlash-b16
- [x] Smoke test: baseline tok/s vs DFlash tok/s on one prompt

## Next up
- [x] Module 0: measurement harness + roofline (experiments/00_measure)
- [ ] Module 1: "forward pass cost is flat in #tokens" experiment

## Review
(filled in at end of each module)

## Module 0 review
Harness works (experiments/harness.py). Decode is bandwidth-bound (4-bit = 3.2× faster for 3.56× fewer bytes)
but both models sit at ~40% of the measured 425 GB/s ceiling; the loss is inside MLX bf16 GEMV kernels,
not the Python loop. Sync timing over-reports vs the async_eval generation loop.

## OSS contribution (2026-10-21): mlx-lm issue #1649
Target: ml-explore/mlx-lm#1649 — load() calls snapshot_download() without local_files_only,
so a fully cached model still pays a Hub round-trip (twice per load). Still on main (utils.py _download).
- [ ] Reproduce with numbers on M4 Max (mlx-community/Qwen3-4B-4bit, hf_hub 1.27.0)
- [ ] Post the numbers as a comment on #1649 (Sasi writes the text — repo forbids AI-written posts)
- [ ] Fork, branch, 6-line fix in mlx_lm/utils.py (local_files_only first, fall back on LocalEntryNotFoundError)
- [ ] Run existing tests, open PR linking #1649, disclose AI assistance per CONTRIBUTING.md
