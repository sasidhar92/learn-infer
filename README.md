# Inference Engineering Notebook

A short, hands-on book about why language models answer at the speed they do and how to
make them faster. Every number in it was measured on one laptop (Apple M4 Max, 128 GB) and
every experiment can be re-run in about a minute.

**Read the book:** https://sasidhar92.github.io/learn-infer/ (or open [`docs/index.html`](docs/index.html) locally).

Status: Part I complete through Chapter 5 (how a model makes a token, memory bandwidth,
the roofline, profiling a token). Chapters 6 onward are outlined and fill in as the
experiments are done.

## Reproduce the numbers

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mlx==0.32.0 mlx-lm==0.31.3
experiments/verify/run_all.sh
```

The scripts download `mlx-community/Qwen3-4B-bf16` and `mlx-community/Qwen3-4B-4bit` from
Hugging Face on first run. Raw output lands in `experiments/verify/RESULTS_RAW.txt`.

## Layout

- `book/index.html` — the book source (body fragment); `book/build.py` wraps it into `docs/index.html` for GitHub Pages
- `experiments/verify/` — numbered experiment scripts (00–06) and the runner
- `experiments/00_measure/` — earlier hand-written versions of the same experiments
- `CURRICULUM.md` — the full 10-module plan the book follows
- `tasks/` — working notes

The DFlash reference implementation used in later chapters is a separate repository:
https://github.com/z-lab/dflash (cloned locally into `dflash/`, not committed here).
