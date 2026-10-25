"""Reusable before/after harness for decode benchmarks on Apple Silicon (MLX).

Usage from an experiment:
    from harness import bench, report
    stats = bench("AR bf16", run_fn, prompts, warmup=1, repeats=3)
    report([stats_a, stats_b])

`run_fn(prompt_tokens) -> (num_generated_tokens, decode_seconds, extra_dict)`
"""
import statistics
import time
from dataclasses import dataclass, field

import mlx.core as mx


@dataclass
class RunStats:
    name: str
    tpot_ms: list[float] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    peak_gb: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def tps(self) -> float:
        return 1000.0 / statistics.mean(self.tpot_ms)

    @property
    def tpot_std_ms(self) -> float:
        return statistics.pstdev(self.tpot_ms) if len(self.tpot_ms) > 1 else 0.0


def bench(name, run_fn, prompts, warmup=1, repeats=1, seed=0):
    mx.random.seed(seed)
    for p in prompts[:warmup]:
        run_fn(p)
    mx.reset_peak_memory()
    stats = RunStats(name)
    for _ in range(repeats):
        for p in prompts:
            n, secs, extra = run_fn(p)
            if n > 0:
                stats.tpot_ms.append(1000.0 * secs / n)
                stats.tokens.append(n)
            for k, v in extra.items():
                stats.extra.setdefault(k, []).append(v)
    stats.peak_gb = mx.get_peak_memory() / 1e9
    return stats


def report(all_stats, baseline_index=0):
    base = all_stats[baseline_index]
    print(f"{'config':<28}{'tok/s':>9}{'TPOT ms':>10}{'±':>7}{'speedup':>9}{'peak GB':>9}")
    for s in all_stats:
        print(
            f"{s.name:<28}{s.tps:>9.1f}{statistics.mean(s.tpot_ms):>10.2f}"
            f"{s.tpot_std_ms:>7.2f}{base.tps and s.tps / base.tps:>9.2f}x{s.peak_gb:>8.1f}"
        )


def timed(fn):
    tic = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - tic
