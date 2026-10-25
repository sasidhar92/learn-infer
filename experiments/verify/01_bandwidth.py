"""Achievable memory bandwidth: time a sum over piles of 0.2, 2 and 8 GB (bf16)."""
import time
import mlx.core as mx

def measure(rows, cols):
    x = mx.ones((rows, cols), dtype=mx.bfloat16)
    mx.eval(x)
    mx.eval(mx.sum(x))                      # warm-up
    tic = time.perf_counter()
    for _ in range(5):
        mx.eval(mx.sum(x))
    secs = (time.perf_counter() - tic) / 5
    nbytes = rows * cols * 2
    print(f"{nbytes/1e9:4.1f} GB read in {secs*1000:6.2f} ms -> {nbytes/secs/1e9:5.0f} GB/s")

measure(100, 1_000_000)      # 0.2 GB
measure(1000, 1_000_000)     # 2 GB
measure(4000, 1_000_000)     # 8 GB
