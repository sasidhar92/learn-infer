import time
import mlx.core as mx

#n= 100_000_000
#x = mx.ones((n,), dtype = mx.bfloat16)

n_rows, n_cols = 4000, 1_000_000

x = mx.ones((n_rows, n_cols), dtype = mx.bfloat16)

mx.eval(x)

mx.eval(mx.sum(x))

tic =  time.perf_counter()
for _ in range(5):
  mx.eval(mx.sum(x))

secs = (time.perf_counter()-tic)/5

#bytes_read = n*2

bytes_read = n_rows*n_cols *2 

print(f"read {bytes_read/1e9:.1f} GB in {secs*1000:.1f} ms")

print(f"bandwidth: {bytes_read/secs/ 1e9:.0f} GB/s")

