import statistics
import time 

from huggingface_hub import snapshot_download

REPO = "mlx-community/Qwen3-4B-4bit"

PATTERNS = ["*.json"]
RUNS = 5

def timeit(**kwargs):
  samples_ms = []
  for _ in range(RUNS):
    start = time.perf_counter()
    snapshot_download(REPO, allow_patterns = PATTERNS, **kwargs)
    samples_ms.appen((time.perf_counter()-start)*1000)
  return statistics.median(samples_ms), sample_ms

snapshot_download(REPO, allow_patterns = PATTERNS)
network_median, network_samples = timeit()
local_median, local_samples = timeit(local_files_only=True)

print(f"network check median {network_median:8.2f} ms samples {}")