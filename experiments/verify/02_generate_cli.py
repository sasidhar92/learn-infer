"""The README-style baseline: the exact mlx_lm.generate command, bf16 then 4-bit."""
import subprocess, sys
for model in ["mlx-community/Qwen3-4B-bf16", "mlx-community/Qwen3-4B-4bit"]:
    cmd = [sys.executable.replace("python", "mlx_lm.generate"), "--model", model,
           "--prompt", "Explain why the sky is blue.", "--max-tokens", "200", "--temp", "0"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    stats = [l for l in out.splitlines() if l.startswith(("Prompt:", "Generation:", "Peak"))]
    print(model.split("/")[-1])
    for l in stats: print("   ", l)
