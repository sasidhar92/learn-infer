"""Record the machine state. Run before and after the experiments."""
import subprocess, datetime
def sh(c): return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
print("time       :", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print("power      :", sh("pmset -g batt | head -1"))
print("low power  :", sh("pmset -g | grep -i powermode"))
print("thermal    :", sh("pmset -g therm | grep -i warning | xargs"))
print("top CPU    :", sh("ps aux | sort -nrk 3 | head -3 | awk '{print $3\"% \"$11}' | xargs"))
