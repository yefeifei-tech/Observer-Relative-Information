"""
Regenerate every figure in the paper.

Runs each experiment script in turn with a non-interactive Matplotlib backend
so no plot windows pop up. Each script writes its own .png and .pdf to the
current directory and prints its numerical results.
"""

import os
import subprocess
import sys

SCRIPTS = [
    "fig1_framework_overview.py",
    "exp1_observer_spectrum.py",
    "exp2_negative_knowledge.py",
    "exp3_constraint_discovery.py",
    "exp4_observer_dependent.py",
    "exp5_bayesian_recovery.py",
    "exp5b_scaling_law.py",
    "exp6_phase_diagram.py",
    "exp7_bigram.py",
]

env = dict(os.environ)
env["MPLBACKEND"] = "Agg"          # render to file, no GUI window
env["PYTHONIOENCODING"] = "utf-8"  # allow Unicode in console output

here = os.path.dirname(os.path.abspath(__file__))

for script in SCRIPTS:
    print(f"\n{'='*70}\nRunning {script}\n{'='*70}")
    result = subprocess.run([sys.executable, script], cwd=here, env=env)
    if result.returncode != 0:
        sys.exit(f"FAILED: {script} (exit code {result.returncode})")

print("\nAll figures regenerated successfully.")
