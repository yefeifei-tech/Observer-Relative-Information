# Observer-Relative Information — Reproducibility Code

Computational experiments for the paper:

> **Negative Knowledge and Observer-Relative Information: Recovering Shannon Mutual Information as a Special Case**
> Evan Ye (Independent Researcher)

This repository contains the Python scripts that generate every figure in the paper. The experiments on synthetic Markov chains (`exp1`–`exp6`) compute all quantities **exactly** — no sampling, no random seed, no external data — so they reproduce bit-for-bit. `exp7_bigram.py` validates the framework on real text (letter bigrams from a public-domain book downloaded at run time), and `fig1_framework_overview.py` draws the schematic overview figure.

> Figure numbering differs between manuscript versions of this work, so the scripts below are listed by topic rather than by a fixed figure number.

## Script map

| Script | Topic |
|--------|-------|
| `fig1_framework_overview.py`   | Schematic overview of the observer-relative framework |
| `exp1_observer_spectrum.py`    | Observer spectrum: `I_ORI` varies while `I_Shannon` is fixed; degeneracy at λ = 0 |
| `exp2_negative_knowledge.py`   | Negative knowledge: observers whose model is worse than the transition-ignorant baseline (`A_M < 0`) |
| `exp3_constraint_discovery.py` | Constraint structure: marginal contribution to `R_C` distinguishes structure from noise |
| `exp4_observer_dependent.py`   | Observer-dependent information: one system, four observers, four values of `I_ORI` |
| `exp5_bayesian_recovery.py`    | Recovery from negative knowledge via Bayesian updating; critical evidence `n*` |
| `exp5b_scaling_law.py`         | Recovery scaling law `n* = α·c` and its numerical verification |
| `exp6_phase_diagram.py`        | Knowledge phase diagram in a coarse-grained double-well potential |
| `exp7_bigram.py`               | Empirical validation on natural-language letter bigrams (real data) |

## Key quantities (notation matches the paper)

- **`I_ORI`** — Observer-Relative Information, `E_X[ D_KL( P(Y|X) ‖ P_M(Y|X) ) ]`.
- **`I_Shannon`** — Shannon mutual information `I(X;Y)`; the special case `P_M = P(Y)`.
- **`A_M`** — Observer Advantage, `A_M = I_Shannon − I_ORI`. Can be negative ("negative knowledge").
- **`R_C`** — Constraint Residual, the conditional entropy `H(Y|X)`.

> **Note on two distinct uses of Greek letters.** In `exp1`, **λ** is an *interpolation coordinate* mixing the true and marginal models (λ = 1 perfect, λ = 0 transition-ignorant, λ < 0 negative knowledge). In `exp5` / `exp5b`, **α** is the *Bayesian prior strength* of the Dirichlet–Multinomial update, and the scaling law reads `n* = α·c`. These are unrelated parameters; the code keeps them separate, matching the paper.

## Requirements

```
numpy
matplotlib
scipy        # used only by exp5b_scaling_law.py (root finding)
```

Install with:

```bash
pip install -r requirements.txt
```

Tested with Python 3.8+, NumPy 1.20+, Matplotlib 3.3+, SciPy 1.5+.

## Running

Each script is independent and writes its figure (`.png` and `.pdf`) to the current directory:

```bash
python fig1_framework_overview.py
python exp1_observer_spectrum.py
python exp2_negative_knowledge.py
python exp3_constraint_discovery.py
python exp4_observer_dependent.py
python exp5_bayesian_recovery.py
python exp5b_scaling_law.py
python exp6_phase_diagram.py
python exp7_bigram.py   # downloads a public-domain text at run time
```

To regenerate all figures at once:

```bash
python run_all.py
```

Each script also prints the numerical results (information values, critical evidence `n*`, scaling constant `c`, etc.) to the console.

## Reproducibility notes

- **Deterministic (`exp1`–`exp6`).** Every result is an exact expectation over the stationary distribution of a fixed transition matrix. No random number generator is used, so output is identical across runs and platforms.
- **Experiment 7 (real data).** `exp7_bigram.py` downloads a public-domain book from Project Gutenberg at run time and estimates the bigram statistics from it; if the download is unavailable it falls back to a short embedded passage. The numbers depend on the text, but the qualitative predictions — degeneracy at the transition-ignorant observer, and `A_M < 0` for the reversed-bigram observer — hold for any English text.
- **Self-contained.** Each script defines its own system and helper functions; none depends on the output of another.
- **Console encoding.** A couple of scripts print Unicode (`✓`, `×`) and Chinese annotations. On Windows terminals set `PYTHONIOENCODING=utf-8` if you see an encoding error.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{ye_ori,
  title   = {Negative Knowledge and Observer-Relative Information:
             Recovering Shannon Mutual Information as a Special Case},
  author  = {Ye, Evan},
  year    = {2026},
  note    = {arXiv:XXXX.XXXXX}
}
```

(Replace the arXiv identifier once assigned.)

## License

Released under the MIT License — see [LICENSE](LICENSE).
