# Observer-Relative Information — Reproducibility Code

Computational experiments for the paper:

> **Negative Knowledge and Observer-Relative Information: Recovering Shannon Mutual Information as a Special Case**
> Evan Ye (Independent Researcher)

This repository contains the seven self-contained Python scripts that generate every figure in the paper. All quantities are computed **exactly** from analytically specified Markov chains — there is no sampling, no random seed, and no external data. Running a script reproduces the corresponding figure bit-for-bit.

## Figure map

| Script | Paper figure | Topic |
|--------|--------------|-------|
| `exp1_observer_spectrum.py`    | Fig. 1 | Observer spectrum: `I_ORI` varies while `I_Shannon` is fixed; degeneracy at λ = 0 |
| `exp2_negative_knowledge.py`   | Fig. 2 | Negative knowledge: observers whose model is worse than ignorance (`A_M < 0`) |
| `exp3_constraint_discovery.py` | Fig. 3 | Constraint discovery: marginal contribution to `R_C` distinguishes structure from noise |
| `exp4_observer_dependent.py`   | Fig. 4 | Observer-dependent information: one system, four observers, four values of `I_ORI` |
| `exp5_bayesian_recovery.py`    | Fig. 5 | Recovery from negative knowledge via Bayesian updating; critical evidence `n*` |
| `exp5b_scaling_law.py`         | Fig. 6 | Recovery scaling law `n* = α·c` (Theorem 2) and its numerical verification |
| `exp6_phase_diagram.py`        | Fig. 7 | Knowledge phase diagram in a coarse-grained double-well potential |

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
python exp1_observer_spectrum.py
python exp2_negative_knowledge.py
python exp3_constraint_discovery.py
python exp4_observer_dependent.py
python exp5_bayesian_recovery.py
python exp5b_scaling_law.py
python exp6_phase_diagram.py
```

To regenerate all figures at once:

```bash
python run_all.py
```

Each script also prints the numerical results (information values, critical evidence `n*`, scaling constant `c`, etc.) to the console.

## Reproducibility notes

- **Deterministic.** Every result is an exact expectation over the stationary distribution of a fixed transition matrix. No random number generator is used, so output is identical across runs and platforms.
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
