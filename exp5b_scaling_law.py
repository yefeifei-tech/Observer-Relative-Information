"""
Verification: n* = α · c(P_M^(0), T)
======================================
验证恢复时间 n* 精确正比于先验强度 α，
比例系数 c 只依赖初始错误形状，不依赖 α。

1. 扫描多个 α 值，测量 n*，验证 n*/α = const
2. 数值求解超越方程 (*) 得到 c_theory
3. 对比 c_theory 和 c_measured = n*/α
"""

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# ============================================================
# System (same as Exp 5)
# ============================================================
T_true = np.array([[0.8, 0.2], [0.4, 0.6]])
N = 2

evals, evecs = np.linalg.eig(T_true.T)
idx = np.argmin(np.abs(evals - 1.0))
pi = np.real(evecs[:, idx])
pi = pi / pi.sum()

P_marginal = pi @ T_true

print("=" * 65)
print("Theorem Verification: n* = α · c")
print("=" * 65)
print(f"T = {T_true.tolist()}")
print(f"π = ({pi[0]:.4f}, {pi[1]:.4f})")
print(f"P_marginal = ({P_marginal[0]:.4f}, {P_marginal[1]:.4f})")

# ============================================================
# A_M computation
# ============================================================
def compute_A_M(T_true, P_M, pi, P_marginal):
    a_m = 0.0
    for i in range(N):
        for j in range(N):
            if T_true[i,j] > 1e-15 and P_M[i,j] > 1e-15 and P_marginal[j] > 1e-15:
                a_m += pi[i] * T_true[i,j] * np.log2(P_M[i,j] / P_marginal[j])
    return a_m

# ============================================================
# Bayesian update (expected counts)
# ============================================================
def bayesian_update(n, alpha, P_M_0):
    P_M = np.zeros_like(T_true)
    for i in range(N):
        for j in range(N):
            num = n * pi[i] * T_true[i,j] + alpha * P_M_0[i,j]
            denom = sum(n * pi[i] * T_true[i,k] + alpha * P_M_0[i,k] for k in range(N))
            P_M[i,j] = num / denom
    return P_M

# ============================================================
# Find n* by scanning
# ============================================================
def find_n_star(alpha, P_M_0, n_max=2000):
    for n in range(n_max + 1):
        P_M = bayesian_update(n, alpha, P_M_0)
        a_m = compute_A_M(T_true, P_M, pi, P_marginal)
        if a_m >= 0:
            return n
    return None

# ============================================================
# Solve transcendental equation for c
# ============================================================
def A_M_at_c(c, P_M_0):
    """Compute A_M when n = c*α (α cancels out)."""
    P_M = np.zeros_like(T_true)
    for i in range(N):
        for j in range(N):
            num = c * pi[i] * T_true[i,j] + P_M_0[i,j]
            denom = sum(c * pi[i] * T_true[i,k] + P_M_0[i,k] for k in range(N))
            P_M[i,j] = num / denom
    return compute_A_M(T_true, P_M, pi, P_marginal)

def solve_c(P_M_0, c_max=100):
    """Solve A_M(c) = 0 for c > 0."""
    a0 = A_M_at_c(0, P_M_0)
    if a0 >= 0:
        return 0.0  # already positive at c=0

    for c_upper in np.logspace(-1, np.log10(c_max), 100):
        if A_M_at_c(c_upper, P_M_0) >= 0:
            break
    else:
        return None  # no crossing found

    c_star = brentq(lambda c: A_M_at_c(c, P_M_0), 0, c_upper, xtol=1e-10)
    return c_star

# ============================================================
# Test 1: Fixed P_M_0 (fully reversed), vary α
# ============================================================
P_M_0_reversed = np.array([[0.2, 0.8], [0.8, 0.2]])

print("\n" + "=" * 65)
print("Test 1: Fixed initial error (fully reversed), varying α")
print("=" * 65)

alphas_test = [2, 5, 10, 20, 50, 100, 200, 500]
print(f"\n{'α':>6s} | {'n*':>6s} | {'n*/α':>8s} | {'c_theory':>10s} | {'match':>6s}")
print("-" * 48)

c_theory = solve_c(P_M_0_reversed)
print(f"  c_theory (from equation *) = {c_theory:.6f}")

for alpha in alphas_test:
    n_star = find_n_star(alpha, P_M_0_reversed)
    if n_star is not None:
        ratio = n_star / alpha
        predicted = alpha * c_theory
        match = "OK" if abs(n_star - round(predicted)) <= 1 else "X"
        print(f"{alpha:6d} | {n_star:6d} | {ratio:8.4f} | {c_theory:10.6f} | {match:>6s}")
    else:
        print(f"{alpha:6d} | {'N/A':>6s} | {'N/A':>8s} | {c_theory:10.6f} |")

# ============================================================
# Test 2: Fixed α=10, vary P_M_0
# ============================================================
alpha_fixed = 10

initial_models = {
    'Uniform': np.array([[0.5, 0.5], [0.5, 0.5]]),
    'Slightly reversed': np.array([[0.4, 0.6], [0.6, 0.4]]),
    'Fully reversed': np.array([[0.2, 0.8], [0.8, 0.2]]),
}

print("\n" + "=" * 65)
print(f"Test 2: Fixed α = {alpha_fixed}, varying initial error")
print("=" * 65)

print(f"\n{'Model':>20s} | {'n*':>5s} | {'c_meas':>8s} | {'c_theory':>10s} | {'n*_pred':>8s} | {'match':>6s}")
print("-" * 68)

for name, P_M_0 in initial_models.items():
    n_star = find_n_star(alpha_fixed, P_M_0)
    c_th = solve_c(P_M_0)
    c_meas = n_star / alpha_fixed if n_star else None
    n_pred = round(alpha_fixed * c_th) if c_th else None

    if n_star is not None and c_th is not None:
        match = "OK" if abs(n_star - n_pred) <= 1 else "X"
        print(f"{name:>20s} | {n_star:5d} | {c_meas:8.4f} | {c_th:10.6f} | {n_pred:8d} | {match:>6s}")

# ============================================================
# Test 3: Sweep α continuously, n* vs α
# ============================================================
alpha_range = np.arange(1, 201)
n_stars_reversed = []
for alpha in alpha_range:
    ns = find_n_star(alpha, P_M_0_reversed, n_max=3000)
    n_stars_reversed.append(ns if ns is not None else np.nan)
n_stars_reversed = np.array(n_stars_reversed, dtype=float)

# ============================================================
# Test 4: c vs initial error depth
# ============================================================
betas = np.linspace(0, 1, 50)
c_values = []
A_M_0_values = []

# interpolate from perfect (T_true) to the fully-reversed extreme used above,
# so the trend curve spans the full error-depth range and passes through the marked points
T_reversed = np.array([[0.2, 0.8], [0.8, 0.2]])

for beta in betas:
    P_M_0 = (1 - beta) * T_true + beta * T_reversed
    P_M_0 = P_M_0 / P_M_0.sum(axis=1, keepdims=True)

    a_m_0 = compute_A_M(T_true, P_M_0, pi, P_marginal)
    A_M_0_values.append(a_m_0)

    if a_m_0 < 0:
        c_values.append(solve_c(P_M_0))
    else:
        c_values.append(0)

A_M_0_values = np.array(A_M_0_values)
c_values = np.array(c_values)

# ============================================================
# Plot  (clean 3-panel layout, no ASCII text box)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))

# --- Panel 1: n* vs α (linear) ---
ax1 = axes[0]
ax1.plot(alpha_range, n_stars_reversed, 'o', color='#3498db', markersize=3, alpha=0.6,
         label=r'Measured $n^*$')
ax1.plot(alpha_range, c_theory * alpha_range, 'r-', linewidth=2,
         label=rf'$n^* = \alpha\,c,\ c = {c_theory:.4f}$')
ax1.set_xlabel(r'Prior strength $\alpha$', fontsize=12)
ax1.set_ylabel(r'Critical evidence $n^*$', fontsize=12)
ax1.set_title(r'(a) $n^* = \alpha\,c$: exact linear scaling', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3)

# --- Panel 2: n*/α vs α (constant) ---
ax2 = axes[1]
ratios = n_stars_reversed / alpha_range
ax2.plot(alpha_range, ratios, '-', color='#3498db', linewidth=1.5, alpha=0.8,
         label=r'$n^*/\alpha$ (measured)')
ax2.axhline(y=c_theory, color='red', linestyle='--', linewidth=2,
            label=rf'$c = {c_theory:.4f}$ (theory)')
ax2.set_xlabel(r'Prior strength $\alpha$', fontsize=12)
ax2.set_ylabel(r'$n^* / \alpha$', fontsize=13)
ax2.set_title(r'(b) $n^*/\alpha = c$ (independent of $\alpha$)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(c_theory - 0.5, c_theory + 0.5)

# --- Panel 3: c vs initial error depth ---
ax3 = axes[2]
mask = A_M_0_values < 0
ax3.plot(np.abs(A_M_0_values[mask]), c_values[mask], '-', color='#8e44ad', linewidth=2.5)
for name, P_M_0 in initial_models.items():
    a0 = compute_A_M(T_true, P_M_0, pi, P_marginal)
    if a0 < 0:
        c_val = solve_c(P_M_0)
        ax3.plot(abs(a0), c_val, 'o', color='#8e44ad', markersize=10,
                 markeredgecolor='black', markeredgewidth=1.5, zorder=5)
        ax3.annotate(name.split(' ')[0], xy=(abs(a0), c_val),
                     xytext=(abs(a0) - 0.02, c_val + 0.18), fontsize=9,
                     ha='right', color='#6c3483', fontweight='bold')
ax3.set_xlabel(r'Initial error depth $|A_M(0)|$ (bits)', fontsize=12)
ax3.set_ylabel(r'Scaling constant $c$', fontsize=13)
ax3.set_title(r'(c) $c$ grows with initial error depth', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

plt.suptitle(r'Recovery Scaling Law:  $n^* = \alpha \cdot c$',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('exp5b_scaling_law.png', dpi=150, bbox_inches='tight')
plt.savefig('exp5b_scaling_law.pdf', bbox_inches='tight')
print(f"\nSaved: exp5b_scaling_law.png / .pdf")
plt.show()
