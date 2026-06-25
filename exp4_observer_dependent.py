"""
Experiment 4: Observer-Dependent Information
=============================================
论文标题的直接验证：Information is Observer-Relative.

同一个系统，同一个联合分布，四个不同的观察者
测出四个不同的 I_ORI，但 I_Shannon 完全相同。

系统：3 状态 Markov chain
观察者：物理学家（接近完美）、统计学家（方向对但不精确）、门外汉（均匀猜测）
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. 定义系统：3 状态 Markov chain
# ============================================================

T = np.array([
    [0.70, 0.20, 0.10],
    [0.10, 0.60, 0.30],
    [0.30, 0.10, 0.60]
])

state_names = ['A', 'B', 'C']
N = 3

# 稳态分布
eigenvalues, eigenvectors = np.linalg.eig(T.T)
idx = np.argmin(np.abs(eigenvalues - 1.0))
pi = np.real(eigenvectors[:, idx])
pi = pi / pi.sum()

# 边缘分布
P_marginal = pi @ T

print("=" * 65)
print("Experiment 4: Observer-Dependent Information")
print("=" * 65)
print(f"\n系统：3 状态 Markov chain")
print(f"转移矩阵 T:")
for i in range(N):
    row = "  ".join([f"P({state_names[i]}→{state_names[j]})={T[i,j]:.2f}" for j in range(N)])
    print(f"  {row}")
print(f"\n稳态分布 π = {dict(zip(state_names, [f'{p:.4f}' for p in pi]))}")
print(f"边缘分布 P(S_{{t+1}}) = {dict(zip(state_names, [f'{p:.4f}' for p in P_marginal]))}")

# ============================================================
# 2. 核心计算函数
# ============================================================

def kl_divergence(p, q):
    """D_KL(p || q)"""
    result = 0.0
    for pi_val, qi_val in zip(p, q):
        if pi_val > 1e-15:
            if qi_val < 1e-15:
                return float('inf')
            result += pi_val * np.log2(pi_val / qi_val)
    return result

def compute_I_CET(T_true, P_M, pi):
    """I_CET = E[D_KL(P_true || P_M)]"""
    I = 0.0
    for s in range(len(pi)):
        dkl = kl_divergence(T_true[s], P_M[s])
        if dkl == float('inf'):
            return float('inf')
        I += pi[s] * dkl
    return I

def compute_A_M(I_shannon, I_cet):
    """A_M = I_Shannon - I_ORI (Observer Advantage)"""
    return I_shannon - I_cet

# I_Shannon
P_M_ignorant = np.tile(P_marginal, (N, 1))
I_shannon = compute_I_CET(T, P_M_ignorant, pi)

print(f"\nI_Shannon = {I_shannon:.6f} bits (系统的客观约束强度)")

# ============================================================
# 3. 定义三个观察者
# ============================================================

observers = {
    'Physicist\n(Expert)': {
        'P_M': np.array([
            [0.72, 0.19, 0.09],  # 接近真实，微小偏差
            [0.11, 0.58, 0.31],
            [0.29, 0.11, 0.60]
        ]),
        'color': '#2ecc71',
        'icon': '🔬',
        'desc': 'Near-perfect model (small calibration errors)',
        'short': 'Expert'
    },
    'Statistician\n(Partial)': {
        'P_M': np.array([
            [0.50, 0.30, 0.20],  # 方向对，但精度差
            [0.20, 0.45, 0.35],
            [0.35, 0.20, 0.45]
        ]),
        'color': '#3498db',
        'icon': '📊',
        'desc': 'Correct direction, imprecise magnitudes',
        'short': 'Partial'
    },
    'Outsider\n(Ignorant)': {
        'P_M': np.tile(P_marginal, (N, 1)),  # 边缘分布 = 转移无知
        'color': '#f39c12',
        'icon': '❓',
        'desc': 'Uses marginal distribution (transition ignorant)',
        'short': 'Ignorant'
    },
    'Conspiracy\n(Wrong)': {
        'P_M': np.array([
            [0.10, 0.10, 0.80],  # 完全搞反了
            [0.80, 0.10, 0.10],
            [0.10, 0.80, 0.10]
        ]),
        'color': '#e74c3c',
        'icon': '🔮',
        'desc': 'Systematically wrong model',
        'short': 'Wrong'
    }
}

# ============================================================
# 4. 计算结果
# ============================================================

print(f"\n{'='*65}")
print(f"{'Observer':<22s} | {'I_ORI':>10s} | {'A_M':>10s} | {'I_Shannon':>10s} | Status")
print("-" * 72)

names = []
I_CET_vals = []
A_M_vals = []
colors = []

for name, obs in observers.items():
    i_cet = compute_I_CET(T, obs['P_M'], pi)
    a_m = compute_A_M(I_shannon, i_cet)
    
    names.append(name)
    I_CET_vals.append(i_cet)
    A_M_vals.append(a_m)
    colors.append(obs['color'])
    
    clean_name = name.replace('\n', ' ')
    status = "Advantage" if a_m > 1e-8 else ("Baseline" if abs(a_m) < 1e-8 else "DISADVANTAGE")
    print(f"{clean_name:<22s} | {i_cet:10.6f} | {a_m:+10.6f} | {I_shannon:10.6f} | {status}")

print(f"\n  I_Shannon is IDENTICAL for all observers: {I_shannon:.6f} bits")
print(f"  I_ORI varies from {min(I_CET_vals):.6f} to {max(I_CET_vals):.6f} bits")
print(f"  Ratio (max/min I_ORI): {max(I_CET_vals)/max(min(I_CET_vals), 1e-10):.1f}×")

# ============================================================
# 5. 逐状态分析（展示 observer-relativity 的细节）
# ============================================================

print(f"\n{'='*65}")
print("逐状态 D_KL 分析：同一状态，不同观察者看到不同的信息落差")
print(f"{'='*65}")

for s in range(N):
    print(f"\n  当前状态 = {state_names[s]} (π={pi[s]:.4f})")
    print(f"  P_true({state_names[s]}→·) = {T[s]}")
    for name, obs in observers.items():
        clean_name = name.replace('\n', ' ')
        dkl = kl_divergence(T[s], obs['P_M'][s])
        print(f"    {clean_name:<22s}: P_M = {obs['P_M'][s]}  D_KL = {dkl:.6f}")

# ============================================================
# 6. 画图
# ============================================================

fig = plt.figure(figsize=(12, 8.5))

# --- 主图：I_ORI vs I_Shannon 对比（上方全宽）---
ax1 = fig.add_subplot(2, 1, 1)

x = np.arange(len(names))
width = 0.35

bars_cet = ax1.bar(x - width/2, I_CET_vals, width, color=colors, 
                    edgecolor='black', linewidth=1, label=r'$I_{ORI}$ (observer-relative)', alpha=0.9)
bars_sh = ax1.bar(x + width/2, [I_shannon]*len(names), width, color='lightgray',
                   edgecolor='black', linewidth=1, label=r'$I_{Shannon}$ (fixed)', hatch='///', alpha=0.7)

# 数值标注
for i, (cet, sh) in enumerate(zip(I_CET_vals, [I_shannon]*len(names))):
    ax1.text(i - width/2, cet + 0.01, f'{cet:.4f}', ha='center', va='bottom', 
             fontsize=10, fontweight='bold', color=colors[i])
    if i == 0:  # 只标一次 Shannon
        ax1.text(i + width/2, sh + 0.01, f'{sh:.4f}', ha='center', va='bottom',
                 fontsize=9, color='gray')

ax1.set_xticks(x)
ax1.set_xticklabels(names, fontsize=10)
ax1.set_ylabel('Information (bits)', fontsize=13)
ax1.set_title('Same System, Different Observers → Different Information', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, axis='y', alpha=0.3)

# 核心信息框
textstr = (f'$I_{{Shannon}}$ = {I_shannon:.4f} bits (same for all)\n'
           f'$I_{{ORI}}$ ranges from {min(I_CET_vals):.4f} to {max(I_CET_vals):.4f}')
props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8, edgecolor='orange')
ax1.text(0.98, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right', bbox=props)

# --- A_M 图（下方全宽）---
ax2 = fig.add_subplot(2, 1, 2)

bar_colors_am = ['#2ecc71' if v >= 0 else '#e74c3c' for v in A_M_vals]
bars_am = ax2.bar(x, A_M_vals, 0.5, color=bar_colors_am, edgecolor='black', linewidth=1)
ax2.axhline(y=0, color='black', linewidth=1.5)
ax2.axhline(y=I_shannon, color='blue', linestyle='--', linewidth=1, alpha=0.5,
            label=r'$I_{Shannon}$ (max $A_M$)')

for i, v in enumerate(A_M_vals):
    offset = 0.005 if v >= 0 else -0.01
    va = 'bottom' if v >= 0 else 'top'
    ax2.text(i, v + offset, f'{v:+.4f}', ha='center', va=va, fontsize=10, fontweight='bold')

ax2.set_xticks(x)
ax2.set_xticklabels([n.replace('\n', ' ') for n in names], fontsize=9)
ax2.set_ylabel(r'$A_M$ (Observer Advantage)', fontsize=12)
ax2.set_title('Observer Advantage', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, axis='y', alpha=0.3)

plt.suptitle('Experiment 4: Observer-Dependent Information', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('exp4_observer_dependent.png', dpi=150, bbox_inches='tight')
plt.savefig('exp4_observer_dependent.pdf', bbox_inches='tight')
print(f"\n图已保存: exp4_observer_dependent.png / .pdf")
plt.show()

# ============================================================
# 7. 论文核心段落（草稿）
# ============================================================

print("\n" + "=" * 65)
print("论文段落草稿")
print("=" * 65)
print(f"""
We demonstrate observer-relativity of information using a 3-state 
Markov chain with fixed transition matrix T. Four observers, each 
carrying a different internal model P_M, measure different values 
of observer-relative information I_ORI on the same system:

  Expert observer:    I_ORI = {I_CET_vals[0]:.4f} bits  (A_M = {A_M_vals[0]:+.4f})
  Partial observer:   I_ORI = {I_CET_vals[1]:.4f} bits  (A_M = {A_M_vals[1]:+.4f})
  Ignorant observer:  I_ORI = {I_CET_vals[2]:.4f} bits  (A_M = {A_M_vals[2]:+.4f})
  Wrong observer:     I_ORI = {I_CET_vals[3]:.4f} bits  (A_M = {A_M_vals[3]:+.4f})

Shannon mutual information I(X;Y) = {I_shannon:.4f} bits remains
identical across all four observers — it is a property of the system.
I_ORI varies by a factor of {max(I_CET_vals)/max(min(I_CET_vals),1e-10):.0f}x — it is a property
of the system-observer relationship.

The ignorant observer (P_M = marginal distribution) recovers exactly
I_ORI = I_Shannon = {I_shannon:.4f} bits, verifying the degeneracy
theorem. The wrong observer's I_ORI exceeds I_Shannon, yielding
negative Observer Advantage (A_M = {A_M_vals[3]:+.4f}), a phenomenon 
that Shannon's framework cannot express.
""")