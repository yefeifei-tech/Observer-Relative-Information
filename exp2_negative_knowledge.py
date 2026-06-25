"""
Experiment 2: Negative Knowledge (Observer Disadvantage)
=========================================================
直接展示 I_ORI > I_Shannon 的现象——Shannon 框架无法表达的状态。

核心发现：当观察者的模型比"完全无知"更差时，
Observer Advantage A_M < 0，即观察者处于劣势。

系统：2 状态天气模型 (晴/雨)
观察者：从无知到极端错误的多个模型
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. 定义系统（和实验 1 相同）
# ============================================================

T = np.array([
    [0.8, 0.2],   # 晴 → 晴 0.8, 晴 → 雨 0.2
    [0.4, 0.6]    # 雨 → 晴 0.4, 雨 → 雨 0.6
])
state_names = ['Sunny', 'Rainy']
N = 2

# 稳态分布
eigenvalues, eigenvectors = np.linalg.eig(T.T)
idx = np.argmin(np.abs(eigenvalues - 1.0))
pi = np.real(eigenvectors[:, idx])
pi = pi / pi.sum()

# 边缘分布
P_marginal = pi @ T

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

# I_Shannon
P_M_ignorant = np.tile(P_marginal, (N, 1))
I_shannon = compute_I_CET(T, P_M_ignorant, pi)

# ============================================================
# 3. 定义五种观察者
# ============================================================

observers = {
    'Perfect\n(A)': {
        'P_M': T.copy(),
        'color': '#2ecc71',
        'desc': 'P_M = P_true'
    },
    'Partial\nLearner (B)': {
        'P_M': np.array([
            [0.7, 0.3],   # 知道晴→晴偏多，但不精确
            [0.35, 0.65]
        ]),
        'color': '#3498db',
        'desc': '90% correct direction'
    },
    'Transition\nIgnorant (C)': {
        'P_M': np.tile(P_marginal, (N, 1)),
        'color': '#f39c12',
        'desc': 'P_M = marginal'
    },
    'Reversed\n(D)': {
        'P_M': np.array([
            [0.2, 0.8],   # 把晴→晴搞反了
            [0.6, 0.4]    # 把雨→雨搞反了
        ]),
        'color': '#e74c3c',
        'desc': 'Reversed transitions'
    },
    'Extreme\nWrong (E)': {
        'P_M': np.array([
            [0.05, 0.95],  # 极度错误
            [0.95, 0.05]
        ]),
        'color': '#8e44ad',
        'desc': 'Extreme reversal'
    }
}

# ============================================================
# 4. 计算每个观察者的 I_CET 和 A_M
# ============================================================

print("=" * 70)
print("Experiment 2: Negative Knowledge (Observer Disadvantage)")
print("=" * 70)
print(f"\nI_Shannon = {I_shannon:.6f} bits (fixed baseline)\n")
print(f"{'Observer':<20s} | {'I_ORI':>10s} | {'A_M':>10s} | {'A_M < 0?':>10s} | Description")
print("-" * 85)

names = []
I_CET_vals = []
A_M_vals = []
colors = []

for name, obs in observers.items():
    i_cet = compute_I_CET(T, obs['P_M'], pi)
    a_m = I_shannon - i_cet
    
    names.append(name)
    I_CET_vals.append(i_cet)
    A_M_vals.append(a_m)
    colors.append(obs['color'])
    
    flag = "YES ←" if a_m < -1e-8 else "no"
    clean_name = name.replace('\n', ' ')
    print(f"{clean_name:<20s} | {i_cet:10.6f} | {a_m:+10.6f} | {flag:>10s} | {obs['desc']}")

print(f"\n{'I_Shannon':<20s} | {I_shannon:10.6f} | {'':>10s} | {'':>10s} | Baseline (transition-ignorant)")

# ============================================================
# 5. 画图
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- 图 1: I_CET 柱状图 ---
ax1 = axes[0]
x = np.arange(len(names))
bars1 = ax1.bar(x, I_CET_vals, color=colors, width=0.6, edgecolor='black', linewidth=0.8)
ax1.axhline(y=I_shannon, color='red', linestyle='--', linewidth=2, 
            label=r'$I_{Shannon}$ (ignorance baseline)')

# 在柱子上标数值
for i, (val, am) in enumerate(zip(I_CET_vals, A_M_vals)):
    ax1.text(i, val + 0.01, f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(names, fontsize=9)
ax1.set_ylabel(r'$I_{ORI}$ (bits)', fontsize=13)
ax1.set_title(r'$I_{ORI}$: Observer-Relative Information', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, axis='y', alpha=0.3)

# 标注负知识区域
ax1.annotate('Negative Knowledge\nRegion', xy=(3, I_CET_vals[3]), 
             xytext=(2.5, max(I_CET_vals) * 0.85),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=10, color='red', fontweight='bold', ha='center')

# --- 图 2: A_M 柱状图 ---
ax2 = axes[1]
bar_colors_am = ['#2ecc71' if v >= 0 else '#e74c3c' for v in A_M_vals]
bars2 = ax2.bar(x, A_M_vals, color=bar_colors_am, width=0.6, edgecolor='black', linewidth=0.8)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.axhline(y=I_shannon, color='blue', linestyle='--', linewidth=1.5, alpha=0.5,
            label=r'$I_{Shannon}$ (maximum $A_M$)')

# 在柱子上标数值（白色半透明底框，避免与虚线/区域标注重叠）
for i, val in enumerate(A_M_vals):
    offset = 0.012 if val >= 0 else -0.018
    va = 'bottom' if val >= 0 else 'top'
    ax2.text(i, val + offset, f'{val:+.4f}', ha='center', va=va, fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.75, edgecolor='none'))

ax2.set_xticks(x)
ax2.set_xticklabels(names, fontsize=9)
ax2.set_ylabel(r'$A_M$ (Observer Advantage, bits)', fontsize=13)
ax2.set_title(r'$A_M = I_{Shannon} - I_{ORI}$: Observer Advantage', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, axis='y', alpha=0.3)

# 标注区域（移到空白区，加深颜色 + 白底框，避免与柱子/数值标签重叠）
ymin = min(A_M_vals) * 1.2
ymax = I_shannon * 1.3
ax2.fill_between([-0.5, len(names)-0.5], 0, ymin, alpha=0.05, color='red')
ax2.fill_between([-0.5, len(names)-0.5], 0, ymax, alpha=0.05, color='green')
ax2.text(1.5, ymin * 0.5, 'Disadvantage\n(Negative Knowledge)',
         ha='center', va='center', fontsize=10, color='#c0392b', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#c0392b', linewidth=0.8))
ax2.text(3.0, ymax * 0.55, 'Advantage',
         ha='center', va='center', fontsize=10, color='#27ae60', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#27ae60', linewidth=0.8))

plt.tight_layout()
plt.savefig('exp2_negative_knowledge.png', dpi=150, bbox_inches='tight')
plt.savefig('exp2_negative_knowledge.pdf', bbox_inches='tight')
print(f"\n图已保存: exp2_negative_knowledge.png / .pdf")
plt.show()

# ============================================================
# 6. 核心论点总结
# ============================================================

print("\n" + "=" * 70)
print("核心论点")
print("=" * 70)
print("""
在 Shannon 框架中：
  I(X;Y) 是系统的客观属性，与观察者无关。
  所有观察者面对同一个系统，看到同一个 I(X;Y)。

在 Observer-Relative Information 框架中：
  I_ORI 是系统与观察者之间的关系属性。
  同一个系统，不同观察者测出不同的 I_ORI：
""")

for name, i_cet, a_m in zip(names, I_CET_vals, A_M_vals):
    clean_name = name.replace('\n', ' ')
    status = "advantage" if a_m > 0 else ("baseline" if abs(a_m) < 1e-8 else "DISADVANTAGE")
    print(f"  {clean_name:<20s}: I_ORI = {i_cet:.4f} bits, A_M = {a_m:+.4f} ({status})")

print(f"\n  I_Shannon (fixed)   : {I_shannon:.4f} bits (same for all observers)")
print(f"""
关键发现：
  Observer D 和 E 的 I_ORI > I_Shannon。
  这意味着它们的模型比"完全无知"更差。
  Shannon 互信息无法表达这种状态。
  Observer Advantage A_M < 0 = 负知识。
""")