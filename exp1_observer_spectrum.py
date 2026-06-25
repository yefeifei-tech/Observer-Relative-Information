"""
Experiment 1: Observer Spectrum
================================
验证 I_ORI 随观察者模型质量连续变化，而 I_Shannon 保持不变。

系统：2 状态天气模型 (晴/雨)
T = [[0.8, 0.2],
     [0.4, 0.6]]

观察者：通过插值参数 λ 构造
P_M^(λ) = λ · P_true + (1-λ) · P_marginal
λ = 1.0  → 完美观察者
λ = 0.0  → 转移无知 (Shannon 基准线)
λ < 0    → 比无知更差 (负知识区域)
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. 定义系统
# ============================================================

# 转移矩阵 T[i,j] = P(S_{t+1}=j | S_t=i)
T = np.array([
    [0.8, 0.2],   # 晴 → 晴 0.8, 晴 → 雨 0.2
    [0.4, 0.6]    # 雨 → 晴 0.4, 雨 → 雨 0.6
])

state_names = ['晴', '雨']
N = len(state_names)

# ============================================================
# 2. 计算稳态分布 π
# ============================================================

# π·T = π, 即 π 是 T^T 的特征值 1 对应的特征向量
eigenvalues, eigenvectors = np.linalg.eig(T.T)
# 找特征值最接近 1 的
idx = np.argmin(np.abs(eigenvalues - 1.0))
pi = np.real(eigenvectors[:, idx])
pi = pi / pi.sum()  # 归一化

print("=" * 60)
print("系统参数")
print("=" * 60)
print(f"转移矩阵 T:")
for i in range(N):
    row = ", ".join([f"P({state_names[i]}→{state_names[j]})={T[i,j]:.1f}" for j in range(N)])
    print(f"  {row}")
print(f"稳态分布 π: {dict(zip(state_names, [f'{p:.4f}' for p in pi]))}")

# ============================================================
# 3. 计算边缘分布 P(S_{t+1})
# ============================================================

P_marginal = pi @ T  # P(S_{t+1}) = Σ_i π(i) · T[i,:]
print(f"边缘分布 P(S_{{t+1}}): {dict(zip(state_names, [f'{p:.4f}' for p in P_marginal]))}")

# ============================================================
# 4. 核心计算函数
# ============================================================

def kl_divergence(p, q):
    """D_KL(p || q)，处理 p=0 的情况"""
    result = 0.0
    for pi_val, qi_val in zip(p, q):
        if pi_val > 1e-15:  # p=0 时贡献为 0
            if qi_val < 1e-15:  # q→0 而 p>0 时 D_KL → ∞
                return float('inf')
            result += pi_val * np.log2(pi_val / qi_val)
    return result


def compute_I_CET(T_true, P_M, pi):
    """
    I_CET = E_{S_t}[D_KL(P_true(·|s) || P_M(·|s))]
    = Σ_s π(s) · D_KL(P_true(·|s) || P_M(·|s))
    """
    I = 0.0
    for s in range(len(pi)):
        dkl = kl_divergence(T_true[s], P_M[s])
        if dkl == float('inf'):
            return float('inf')
        I += pi[s] * dkl
    return I


def compute_I_Shannon(T_true, pi, P_marginal):
    """
    I_Shannon = I(S_t; S_{t+1})
    = E_{S_t}[D_KL(P_true(·|s) || P(S_{t+1}))]
    
    即 I_CET 在 P_M = P_marginal 时的值
    """
    P_M_ignorant = np.tile(P_marginal, (len(pi), 1))  # 每行都是边缘分布
    return compute_I_CET(T_true, P_M_ignorant, pi)

# ============================================================
# 5. 计算 I_Shannon (固定值)
# ============================================================

I_shannon = compute_I_Shannon(T, pi, P_marginal)
print(f"\nI_Shannon = {I_shannon:.6f} bits")

# 验证：用标准公式 I = H(S_{t+1}) - H(S_{t+1}|S_t)
H_marginal = -np.sum(P_marginal * np.log2(P_marginal + 1e-15))
H_conditional = np.sum(pi[s] * (-np.sum(T[s] * np.log2(T[s] + 1e-15))) for s in range(N))
I_shannon_verify = H_marginal - H_conditional
print(f"I_Shannon (验证) = H(S_{{t+1}}) - H(S_{{t+1}}|S_t) = {H_marginal:.4f} - {H_conditional:.4f} = {I_shannon_verify:.6f} bits")

# ============================================================
# 6. Observer Spectrum：α 从 -0.5 到 1.0
# ============================================================

alphas = np.linspace(-0.5, 1.0, 301)
I_CET_values = []
I_acquired_values = []

for alpha in alphas:
    # P_M^(α) = α · P_true + (1-α) · P_marginal
    P_M = np.zeros_like(T)
    for s in range(N):
        P_M[s] = alpha * T[s] + (1 - alpha) * P_marginal
        # 确保概率非负（α < 0 时可能出现）
        P_M[s] = np.clip(P_M[s], 1e-10, None)
        P_M[s] = P_M[s] / P_M[s].sum()  # 重新归一化
    
    i_cet = compute_I_CET(T, P_M, pi)
    i_acq = I_shannon - i_cet
    
    I_CET_values.append(i_cet)
    I_acquired_values.append(i_acq)

I_CET_values = np.array(I_CET_values)
I_acquired_values = np.array(I_acquired_values)

# ============================================================
# 7. 打印关键点
# ============================================================

print("\n" + "=" * 60)
print("关键观察者的 I_ORI 值")
print("=" * 60)

key_alphas = [1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5]
for a in key_alphas:
    idx = np.argmin(np.abs(alphas - a))
    label = {
        1.0: "完美观察者",
        0.75: "75% 学习",
        0.5: "50% 学习", 
        0.25: "25% 学习",
        0.0: "转移无知 (Shannon 基准)",
        -0.25: "负知识 (轻度)",
        -0.5: "负知识 (重度)"
    }[a]
    marker = " ← I_ORI = I_Shannon" if abs(a) < 0.01 else ""
    marker = " ← I_ORI = 0" if abs(a - 1.0) < 0.01 else marker
    marker = " ← I_ORI > I_Shannon (负知识!)" if a < 0 else marker
    print(f"  λ={a:+.2f}  {label:20s}  I_ORI={I_CET_values[idx]:.6f}  A_M={I_acquired_values[idx]:+.6f}{marker}")

print(f"\n  I_Shannon (固定) = {I_shannon:.6f} bits")

# ============================================================
# 8. 画图
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- 图 1: I_ORI vs λ ---
ax1 = axes[0]
ax1.plot(alphas, I_CET_values, 'b-', linewidth=2.5, label=r'$I_{ORI}(\lambda)$')
ax1.axhline(y=I_shannon, color='r', linestyle='--', linewidth=1.5, 
            label=r'$I_{Shannon}$ (fixed)')
ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax1.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

# 标注关键点
ax1.plot(1.0, 0, 'go', markersize=10, zorder=5, label='Perfect observer')
ax1.plot(0.0, I_shannon, 'rs', markersize=10, zorder=5, label='Transition ignorance')

# 负知识区域着色
ax1.fill_between(alphas, I_shannon, I_CET_values, 
                  where=(I_CET_values > I_shannon),
                  alpha=0.15, color='red', label='Negative knowledge region')

ax1.set_xlabel(r'Observer quality $\lambda$', fontsize=13)
ax1.set_ylabel('Information (bits)', fontsize=13)
ax1.set_title('Experiment 1: Observer Spectrum', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.set_xlim(-0.55, 1.05)
ax1.grid(True, alpha=0.3)

# 添加注释
ax1.annotate(r'$\lambda=0$: $I_{ORI} = I_{Shannon}$',
             xy=(0, I_shannon), xytext=(0.15, I_shannon * 1.5),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=10, ha='left')
ax1.annotate(r'$\lambda=1$: $I_{ORI} = 0$',
             xy=(1.0, 0), xytext=(0.65, I_shannon * 0.3),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=10, ha='left')

# --- 图 2: A_M vs λ ---
ax2 = axes[1]
ax2.plot(alphas, I_acquired_values, 'g-', linewidth=2.5, label=r'$A_M(\lambda)$')
ax2.axhline(y=I_shannon, color='r', linestyle='--', linewidth=1.5,
            label=r'$I_{Shannon}$ (maximum)')
ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
ax2.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

# 负知识区域着色
ax2.fill_between(alphas, 0, I_acquired_values,
                  where=(I_acquired_values < 0),
                  alpha=0.15, color='red', label='Negative knowledge')

ax2.set_xlabel(r'Observer quality $\lambda$', fontsize=13)
ax2.set_ylabel('Information (bits)', fontsize=13)
ax2.set_title(r'$A_M = I_{Shannon} - I_{ORI}$', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right', fontsize=10)
ax2.set_xlim(-0.55, 1.05)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exp1_observer_spectrum.png', dpi=150, bbox_inches='tight')
plt.savefig('exp1_observer_spectrum.pdf', bbox_inches='tight')
print(f"\n图已保存: exp1_observer_spectrum.png / .pdf")
plt.show()

# ============================================================
# 9. 核心验证：λ=0 时 I_ORI 是否精确等于 I_Shannon
# ============================================================

print("\n" + "=" * 60)
print("核心验证：退化定理")
print("=" * 60)
idx_zero = np.argmin(np.abs(alphas - 0.0))
print(f"  I_ORI(λ=0)  = {I_CET_values[idx_zero]:.10f}")
print(f"  I_Shannon   = {I_shannon:.10f}")
print(f"  差值         = {abs(I_CET_values[idx_zero] - I_shannon):.2e}")
print(f"  退化定理验证: {'✓ PASS' if abs(I_CET_values[idx_zero] - I_shannon) < 1e-8 else '✗ FAIL'}")

# ============================================================
# 10. 论文可用的数据表
# ============================================================

print("\n" + "=" * 60)
print("论文表格数据")
print("=" * 60)
print(f"{'λ':>6s} | {'I_ORI':>10s} | {'A_M':>12s} | {'I_Shannon':>10s} | {'I_ORI > I_Sh?':>14s}")
print("-" * 62)
for a in [1.0, 0.75, 0.5, 0.25, 0.0, -0.1, -0.25, -0.5]:
    idx = np.argmin(np.abs(alphas - a))
    flag = "YES (负知识)" if I_CET_values[idx] > I_shannon + 1e-8 else "NO"
    print(f"{a:+6.2f} | {I_CET_values[idx]:10.6f} | {I_acquired_values[idx]:+12.6f} | {I_shannon:10.6f} | {flag:>14s}")