"""
Experiment 5: Bayesian Learning Dynamics (Recovery from Negative Knowledge)
============================================================================
展示观察者从错误信念出发，通过贝叶斯更新逐步恢复的过程。

核心发现：
  - 初始错误越深(A_M(0) 越负)，需要越多数据才能恢复
  - 先验越固执(α 越大)，恢复越慢
  - 存在临界数据量 n*，使得 A_M(n*) = 0

系统：2 状态天气模型（和 Experiment 1-2 相同）
方法：Dirichlet-Multinomial 共轭，全部用期望值精确计算
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. 系统定义（和 Experiment 1-2 相同）
# ============================================================

T = np.array([
    [0.8, 0.2],
    [0.4, 0.6]
])
N = 2

# 稳态分布
eigenvalues, eigenvectors = np.linalg.eig(T.T)
idx = np.argmin(np.abs(eigenvalues - 1.0))
pi = np.real(eigenvectors[:, idx])
pi = pi / pi.sum()

# 边缘分布
P_marginal = pi @ T

# I_Shannon
def kl_divergence(p, q):
    result = 0.0
    for pi_val, qi_val in zip(p, q):
        if pi_val > 1e-15:
            if qi_val < 1e-15:
                return float('inf')
            result += pi_val * np.log2(pi_val / qi_val)
    return result

def compute_A_M(T_true, P_M, pi, P_marginal):
    """A_M = Σ π(i) Σ T_ij log2(P_M(j|i) / P_marginal(j))"""
    a_m = 0.0
    for i in range(len(pi)):
        for j in range(len(pi)):
            if T_true[i, j] > 1e-15:
                if P_M[i, j] < 1e-15:
                    return float('-inf')
                a_m += pi[i] * T_true[i, j] * np.log2(P_M[i, j] / P_marginal[j])
    return a_m

def compute_I_ORI(T_true, P_M, pi):
    """I_ORI = E[D_KL(P_true || P_M)]"""
    I = 0.0
    for s in range(len(pi)):
        dkl = kl_divergence(T_true[s], P_M[s])
        if dkl == float('inf'):
            return float('inf')
        I += pi[s] * dkl
    return I

P_M_ignorant = np.tile(P_marginal, (N, 1))
I_shannon = compute_I_ORI(T, P_M_ignorant, pi)

print("=" * 65)
print("Experiment 5: Bayesian Learning Dynamics")
print("=" * 65)
print(f"System: T = [[0.8, 0.2], [0.4, 0.6]]")
print(f"π = ({pi[0]:.4f}, {pi[1]:.4f})")
print(f"P_marginal = ({P_marginal[0]:.4f}, {P_marginal[1]:.4f})")
print(f"I_Shannon = {I_shannon:.6f} bits")

# ============================================================
# 2. 贝叶斯更新函数
# ============================================================

def bayesian_update(T_true, pi, P_M_0, alpha, n):
    """
    贝叶斯更新（用期望观测次数，精确计算）
    
    P_M^(n)(j|i) = (E[n_ij] + α · P_M^(0)(j|i)) / (Σ_k E[n_ik] + α)
    E[n_ij] = n · π(i) · T_ij
    """
    P_M_n = np.zeros_like(T_true)
    for i in range(T_true.shape[0]):
        for j in range(T_true.shape[1]):
            E_nij = n * pi[i] * T_true[i, j]
            numerator = E_nij + alpha * P_M_0[i, j]
            denominator = sum(n * pi[i] * T_true[i, k] + alpha * P_M_0[i, k] 
                            for k in range(T_true.shape[1]))
            P_M_n[i, j] = numerator / denominator
    return P_M_n

def find_n_star(T_true, pi, P_marginal, P_M_0, alpha, n_max=1000):
    """找到 A_M 首次穿过零点的 n*"""
    for n in range(n_max + 1):
        P_M_n = bayesian_update(T_true, pi, P_M_0, alpha, n)
        a_m = compute_A_M(T_true, P_M_n, pi, P_marginal)
        if a_m >= 0:
            return n
    return None  # 在 n_max 内未穿过

# ============================================================
# 3. 左图：不同初始错误深度，固定 α=10
# ============================================================

alpha_fixed = 10
n_range = np.arange(0, 301)

initial_models = {
    'Shallow\n(uniform)': {
        'P_M_0': np.array([[0.5, 0.5], [0.5, 0.5]]),
        'color': '#3498db',
        'linestyle': '-',
        'desc': 'Absolute ignorance'
    },
    'Medium\n(slightly reversed)': {
        'P_M_0': np.array([[0.4, 0.6], [0.6, 0.4]]),
        'color': '#f39c12',
        'linestyle': '-',
        'desc': 'Slightly reversed'
    },
    'Deep\n(fully reversed)': {
        'P_M_0': np.array([[0.2, 0.8], [0.8, 0.2]]),
        'color': '#e74c3c',
        'linestyle': '-',
        'desc': 'Fully reversed'
    }
}

print(f"\n--- Left panel: varying initial error depth, α = {alpha_fixed} ---")

left_data = {}
for name, config in initial_models.items():
    A_M_curve = []
    for n in n_range:
        P_M_n = bayesian_update(T, pi, config['P_M_0'], alpha_fixed, n)
        a_m = compute_A_M(T, P_M_n, pi, P_marginal)
        A_M_curve.append(a_m)
    
    n_star = find_n_star(T, pi, P_marginal, config['P_M_0'], alpha_fixed)
    left_data[name] = {
        'curve': np.array(A_M_curve),
        'n_star': n_star,
        'A_M_0': A_M_curve[0],
        'config': config
    }
    
    clean_name = name.replace('\n', ' ')
    print(f"  {clean_name:<30s}: A_M(0) = {A_M_curve[0]:+.4f}, n* = {n_star}")

# ============================================================
# 4. 右图：不同先验强度，固定初始模型为"完全学反"
# ============================================================

P_M_0_fixed = np.array([[0.2, 0.8], [0.8, 0.2]])

prior_strengths = {
    'α = 5\n(easily persuaded)': {
        'alpha': 5,
        'color': '#2ecc71',
        'linestyle': '-'
    },
    'α = 20\n(moderate)': {
        'alpha': 20,
        'color': '#f39c12',
        'linestyle': '-'
    },
    'α = 50\n(very stubborn)': {
        'alpha': 50,
        'color': '#e74c3c',
        'linestyle': '-'
    }
}

print(f"\n--- Right panel: varying prior strength, P_M(0) = fully reversed ---")

right_data = {}
n_range_right = np.arange(0, 301)

for name, config in prior_strengths.items():
    A_M_curve = []
    for n in n_range_right:
        P_M_n = bayesian_update(T, pi, P_M_0_fixed, config['alpha'], n)
        a_m = compute_A_M(T, P_M_n, pi, P_marginal)
        A_M_curve.append(a_m)
    
    n_star = find_n_star(T, pi, P_marginal, P_M_0_fixed, config['alpha'], n_max=500)
    right_data[name] = {
        'curve': np.array(A_M_curve),
        'n_star': n_star,
        'config': config
    }
    
    clean_name = name.replace('\n', ' ')
    n_star_str = str(n_star) if n_star is not None else ">500"
    print(f"  {clean_name:<30s}: α = {config['alpha']}, n* = {n_star_str}")

# ============================================================
# 5. 画图
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- 左图：不同初始错误深度 ---
ax1 = axes[0]

for k, (name, data) in enumerate(left_data.items()):
    config = data['config']
    ax1.plot(n_range, data['curve'], color=config['color'], linestyle=config['linestyle'],
             linewidth=2.5, label=name.replace('\n', ' '))

    # 标注 n*（三个标记挨得近：用长箭头把文字放到右下空白区，彼此错开，避免压曲线）
    if data['n_star'] is not None and data['n_star'] <= 300:
        ax1.plot(data['n_star'], 0, 'o', color=config['color'], markersize=10,
                 markeredgecolor='black', markeredgewidth=1.5, zorder=5)
        ax1.annotate(f"n*={data['n_star']}",
                     xy=(data['n_star'], 0),
                     xytext=(55 + 38 * k, -0.30 - 0.16 * k),
                     fontsize=10, color=config['color'], fontweight='bold',
                     ha='left', va='center',
                     arrowprops=dict(arrowstyle='->', color=config['color'],
                                     lw=1.5, shrinkA=3, shrinkB=4))

# 参考线
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.axhline(y=I_shannon, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax1.text(290, I_shannon + 0.005, f'$I_{{Shannon}}$ = {I_shannon:.4f}', 
         fontsize=9, ha='right', va='bottom', color='gray')
ax1.text(290, 0.005, 'zero point', fontsize=9, ha='right', va='bottom', color='black', alpha=0.7)

# 负知识区域
ax1.fill_between(n_range, min(left_data['Deep\n(fully reversed)']['curve']) * 1.1, 0, 
                  alpha=0.05, color='red')
ax1.text(5, min(left_data['Deep\n(fully reversed)']['curve']) * 0.5, 
         'Negative\nKnowledge', fontsize=10, color='red', alpha=0.5, fontweight='bold')

ax1.set_xlabel('Number of observations $n$', fontsize=12)
ax1.set_ylabel('$A_M$ (Observer Advantage, bits)', fontsize=12)
ax1.set_title('Deeper initial error → more data needed', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='lower right')
ax1.set_xlim(-5, 305)
ax1.grid(True, alpha=0.3)

# --- 右图：不同先验强度 ---
ax2 = axes[1]

for name, data in right_data.items():
    config = data['config']
    ax2.plot(n_range_right, data['curve'], color=config['color'], linestyle=config['linestyle'],
             linewidth=2.5, label=name.replace('\n', ' '))
    
    # 标注 n*
    if data['n_star'] is not None and data['n_star'] <= 300:
        ax2.plot(data['n_star'], 0, 'o', color=config['color'], markersize=10,
                 markeredgecolor='black', markeredgewidth=1.5, zorder=5)
        ax2.annotate(f"n*={data['n_star']}", 
                     xy=(data['n_star'], 0),
                     xytext=(data['n_star'] + 15, -0.03),
                     fontsize=9, color=config['color'], fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=config['color'], lw=1.5))

# 参考线
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.axhline(y=I_shannon, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax2.text(290, I_shannon + 0.005, f'$I_{{Shannon}}$ = {I_shannon:.4f}',
         fontsize=9, ha='right', va='bottom', color='gray')
ax2.text(290, 0.005, 'zero point', fontsize=9, ha='right', va='bottom', color='black', alpha=0.7)

# 负知识区域
ax2.fill_between(n_range_right, min(right_data['α = 50\n(very stubborn)']['curve']) * 1.1, 0,
                  alpha=0.05, color='red')

ax2.set_xlabel('Number of observations $n$', fontsize=12)
ax2.set_ylabel('$A_M$ (Observer Advantage, bits)', fontsize=12)
ax2.set_title('Stronger prior → slower recovery', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9, loc='lower right')
ax2.set_xlim(-5, 305)
ax2.grid(True, alpha=0.3)

plt.suptitle('Experiment 5: Recovery from Negative Knowledge', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('exp5_bayesian_recovery.png', dpi=150, bbox_inches='tight')
plt.savefig('exp5_bayesian_recovery.pdf', bbox_inches='tight')
print(f"\n图已保存: exp5_bayesian_recovery.png / .pdf")
plt.show()

# ============================================================
# 6. 核心结论
# ============================================================

print("\n" + "=" * 65)
print("核心结论")
print("=" * 65)

# 提取数据(避免 f-string 中的反斜杠)
shallow = left_data['Shallow\n(uniform)']
medium = left_data['Medium\n(slightly reversed)']
deep = left_data['Deep\n(fully reversed)']
alpha5 = right_data['α = 5\n(easily persuaded)']
alpha20 = right_data['α = 20\n(moderate)']
alpha50 = right_data['α = 50\n(very stubborn)']

print(f"""
左图（不同初始错误深度，α = {alpha_fixed}）：
  浅坑（均匀先验）：A_M(0) = {shallow['A_M_0']:+.4f}, n* = {shallow['n_star']}
  中坑（略反）：    A_M(0) = {medium['A_M_0']:+.4f}, n* = {medium['n_star']}
  深坑（完全反）：  A_M(0) = {deep['A_M_0']:+.4f}, n* = {deep['n_star']}

  → 初始错误越深，需要越多数据才能爬回零点。

右图（不同先验强度，初始模型 = 完全学反）：
  α = 5  (容易说服)：n* = {alpha5['n_star']}
  α = 20 (中等固执)：n* = {alpha20['n_star']}
  α = 50 (非常固执)：n* = {alpha50['n_star']}

  → 先验越强(越固执)，恢复越慢。

物理含义：
  负知识不仅是一个静态度量，还有动态后果：
  从负知识中恢复需要数据（evidence）。
  恢复速度取决于两个因素：
    1. 初始错误的深度（坑有多深）
    2. 先验的强度（墙有多厚）
  n* 是"从错误信念中恢复所需的最少证据量"。
""")