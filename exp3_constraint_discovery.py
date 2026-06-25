"""
Experiment 3: Constraint Discovery via R_C
============================================
展示 R_C (条件熵) 能识别哪些变量携带真实约束结构。

系统：10 个二值变量
  - x1, x2, x3：互相耦合（有约束结构）
  - x4, ..., x10：独立噪声（无约束结构）

方法：逐步加入变量到条件集，观察 R_C 的下降曲线。
  约束变量加入时：R_C 急剧下降
  噪声变量加入时：R_C 不变

这证明 R_C 可以作为约束发现工具。
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# ============================================================
# 1. 定义约束变量的耦合转移
# ============================================================

# x1, x2, x3 各取 {0, 1}
# 转移规则（环形耦合）：
#   x1' 依赖 (x1, x2): 相同时倾向输出 1，不同时倾向输出 0
#   x2' 依赖 (x2, x3): 同理
#   x3' 依赖 (x3, x1): 同理

def p_next(x_self, x_neighbor, bias_same=0.85, bias_diff=0.2):
    """P(x'=1 | x_self, x_neighbor)"""
    if x_self == x_neighbor:
        return bias_same
    else:
        return bias_diff

# 构建 3 变量联合转移表
# 状态 = (x1, x2, x3)，共 8 个状态
constraint_states = list(product([0, 1], repeat=3))  # 所有 (x1,x2,x3) 组合

# T_constraint[s][s'] = P(x1',x2',x3' | x1,x2,x3)
# 由于 x1', x2', x3' 条件独立（给定各自的依赖），联合概率 = 乘积
n_constraint = len(constraint_states)
T_constraint = np.zeros((n_constraint, n_constraint))

for i, (x1, x2, x3) in enumerate(constraint_states):
    for j, (x1p, x2p, x3p) in enumerate(constraint_states):
        # P(x1'|x1,x2) · P(x2'|x2,x3) · P(x3'|x3,x1)
        p1 = p_next(x1, x2)  # P(x1'=1|x1,x2)
        p2 = p_next(x2, x3)  # P(x2'=1|x2,x3)
        p3 = p_next(x3, x1)  # P(x3'=1|x3,x1)
        
        prob = 1.0
        prob *= p1 if x1p == 1 else (1 - p1)
        prob *= p2 if x2p == 1 else (1 - p2)
        prob *= p3 if x3p == 1 else (1 - p3)
        
        T_constraint[i, j] = prob

# 验证行和
assert np.allclose(T_constraint.sum(axis=1), 1.0), "转移矩阵行和不为 1"

# 计算稳态分布
eigenvalues, eigenvectors = np.linalg.eig(T_constraint.T)
idx = np.argmin(np.abs(eigenvalues - 1.0))
pi_constraint = np.real(eigenvectors[:, idx])
pi_constraint = pi_constraint / pi_constraint.sum()

print("=" * 60)
print("Experiment 3: Constraint Discovery via R_C")
print("=" * 60)
print(f"\n约束变量 (x1, x2, x3): 环形耦合")
print(f"  x1' 依赖 (x1, x2), x2' 依赖 (x2, x3), x3' 依赖 (x3, x1)")
print(f"  同值时 P(x'=1) = 0.85, 异值时 P(x'=1) = 0.20")
print(f"\n噪声变量 (x4, ..., x10): 独立 Bernoulli(0.5), 无时间结构")
print(f"\n约束变量状态数: {n_constraint}")

# ============================================================
# 2. 条件熵计算函数
# ============================================================

def entropy(probs):
    """H(X) = -Σ p log2 p"""
    probs = probs[probs > 1e-15]
    return -np.sum(probs * np.log2(probs))

def conditional_entropy_constraint(T, pi):
    """H(X'|X) for the constraint variables"""
    h = 0.0
    for i in range(len(pi)):
        h += pi[i] * entropy(T[i])
    return h

# ============================================================
# 3. 计算基线值
# ============================================================

# 约束变量的边缘分布
P_marginal_constraint = pi_constraint @ T_constraint

# 约束变量的各种熵
H_constraint_marginal = entropy(P_marginal_constraint)
H_constraint_conditional = conditional_entropy_constraint(T_constraint, pi_constraint)
I_constraint = H_constraint_marginal - H_constraint_conditional

# 噪声变量：每个独立 Bernoulli(0.5), 无时间结构
# H(xk') = 1 bit, H(xk'|anything) = 1 bit
H_noise_per_var = 1.0  # bit
n_noise = 7

print(f"\n--- 基线值 ---")
print(f"H(constraint vars next)           = {H_constraint_marginal:.4f} bits")
print(f"H(constraint vars next | all 3)   = {H_constraint_conditional:.4f} bits")
print(f"I(constraint) = Shannon MI        = {I_constraint:.4f} bits")
print(f"H(each noise var)                 = {H_noise_per_var:.4f} bits (不可约)")
print(f"H(all 7 noise vars)               = {n_noise * H_noise_per_var:.4f} bits (不可约)")
print(f"H(full S_{{t+1}})                   = {H_constraint_marginal + n_noise:.4f} bits")

# ============================================================
# 4. 逐步加入变量，计算条件熵
# ============================================================

def compute_H_given_partial_constraint(T, pi, known_indices):
    """
    计算 H(x1',x2',x3' | 已知的约束变量子集)
    
    known_indices: 已知的约束变量索引 (0=x1, 1=x2, 2=x3)
    """
    if len(known_indices) == 0:
        # 不知道任何约束变量 → H(x1',x2',x3') = 边缘熵
        return H_constraint_marginal
    
    if len(known_indices) == 3:
        # 知道全部约束变量 → H(x1',x2',x3'|x1,x2,x3)
        return H_constraint_conditional
    
    # 部分知道：对已知变量的每个取值，计算条件下的 next state 分布
    # 然后对已知变量的边缘分布加权平均
    
    n_known_states = 2 ** len(known_indices)
    unknown_indices = [i for i in range(3) if i not in known_indices]
    
    H_total = 0.0
    
    # 遍历已知变量的所有可能取值
    for known_vals in product([0, 1], repeat=len(known_indices)):
        # 计算 P(已知变量 = known_vals)
        p_known = 0.0
        # P(next | 已知变量的特定值) 需要对未知变量取边缘
        p_next_given_known = np.zeros(n_constraint)
        
        for full_state_idx, full_state in enumerate(constraint_states):
            # 检查已知变量是否匹配
            match = all(full_state[k] == v for k, v in zip(known_indices, known_vals))
            if match:
                p_known += pi_constraint[full_state_idx]
                p_next_given_known += pi_constraint[full_state_idx] * T_constraint[full_state_idx]
        
        if p_known > 1e-15:
            p_next_given_known /= p_known  # 归一化
            H_total += p_known * entropy(p_next_given_known)
    
    return H_total

# ============================================================
# 5. 实验 A：按顺序加入（先约束后噪声）
# ============================================================

print(f"\n{'='*60}")
print("实验 A: 按顺序加入变量 (x1 → x2 → x3 → x4 → ... → x10)")
print(f"{'='*60}")

steps_ordered = []
labels_ordered = []
H_values_ordered = []

# 步骤 0: 无条件
H_full_marginal = H_constraint_marginal + n_noise * H_noise_per_var
steps_ordered.append(0)
labels_ordered.append('(none)')
H_values_ordered.append(H_full_marginal)
print(f"  Step 0: 条件集 = 空          H = {H_full_marginal:.4f} bits")

# 步骤 1-3: 加入约束变量
for k in range(1, 4):
    known = list(range(k))
    H_c = compute_H_given_partial_constraint(T_constraint, pi_constraint, known)
    H_total = H_c + n_noise * H_noise_per_var
    steps_ordered.append(k)
    var_name = f"x1..x{k}"
    labels_ordered.append(var_name)
    H_values_ordered.append(H_total)
    delta = H_values_ordered[-2] - H_total
    print(f"  Step {k}: 条件集 = {{{var_name}}}    H = {H_total:.4f} bits  (ΔH = -{delta:.4f}) ← 约束变量")

# 步骤 4-10: 加入噪声变量
for k in range(4, 11):
    # 噪声变量无时间结构, 加入后不改变对任何未来变量的预测
    H_c = H_constraint_conditional  # 约束部分已知全部
    # 噪声部分: 已加入的噪声变量能预测自己的未来吗？
    # 不能! 因为噪声是 i.i.d.，H(xk'|xk) = H(xk') = 1 bit
    H_total = H_c + n_noise * H_noise_per_var
    steps_ordered.append(k)
    var_name = f"x1..x{k}"
    labels_ordered.append(var_name)
    H_values_ordered.append(H_total)
    delta = H_values_ordered[-2] - H_total
    print(f"  Step {k}: 条件集 = {{{var_name}}}   H = {H_total:.4f} bits  (ΔH = -{delta:.4f}) ← 噪声变量")

# ============================================================
# 6. 实验 B：打乱顺序（先噪声后约束）
# ============================================================

print(f"\n{'='*60}")
print("实验 B: 打乱顺序 (x7 → x4 → x10 → x1 → x5 → x2 → x8 → x3 → x6 → x9)")
print(f"{'='*60}")

# 定义一个混合顺序
shuffled_order = [6, 3, 9, 0, 4, 1, 7, 2, 5, 8]  # 0-indexed: x7,x4,x10,x1,x5,x2,x8,x3,x6,x9
shuffled_names = [f'x{i+1}' for i in shuffled_order]

steps_shuffled = [0]
labels_shuffled = ['(none)']
H_values_shuffled = [H_full_marginal]

constraint_so_far = []  # 已加入的约束变量索引 (0,1,2)

print(f"  Step 0: 条件集 = 空          H = {H_full_marginal:.4f} bits")

for step, var_idx in enumerate(shuffled_order, 1):
    is_constraint = var_idx < 3
    if is_constraint:
        constraint_so_far.append(var_idx)
    
    H_c = compute_H_given_partial_constraint(T_constraint, pi_constraint, constraint_so_far)
    H_total = H_c + n_noise * H_noise_per_var
    
    steps_shuffled.append(step)
    labels_shuffled.append(f'+{shuffled_names[step-1]}')
    H_values_shuffled.append(H_total)
    
    delta = H_values_shuffled[-2] - H_total
    tag = "← 约束变量!" if is_constraint else "   噪声"
    print(f"  Step {step:2d}: +{shuffled_names[step-1]:>3s}  H = {H_total:.4f} bits  (ΔH = -{delta:.4f}) {tag}")

# ============================================================
# 7. 计算边际贡献 (每步的 ΔH)
# ============================================================

delta_ordered = [0] + [H_values_ordered[i] - H_values_ordered[i+1] for i in range(len(H_values_ordered)-1)]
delta_shuffled = [0] + [H_values_shuffled[i] - H_values_shuffled[i+1] for i in range(len(H_values_shuffled)-1)]

# ============================================================
# 8. 画图
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# --- 图 1 (左上): 有序加入的 R_C 曲线 ---
ax1 = axes[0, 0]
ax1.plot(steps_ordered, H_values_ordered, 'b-o', linewidth=2.5, markersize=8, zorder=5)

# 约束区域和噪声区域着色
ax1.axvspan(-0.5, 3.5, alpha=0.1, color='green', label='Constraint variables (x1-x3)')
ax1.axvspan(3.5, 10.5, alpha=0.1, color='red', label='Noise variables (x4-x10)')

# 标注最终 R_C
ax1.axhline(y=H_values_ordered[-1], color='gray', linestyle=':', linewidth=1, alpha=0.5)

ax1.set_xlabel('Number of conditioning variables', fontsize=12)
ax1.set_ylabel(r'$H(S_{t+1} \,|\, \mathrm{conditioning\;set})$ (bits)', fontsize=12)
ax1.set_title('(a) Ordered: constraints first, then noise', fontsize=13, fontweight='bold')
ax1.set_xticks(steps_ordered)
ax1.set_xticklabels(['0'] + [f'x1..{k}' for k in range(1, 11)], fontsize=8, rotation=45)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# --- 图 2 (右上): 有序加入的边际贡献 ---
ax2 = axes[0, 1]
bar_colors = ['gray'] + ['#2ecc71']*3 + ['#e74c3c']*7
ax2.bar(steps_ordered, delta_ordered, color=bar_colors, width=0.6, edgecolor='black', linewidth=0.8)
ax2.set_xlabel('Variable added at this step', fontsize=12)
ax2.set_ylabel(r'$\Delta H$ (marginal contribution, bits)', fontsize=12)
ax2.set_title('(b) Marginal contribution of each variable', fontsize=13, fontweight='bold')
ax2.set_xticks(steps_ordered)
ax2.set_xticklabels(['(none)'] + [f'x{k}' for k in range(1, 11)], fontsize=9)
ax2.grid(True, axis='y', alpha=0.3)

# 标注
ax2.annotate('Constraint\nvariables', xy=(2, delta_ordered[2]), xytext=(2, max(delta_ordered)*0.8),
             fontsize=10, color='green', fontweight='bold', ha='center')
ax2.annotate('Noise variables\n(zero contribution)', xy=(7, 0), xytext=(7, max(delta_ordered)*0.4),
             arrowprops=dict(arrowstyle='->', color='red'),
             fontsize=10, color='red', fontweight='bold', ha='center')

# --- 图 3 (左下): 打乱顺序的 R_C 曲线 ---
ax3 = axes[1, 0]
# 标记哪些步骤加入了约束变量
constraint_steps = [i+1 for i, idx in enumerate(shuffled_order) if idx < 3]
noise_steps = [i+1 for i, idx in enumerate(shuffled_order) if idx >= 3]

ax3.plot(steps_shuffled, H_values_shuffled, 'b-o', linewidth=2, markersize=6, zorder=3)

# 用不同颜色标记约束变量和噪声变量的加入点
for s in constraint_steps:
    ax3.plot(s, H_values_shuffled[s], 'go', markersize=12, zorder=5, 
             markeredgecolor='black', markeredgewidth=1.5)
for s in noise_steps:
    ax3.plot(s, H_values_shuffled[s], 'rs', markersize=8, zorder=4,
             markeredgecolor='black', markeredgewidth=1)

ax3.plot([], [], 'go', markersize=10, label='Constraint variable added')
ax3.plot([], [], 'rs', markersize=8, label='Noise variable added')

ax3.set_xlabel('Step', fontsize=12)
ax3.set_ylabel(r'$H(S_{t+1} \,|\, \mathrm{conditioning\;set})$ (bits)', fontsize=12)
ax3.set_title('(c) Shuffled order: drops only when constraint var added', fontsize=13, fontweight='bold')
ax3.set_xticks(steps_shuffled)
ax3.set_xticklabels(['0'] + [f'+{n}' for n in shuffled_names], fontsize=8, rotation=45)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# --- 图 4 (右下): 打乱顺序的边际贡献 ---
ax4 = axes[1, 1]
bar_colors_shuffled = ['gray']
for idx in shuffled_order:
    bar_colors_shuffled.append('#2ecc71' if idx < 3 else '#e74c3c')
ax4.bar(steps_shuffled, delta_shuffled, color=bar_colors_shuffled, width=0.6, 
        edgecolor='black', linewidth=0.8)
ax4.set_xlabel('Variable added at this step', fontsize=12)
ax4.set_ylabel(r'$\Delta H$ (marginal contribution, bits)', fontsize=12)
ax4.set_title('(d) Shuffled: constraint vars contribute regardless of order', fontsize=13, fontweight='bold')
ax4.set_xticks(steps_shuffled)
ax4.set_xticklabels(['(-)'] + [f'+{n}' for n in shuffled_names], fontsize=8, rotation=45)
ax4.grid(True, axis='y', alpha=0.3)

# 图例
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', edgecolor='black', label='Constraint variable'),
                   Patch(facecolor='#e74c3c', edgecolor='black', label='Noise variable')]
ax4.legend(handles=legend_elements, fontsize=9)

plt.suptitle('Experiment 3: Constraint Discovery via $R_C$', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('exp3_constraint_discovery.png', dpi=150, bbox_inches='tight')
plt.savefig('exp3_constraint_discovery.pdf', bbox_inches='tight')
print(f"\n图已保存: exp3_constraint_discovery.png / .pdf")
plt.show()

# ============================================================
# 9. 核心结论
# ============================================================

print("\n" + "=" * 60)
print("核心结论")
print("=" * 60)

total_drop_constraint = H_values_ordered[0] - H_values_ordered[3]
total_drop_noise = H_values_ordered[3] - H_values_ordered[-1]

print(f"""
有序加入结果:
  加入 x1, x2, x3 (约束变量): H 下降 {total_drop_constraint:.4f} bits
  加入 x4-x10 (噪声变量):     H 下降 {total_drop_noise:.4f} bits

打乱顺序结果:
  R_C 只在约束变量被加入时下降，与加入顺序无关。
  噪声变量无论何时加入，贡献为零。

结论:
  R_C 对变量的边际贡献 = 该变量携带的约束信息量。
  ΔH ≈ 0 → 该变量不属于约束结构（噪声）。
  ΔH > 0 → 该变量属于约束结构（携带预测信息）。
  
  R_C 可以作为约束发现(Constraint Discovery)工具:
  逐步加入变量，观察 ΔH，就能识别哪些变量属于真实约束结构。
""")