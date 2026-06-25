"""
Experiment 6: Knowledge Phase Diagram in a Physical System
==============================================================
粗粒化双势阱：3 状态 {Left well, Barrier, Right well}
所有状态之间都可以转移（密集转移矩阵）。

真实系统：粒子偏好右阱（右阱更深）
观察者信念：可能认为左阱更深（信念偏差 delta）

少状态 + 密集矩阵 = 拓扑优势消失 → 方向性错误可以导致负知识。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ============================================================
# Core functions
# ============================================================

def kl_div(p, q):
    r = 0.0
    for pi, qi in zip(p, q):
        if pi > 1e-15:
            qi = max(qi, 1e-15)
            r += pi * np.log2(pi / qi)
    return r

def compute_I_ORI(T_true, P_M, pi):
    return sum(pi[s] * kl_div(T_true[s], P_M[s]) for s in range(len(pi)))

def compute_I_Shannon(T_true, pi):
    pm = pi @ T_true
    P_M_ign = np.tile(pm, (len(pi), 1))
    return compute_I_ORI(T_true, P_M_ign, pi)

def compute_A_M(T_true, P_M, pi):
    return compute_I_Shannon(T_true, pi) - compute_I_ORI(T_true, P_M, pi)

def stationary_dist(T):
    evals, evecs = np.linalg.eig(T.T)
    idx = np.argmin(np.abs(evals - 1.0))
    pi = np.real(evecs[:, idx])
    pi = pi / pi.sum()        # normalize first (fixes eigenvector sign convention)
    pi = np.maximum(pi, 0)    # clip residual numerical noise
    return pi / pi.sum()

# ============================================================
# Physical model: 3-state coarse-grained double well
# ============================================================

def make_transition_matrix(E_barrier, asymmetry, kT):
    """
    3 states: L(left well), B(barrier top), R(right well)
    
    Energy levels:
      E_L = -asymmetry  (negative asymmetry = left well deeper)
      E_B = E_barrier    (barrier top)
      E_R = +asymmetry   (positive asymmetry = right well deeper)
    
    Transition rates (Arrhenius): k(i->j) = exp(-max(E_j - E_i, 0) / kT)
    Then normalize each row.
    """
    E = np.array([-asymmetry, E_barrier, asymmetry])
    # Note: positive asymmetry -> E_R = +asymmetry (higher energy for right)
    # We want positive asymmetry -> right well DEEPER -> E_R should be LOWER
    # So flip: E_R = -asymmetry
    E = np.array([asymmetry, E_barrier, -asymmetry])
    # Now: asymmetry > 0 -> E_L > E_R -> right well deeper
    
    N = 3
    T = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            dE = E[j] - E[i]
            T[i, j] = np.exp(-max(dE, 0) / kT)
        T[i] /= T[i].sum()
    
    return T, E

states = ['Left', 'Barrier', 'Right']

# ============================================================
# Experiment parameters
# ============================================================

kT = 1.0
asym_true = 1.5  # positive = right well deeper

print("=" * 65)
print("Experiment 6: Knowledge Phase Diagram (Coarse-Grained Double Well)")
print("=" * 65)
print(f"3 states: Left well, Barrier, Right well")
print(f"True asymmetry = {asym_true} (right well deeper)")
print(f"kT = {kT}")

# Show true transition matrix
T_true, E_true = make_transition_matrix(3.0, asym_true, kT)
pi_true = stationary_dist(T_true)
print(f"\nTrue energy levels (E_b=3.0): L={E_true[0]:.1f}, B={E_true[1]:.1f}, R={E_true[2]:.1f}")
print(f"True T:\n{np.array2string(T_true, precision=3)}")
print(f"Stationary: {dict(zip(states, [f'{p:.3f}' for p in pi_true]))}")

# ============================================================
# Panel 1 data: A_M vs observer asymmetry belief
# ============================================================

asym_M_range = np.linspace(-4.0, 4.0, 400)
barrier_heights = [1.5, 3.0, 5.0, 8.0]
barrier_colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']

results = {}
for E_b in barrier_heights:
    T_true, E_true = make_transition_matrix(E_b, asym_true, kT)
    pi = stationary_dist(T_true)
    I_sh = compute_I_Shannon(T_true, pi)
    
    A_M_curve = []
    for asym_M in asym_M_range:
        P_M, _ = make_transition_matrix(E_b, asym_M, kT)
        a_m = compute_A_M(T_true, P_M, pi)
        A_M_curve.append(a_m)
    A_M_curve = np.array(A_M_curve)
    
    zeros = []
    for i in range(len(A_M_curve)-1):
        if A_M_curve[i] * A_M_curve[i+1] < 0:
            d = asym_M_range[i] + (asym_M_range[i+1]-asym_M_range[i]) * (-A_M_curve[i])/(A_M_curve[i+1]-A_M_curve[i])
            zeros.append(d)
    
    results[E_b] = {'A_M': A_M_curve, 'I_Shannon': I_sh, 'zeros': zeros}
    
    idx_p = np.argmin(np.abs(asym_M_range - asym_true))
    idx_0 = np.argmin(np.abs(asym_M_range))
    idx_n = np.argmin(np.abs(asym_M_range + asym_true))
    print(f"\n  E_b={E_b}: I_Sh={I_sh:.4f}")
    print(f"    A_M(perfect)={A_M_curve[idx_p]:+.4f}, A_M(sym)={A_M_curve[idx_0]:+.4f}, A_M(reversed)={A_M_curve[idx_n]:+.4f}")
    if zeros: print(f"    Zero crossings: {[f'{z:.2f}' for z in zeros]}")
    else: print(f"    No zero crossing (A_M always positive)")

# ============================================================
# Panel 3 data: 2D phase diagram
# ============================================================

print("\nComputing 2D phase diagram...")
barrier_range_2d = np.linspace(1.0, 10.0, 100)
asym_range_2d = np.linspace(-4.0, 4.0, 100)
A_M_2d = np.zeros((len(barrier_range_2d), len(asym_range_2d)))

for bi, E_b in enumerate(barrier_range_2d):
    T_true, _ = make_transition_matrix(E_b, asym_true, kT)
    pi = stationary_dist(T_true)
    for ai, asym_M in enumerate(asym_range_2d):
        P_M, _ = make_transition_matrix(E_b, asym_M, kT)
        A_M_2d[bi, ai] = compute_A_M(T_true, P_M, pi)
print("Done.")

has_negative = np.any(A_M_2d < 0)
print(f"Negative knowledge present: {has_negative}")
if has_negative:
    print(f"  Min A_M = {A_M_2d.min():.4f} at barrier={barrier_range_2d[np.unravel_index(A_M_2d.argmin(), A_M_2d.shape)[0]]:.1f}, "
          f"asym_M={asym_range_2d[np.unravel_index(A_M_2d.argmin(), A_M_2d.shape)[1]]:.1f}")

# ============================================================
# Plot
# ============================================================

fig = plt.figure(figsize=(14, 11))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.25], hspace=0.30, wspace=0.22)

# --- Panel 1: Energy levels ---
ax0 = fig.add_subplot(gs[0, 0])
E_b_show = 3.0
_, E_true_show = make_transition_matrix(E_b_show, asym_true, kT)
_, E_wrong_show = make_transition_matrix(E_b_show, -asym_true, kT)

bar_width = 0.3
x_pos = np.array([0, 1, 2])

bars1 = ax0.bar(x_pos - bar_width/2, E_true_show, bar_width, color=['#3498db','#95a5a6','#3498db'], 
                edgecolor='black', linewidth=1.5, label='True energy', alpha=0.8)
bars2 = ax0.bar(x_pos + bar_width/2, E_wrong_show, bar_width, color=['#e74c3c','#95a5a6','#e74c3c'],
                edgecolor='black', linewidth=1.5, label='Wrong belief', alpha=0.5, hatch='///')

ax0.set_xticks(x_pos)
ax0.set_xticklabels(['Left Well', 'Barrier', 'Right Well'], fontsize=11)
ax0.set_ylabel('Energy', fontsize=12)
ax0.set_title(f'True vs Wrong Energy Levels ($E_b={E_b_show}$)', fontsize=13, fontweight='bold')
ax0.legend(fontsize=10)
ax0.grid(True, axis='y', alpha=0.3)

ax0.annotate('True: right\nis deeper', xy=(2-bar_width/2, E_true_show[2]+0.05), fontsize=9, color='blue',
             ha='center', fontweight='bold')
ax0.annotate('Wrong: thinks\nleft is deeper', xy=(0+bar_width/2, E_wrong_show[0]+0.05), fontsize=9, color='red',
             ha='center', fontweight='bold')

# --- Panel 2: A_M vs delta_M ---
ax1 = fig.add_subplot(gs[0, 1])
for E_b, color in zip(barrier_heights, barrier_colors):
    r = results[E_b]
    ax1.plot(asym_M_range, r['A_M'], color=color, linewidth=2.5, label=f'$E_b = {E_b}$')
    for z in r['zeros']:
        ax1.plot(z, 0, 'o', color=color, markersize=8, markeredgecolor='black', markeredgewidth=1.5, zorder=5)

ax1.axhline(y=0, color='black', linestyle='--', linewidth=2, label='$A_M = 0$ (phase boundary)')
ax1.axvline(x=asym_true, color='green', linestyle=':', linewidth=1.5, alpha=0.7)
ax1.text(asym_true+0.1, max(r['A_M'])*0.9, '$\\delta_{true}$', fontsize=10, color='green')

# Shade negative
any_neg = False
for E_b in barrier_heights:
    r = results[E_b]
    if np.any(np.array(r['A_M']) < 0):
        any_neg = True
        ax1.fill_between(asym_M_range, min(r['A_M'])*1.05, 0, where=(np.array(r['A_M'])<0),
                          alpha=0.05, color='red')

if any_neg:
    ax1.text(-3, min([min(results[e]['A_M']) for e in barrier_heights])*0.5,
             'Negative\nKnowledge', fontsize=12, color='red', fontweight='bold', alpha=0.6)

ax1.text(asym_true, max([max(results[e]['A_M']) for e in barrier_heights])*0.5,
         'Positive\nKnowledge', fontsize=12, color='green', fontweight='bold', alpha=0.6)

ax1.set_xlabel('Observer belief asymmetry $\\delta_M$', fontsize=12)
ax1.set_ylabel('$A_M$ (Observer Advantage, bits)', fontsize=12)
ax1.set_title('Knowledge vs Belief Error', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)

# --- Panel 3: 2D phase diagram (hero panel, spans full width) ---
ax2 = fig.add_subplot(gs[1, :])

if has_negative:
    vmax_val = np.percentile(A_M_2d[A_M_2d>0], 95) if np.any(A_M_2d>0) else 0.1
    vmin_val = np.percentile(A_M_2d[A_M_2d<0], 5) if np.any(A_M_2d<0) else -0.01
    norm = TwoSlopeNorm(vmin=vmin_val, vcenter=0, vmax=vmax_val)
    cmap = 'RdYlGn'
else:
    norm = None
    cmap = 'Greens'

im = ax2.pcolormesh(asym_range_2d, barrier_range_2d, A_M_2d, cmap=cmap, norm=norm, shading='auto')
plt.colorbar(im, ax=ax2, label='$A_M$ (bits)')

if has_negative:
    try:
        cs = ax2.contour(asym_range_2d, barrier_range_2d, A_M_2d, levels=[0], colors='black', linewidths=3)
        ax2.clabel(cs, fmt='$A_M=0$', fontsize=10)
    except: pass

ax2.axvline(x=asym_true, color='white', linestyle='--', linewidth=2, alpha=0.8)
ax2.text(asym_true+0.15, 9, '$\\delta_{true}$', color='white', fontsize=11, fontweight='bold')

ax2.set_xlabel('Observer belief $\\delta_M$', fontsize=12)
ax2.set_ylabel('Barrier height $E_b$', fontsize=12)
ax2.set_title('2D Phase Diagram: $A_M(\\delta_M, E_b)$', fontsize=13, fontweight='bold')

plt.suptitle('Experiment 6: Knowledge Phase Diagram — Coarse-Grained Double Well',
             fontsize=15, fontweight='bold', y=0.99)
plt.savefig('exp6_phase_diagram.png', dpi=150, bbox_inches='tight')
plt.savefig('exp6_phase_diagram.pdf', bbox_inches='tight')
print(f"\nSaved: exp6_phase_diagram.png / .pdf")
plt.show()