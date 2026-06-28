"""
Figure 1: Overview of the observer-relative information-theoretic framework.
Pure matplotlib schematic (no external graph tools). Outputs png + pdf.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BLUE = '#E3F2FD'      # inner core boxes
GRAYBG = '#F5F5F5'    # framework container
EDGE = '#1565C0'      # blue edge
DARK = '#222222'
GRAY = '#555555'

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-0.3, 10)
ax.set_ylim(0, 10)
ax.axis('off')


def rbox(x, y, w, h, fc=BLUE, ec=EDGE, lw=1.4, dashed=False, z=2):
    bb = FancyBboxPatch((x, y), w, h,
                        boxstyle="round,pad=0.015,rounding_size=0.10",
                        linewidth=lw, facecolor=fc, edgecolor=ec, zorder=z)
    if dashed:
        bb.set_linestyle((0, (4, 3)))
    ax.add_patch(bb)


def lines(cx, cy, items, spacing=0.30):
    """items: list of (text, fontsize, weight). Vertically centered at (cx, cy)."""
    n = len(items)
    ys = [cy + (n - 1) * spacing / 2 - i * spacing for i in range(n)]
    for (s, fs, w), yy in zip(items, ys):
        ax.text(cx, yy, s, ha='center', va='center', fontsize=fs,
                fontweight=w, color=DARK, zorder=4)


def arrow(p0, p1, dashed=False, rad=0.0, lw=1.6, color=GRAY):
    a = FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=16,
                        linewidth=lw, color=color, zorder=3,
                        connectionstyle=f"arc3,rad={rad}")
    if dashed:
        a.set_linestyle((0, (4, 3)))
    ax.add_patch(a)


# ---- Layer 1: Observer Model ----
rbox(4.6, 8.95, 2.8, 0.75, fc=BLUE)
lines(6.0, 9.32, [(r'Observer Model  $P_M$', 12, 'bold')])
arrow((6.0, 8.93), (6.0, 8.55))

# ---- Layer 2: Framework container ----
rbox(2.3, 3.5, 7.4, 5.0, fc=GRAYBG, ec=DARK, lw=1.7, z=1)
ax.text(6.0, 8.15, 'Observer-Relative Framework', ha='center', va='center',
        fontsize=13, fontweight='bold', color=DARK, zorder=4)

# three core quantities
rbox(2.6, 6.45, 2.1, 1.1)
lines(3.65, 7.0, [(r'$I_{\mathrm{ORI}}$', 14, 'bold'),
                  ('Observer-Relative', 8.5, 'normal'),
                  ('Information', 8.5, 'normal')], spacing=0.30)
rbox(4.95, 6.45, 2.1, 1.1)
lines(6.0, 7.0, [(r'$A_M$', 14, 'bold'),
                 ('Observer', 8.5, 'normal'),
                 ('Advantage', 8.5, 'normal')], spacing=0.30)
rbox(7.3, 6.45, 2.1, 1.1)
lines(8.35, 7.0, [(r'$R_C$', 14, 'bold'),
                  ('Constraint', 8.5, 'normal'),
                  ('Residual', 8.5, 'normal')], spacing=0.30)

# divider
ax.plot([2.55, 9.45], [6.05, 6.05], color=GRAY, lw=1.0, zorder=2)

# derived structures
rbox(2.6, 4.2, 3.0, 1.15)
lines(4.10, 4.78, [('Observer Spectrum', 11, 'bold'),
                   ('(Table 1)', 9, 'normal')], spacing=0.36)
rbox(6.4, 4.2, 3.0, 1.15)
lines(7.90, 4.78, [('Triple Decomposition', 11, 'bold'),
                   (r'$H = A_M + I_{\mathrm{ORI}} + R_C$', 10, 'normal')], spacing=0.36)

# ---- Layer 3: consequences ----
arrow((4.0, 3.5), (4.0, 3.08))
arrow((8.0, 3.5), (8.0, 3.08))

rbox(2.3, 1.75, 3.4, 1.3, fc='#FDECEA', ec='#C0392B')
lines(4.0, 2.40, [('Negative Knowledge', 11, 'bold'),
                  (r'$A_M < 0$', 11, 'normal'),
                  ('Asymmetry Theorem', 8.5, 'normal')], spacing=0.34)
rbox(6.3, 1.75, 3.4, 1.3, fc='#E8F5E9', ec='#27AE60')
lines(8.0, 2.40, [('Recovery Scaling Law', 11, 'bold'),
                  (r'$n^* = \alpha\, c$', 11, 'normal'),
                  ('Theorem 1', 8.5, 'normal')], spacing=0.34)

# ---- Shannon special case (bottom-left, dashed) ----
rbox(0.05, 0.35, 3.3, 1.0, fc='white', ec=GRAY, lw=1.3, dashed=True)
lines(1.70, 0.85, [('Shannon mutual information', 8.7, 'bold'),
                   (r'special case:  $P_M = P(Y)$', 8.7, 'normal')], spacing=0.34)
# dashed arrow up the clear left margin into the I_ORI box
arrow((1.70, 1.36), (2.62, 6.55), dashed=True, rad=-0.18, lw=1.3, color=GRAY)
ax.text(0.55, 4.35, r'$P_M = P(Y)$' '\n' r'$\Rightarrow I_{\mathrm{ORI}} = I_{\mathrm{Shannon}}$',
        ha='center', va='center', fontsize=8, color=GRAY, zorder=4)

plt.savefig('fig1_framework_overview.png', dpi=150, bbox_inches='tight')
plt.savefig('fig1_framework_overview.pdf', bbox_inches='tight')
print("Saved: fig1_framework_overview.png / .pdf")
