"""
Experiment 7: Empirical Validation on Natural Language (letter bigrams)
======================================================================
Estimate a 26-state letter-bigram Markov chain from a public-domain
English book, then evaluate three observers:

  (A) Expert            : P_M = T_true (empirical bigram matrix)
  (B) Transition-ignorant: P_M(j|i) = P(j)   (marginal letter frequency)
  (C) Misspecified      : P_M = column-reversed T_true (renormalized)

Quantities (base-2, bits):
  I_ORI = sum_i pi(i) sum_j T_ij log2( T_ij / P_M(j|i) )
  A_M   = I_Shannon - I_ORI
Degeneracy: observer (B) recovers I_ORI = I_Shannon exactly.

Dependencies: numpy, matplotlib (urllib from the standard library).
"""

import urllib.request
import numpy as np
import matplotlib.pyplot as plt

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
IDX = {c: i for i, c in enumerate(ALPHABET)}
N = 26

# ------------------------------------------------------------------
# 1. Obtain text (Project Gutenberg; fallback to a short embedded text)
# ------------------------------------------------------------------
def get_text():
    url = "https://www.gutenberg.org/cache/epub/11/pg11.txt"  # Alice's Adventures in Wonderland
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as r:
            txt = r.read().decode('utf-8', errors='ignore')
        if len(txt) > 50000:
            return txt, "Alice's Adventures in Wonderland (Project Gutenberg eBook #11)"
    except Exception as e:
        print(f"  [download failed: {e}; using embedded fallback]")
    # Fallback: a public-domain passage (short; statistics noisier but valid)
    fallback = (
        "Alice was beginning to get very tired of sitting by her sister on the "
        "bank and of having nothing to do once or twice she had peeped into the "
        "book her sister was reading but it had no pictures or conversations in "
        "it and what is the use of a book thought Alice without pictures or "
        "conversations so she was considering in her own mind as well as she "
        "could for the hot day made her feel very sleepy and stupid whether the "
        "pleasure of making a daisy chain would be worth the trouble of getting "
        "up and picking the daisies when suddenly a white rabbit with pink eyes "
        "ran close by her "
    ) * 60
    return fallback, "embedded fallback passage"

text, source = get_text()

# ------------------------------------------------------------------
# 2. Estimate bigram chain
# ------------------------------------------------------------------
seq = [IDX[c] for c in text.lower() if c in IDX]
print("=" * 66)
print("Experiment 7: Empirical Validation on Natural Language")
print("=" * 66)
print(f"Source: {source}")
print(f"Letters used (a-z): {len(seq)}")

counts = np.zeros((N, N))
for a, b in zip(seq[:-1], seq[1:]):
    counts[a, b] += 1.0

eps = 1.0  # add-one (Laplace) smoothing -> no zero probabilities
T_true = (counts + eps) / (counts.sum(axis=1, keepdims=True) + N * eps)

# stationary / letter distribution from empirical unigram frequencies
unig = np.bincount(seq, minlength=N).astype(float)
pi = unig / unig.sum()
P_marg = pi @ T_true

# ------------------------------------------------------------------
# 3. Core quantities
# ------------------------------------------------------------------
def I_ORI(P_M):
    val = 0.0
    for i in range(N):
        for j in range(N):
            if T_true[i, j] > 1e-15:
                val += pi[i] * T_true[i, j] * np.log2(T_true[i, j] / P_M[i, j])
    return val

P_M_ignorant = np.tile(P_marg, (N, 1))
I_shannon = I_ORI(P_M_ignorant)

# Observer C: reverse column order of each row, then renormalize
P_M_reversed = T_true[:, ::-1].copy()
P_M_reversed /= P_M_reversed.sum(axis=1, keepdims=True)

observers = {
    'Expert\n(empirical bigram)':       T_true,
    'Transition-ignorant\n(marginal)':  P_M_ignorant,
    'Misspecified\n(reversed bigram)':  P_M_reversed,
}

print(f"\nI_Shannon = {I_shannon:.4f} bits (letter-bigram mutual information)\n")
print(f"{'Observer':<28s} | {'I_ORI':>8s} | {'A_M':>9s}")
print("-" * 52)
names, I_vals, A_vals = [], [], []
for name, P_M in observers.items():
    iori = I_ORI(P_M)
    a_m = I_shannon - iori
    names.append(name); I_vals.append(iori); A_vals.append(a_m)
    print(f"{name.replace(chr(10), ' '):<28s} | {iori:8.4f} | {a_m:+9.4f}")

ratio_C = I_vals[2] / I_shannon
print(f"\nMisspecified observer: I_ORI = {ratio_C:.1f} x I_Shannon, "
      f"A_M = {A_vals[2]:+.4f} bits (negative knowledge)")
print(f"Degeneracy check: I_ORI(ignorant) - I_Shannon = {I_vals[1] - I_shannon:.2e}")

# ------------------------------------------------------------------
# 4. Plot (two panels, same format as Experiment 2)
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors = ['#2ecc71', '#f39c12', '#8e44ad']
x = np.arange(len(names))

ax1 = axes[0]
ax1.bar(x, I_vals, color=colors, width=0.6, edgecolor='black', linewidth=0.8)
ax1.axhline(y=I_shannon, color='red', linestyle='--', linewidth=2,
            label=r'$I_{Shannon}$ (ignorance baseline)')
for i, v in enumerate(I_vals):
    ax1.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=9)
ax1.set_ylabel(r'$I_{ORI}$ (bits)', fontsize=13)
ax1.set_title(r'$I_{ORI}$: Observer-Relative Information', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, axis='y', alpha=0.3)

ax2 = axes[1]
bar_colors = ['#2ecc71' if v >= 0 else '#e74c3c' for v in A_vals]
ax2.bar(x, A_vals, color=bar_colors, width=0.6, edgecolor='black', linewidth=0.8)
ax2.axhline(y=0, color='black', linewidth=1)
ax2.axhline(y=I_shannon, color='blue', linestyle='--', linewidth=1.5, alpha=0.5,
            label=r'$I_{Shannon}$ (maximum $A_M$)')
for i, v in enumerate(A_vals):
    off = 0.02 if v >= 0 else -0.03
    va = 'bottom' if v >= 0 else 'top'
    ax2.text(i, v + off, f'{v:+.4f}', ha='center', va=va, fontsize=9, fontweight='bold')
ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=9)
ax2.set_ylabel(r'$A_M$ (Observer Advantage, bits)', fontsize=13)
ax2.set_title(r'$A_M = I_{Shannon} - I_{ORI}$: Observer Advantage', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('exp7_bigram.png', dpi=150, bbox_inches='tight')
plt.savefig('exp7_bigram.pdf', bbox_inches='tight')
print("\nSaved: exp7_bigram.png / .pdf")
plt.show()
