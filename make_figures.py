"""Redraw Figures 1 and 2 of "The Complexity of Query-Answer Survival under Correlated Data Loss".

Usage:  python make_figures.py [results_folder]
The folder (default: ./results) must contain q2_sf01_per_theta.csv and dense_per_theta.csv,
as written by S1_reproducibility_notebook.ipynb. Figures are saved to the same folder.
Requires numpy and matplotlib.
"""
import csv, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = sys.argv[1] if len(sys.argv) > 1 else 'results'
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'],
                     'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e',
                     'ytick.color': '#52514e', 'axes.grid': True, 'grid.color': '#e6e5e1', 'grid.linewidth': 0.6,
                     'mathtext.fontset': 'stix'})
BLUE, ORANGE, YELLOW, INK, GREY = '#2a78d6', '#eb6834', '#eda100', '#0b0b0b', '#52514e'
th = np.linspace(0, 1, 401); qq = 1 - th

# Figure 1(a): 4-cycle a1 versus path a2 under tuple loss (Section 7.1, equation (14))
S1 = qq**2 * (2 - qq)**2
S2 = qq**2 * (qq**3 - qq**2 - 3 * qq + 4)

# Figure 1(b): family-A answers with component sizes (1, 5) and (2, 3) under hybrid loss (Section 7.3)
def component(d, q):          # R-fact, its unit, and at least one of d S-facts
    return q**2 * (1 - (1 - q)**d)

def answer(ds, q):
    p = np.ones_like(q, dtype=float)
    for d in ds:
        p = p * (1 - component(d, q))
    return 1 - p

def diff(t):
    return float(answer([1, 5], np.array([1 - t]))[0] - answer([2, 3], np.array([1 - t]))[0])

lo, hi = 0.5, 0.95
for _ in range(80):           # bisection for the crossing severity
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if diff(lo) * diff(mid) > 0 else (lo, mid)
theta_cross = (lo + hi) / 2
print('crossing severity:', round(theta_cross, 4))

fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.6), constrained_layout=True)
a = ax[0]
a.plot(th, S1, color=BLUE, lw=1.6, label='$a_1$ (4-cycle)')
a.plot(th, S2, color=ORANGE, lw=1.6, ls='--', label='$a_2$ (path)')
a.set_xlabel('Severity θ'); a.set_ylabel('Survival $S(θ)$')
a.set_title('(a) Equal resilience, strict dominance', fontsize=9, loc='left')
a.legend(frameon=False, fontsize=8, loc='lower left'); a.set_xlim(0, 1); a.set_ylim(0, 1.02)
ins = a.inset_axes([0.62, 0.55, 0.36, 0.4]); ins.plot(th, S2 - S1, color=GREY, lw=1.2)
ins.set_title('$S_2-S_1$', fontsize=7); ins.tick_params(labelsize=6); ins.grid(False)
a = ax[1]
a.axhline(0, color=GREY, lw=0.8)
a.plot(th, answer([1, 5], qq) - answer([2, 3], qq), color=BLUE, lw=1.6)
a.plot([theta_cross], [0], 'o', ms=6, mfc='white', mec=INK, mew=1.2, zorder=5)
a.annotate(f'crossing at θ ≈ {theta_cross:.3f};\n(1, 5) more robust beyond it', (theta_cross, 0),
           xytext=(0.8, -0.032), fontsize=8, ha='center', arrowprops=dict(arrowstyle='-', color=GREY, lw=0.8))
a.text(0.33, -0.012, '(2, 3) more robust', fontsize=8, color=GREY, ha='center')
a.set_xlabel('Severity θ'); a.set_ylabel('Difference in survival')
a.set_title('(b) Equal resilience, crossing curves', fontsize=9, loc='left'); a.set_xlim(0, 1)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(RES, f'figure1.{ext}'), dpi=300)
plt.close(fig)

# Figure 2(a): Q2 grouped by supplier, SF 0.1; Figure 2(b): dense sharing
rows = list(csv.DictReader(open(os.path.join(RES, 'q2_sf01_per_theta.csv'))))
col = lambda k: np.array([float(r[k]) for r in rows])
t2, ex, lob, upb, uf = col('theta'), col('exact'), col('lower'), col('matched_independence'), col('units_ignored')
dr = list(csv.DictReader(open(os.path.join(RES, 'dense_per_theta.csv'))))
dcol = lambda k: np.array([float(r[k]) for r in dr])

fig, ax = plt.subplots(1, 2, figsize=(6.5, 2.7), constrained_layout=True)
a = ax[0]
a.plot(t2, uf, color=YELLOW, lw=1.6, ls=':', marker='^', ms=3.5, label='Unit-free model (q)')
a.plot(t2, upb, color=ORANGE, lw=1.6, ls='--', label='Upper bound (Prop. 5)')
a.plot(t2, ex, color=INK, lw=1.0, marker='o', ms=3.5, mfc='white', label='Exact')
a.plot(t2, lob, color=BLUE, lw=1.6, ls='-.', label='Lower bound (18)')
a.set_xlabel('Severity θ'); a.set_ylabel('Mean survival')
a.set_title('(a) $Q_2$ by supplier, SF 0.1 (10,000 answers)', fontsize=9, loc='left')
a.legend(frameon=False, fontsize=7.5, loc='lower left'); a.set_xlim(0, 1); a.set_ylim(0, 1.03)
a = ax[1]; td = dcol('theta')
a.plot(td, dcol('mean_width'), color=INK, lw=1.0, marker='o', ms=3.5, mfc='white', label='Mean bracket width')
a.plot(td, dcol('mean_err_lower'), color=BLUE, lw=1.6, ls='-.', label='Mean error, lower bound')
a.plot(td, dcol('mean_err_upper'), color=ORANGE, lw=1.6, ls='--', label='Mean error, upper bound')
a.plot(td, dcol('max_err_upper'), color=ORANGE, lw=1.0, ls=':', label='Max. error, upper bound')
a.set_xlabel('Severity θ'); a.set_ylabel('Absolute difference')
a.set_title('(b) Dense sharing (100 instances)', fontsize=9, loc='left')
a.legend(frameon=False, fontsize=7.5, loc='upper left', ncol=2); a.set_xlim(0, 1); a.set_ylim(0, 0.72)
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(RES, f'figure2.{ext}'), dpi=300)
plt.close(fig)
print('Figures written to', os.path.abspath(RES))
