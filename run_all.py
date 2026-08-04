"""
run_all.py
==========
Reproduce every figure in the README and print the key diagnostics.

    python run_all.py

Figures are written to ./figures/.  Runtime is a few minutes, dominated by the
bifurcation scan.
"""
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import chemostat as cm

FIG = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG, exist_ok=True)

# ---- shared aesthetic ------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
COL = {"n": "#3b6ea5", "r": "#c1440e", "c": "#e0a800", "p": "#2a7f62"}


def banner(msg):
    print(f"\n{'='*64}\n{msg}\n{'='*64}")


# ---------------------------------------------------------------------------
# 1. Chaotic time series
# ---------------------------------------------------------------------------
def fig_time_series():
    banner("Figure 1 / 6 : chaotic time series")
    t, Y = cm.simulate(cm.D_CHAOS, t_end=120_000, n_points=120_000)
    t = t / 24.0  # hours -> days for readability
    n, r, c, p = Y

    fig, ax = plt.subplots(4, 1, figsize=(9, 8), sharex=True)
    ax[0].plot(t, n, color=COL["n"], lw=0.8);  ax[0].set_ylabel("nutrient\n(mg/cc)")
    ax[1].plot(t, r, color=COL["r"], lw=0.8);  ax[1].set_ylabel("rods\n(cells/cc)")
    ax[2].plot(t, c, color=COL["c"], lw=0.8);  ax[2].set_ylabel("cocci\n(cells/cc)")
    ax[3].plot(t, p, color=COL["p"], lw=0.8);  ax[3].set_ylabel("predators\n(cells/cc)")
    ax[3].set_xlabel("time (days)")
    ax[0].set_title(f"Deterministic chaos at D = {cm.D_CHAOS}/hr "
                    "— irregular, non-repeating, all three microbes coexist")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "01_time_series.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Strange attractor (3-D phase portrait)
# ---------------------------------------------------------------------------
def fig_attractor():
    banner("Figure 2 / 6 : strange attractor")
    t, Y = cm.simulate(cm.D_CHAOS, t_end=400_000, n_points=400_000,
                       discard_frac=0.15)
    r, c, p = Y[1], Y[2], Y[3]

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(r, c, p, lw=0.25, color="#4b3b7a", alpha=0.8)
    ax.set_xlabel("rods (cells/cc)")
    ax.set_ylabel("cocci (cells/cc)")
    ax.set_zlabel("predators (cells/cc)")
    ax.set_title("Strange attractor of the chemostat "
                 f"(D = {cm.D_CHAOS}/hr)")
    ax.view_init(elev=22, azim=-60)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "02_strange_attractor.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Three regimes
# ---------------------------------------------------------------------------
def fig_regimes():
    banner("Figure 3 / 6 : three dilution-rate regimes")
    cases = [
        (cm.D_CHAOS,     "chaos — all coexist"),
        (cm.D_COCCI_DIE, "steady state — cocci wash out"),
        (cm.D_RODS_DIE,  "steady state — rods wash out"),
    ]
    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for a, (D, label) in zip(ax, cases):
        t, Y = cm.simulate(D, t_end=90_000, n_points=90_000)
        td = t / 24.0
        a.plot(td, Y[1], color=COL["r"], lw=0.7, label="rods")
        a.plot(td, Y[2], color=COL["c"], lw=0.7, label="cocci")
        a.plot(td, Y[3] * 100, color=COL["p"], lw=0.7, label="predators ×100")
        a.set_yscale("symlog", linthresh=1e3)
        a.set_ylabel("cells/cc")
        a.set_title(f"D = {D}/hr   —   {label}", fontsize=10)
        alive = cm.survivors(Y)
        print(f"   D={D}: survivors -> {alive}")
    ax[0].legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax[-1].set_xlabel("time (days)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "03_regimes.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Bifurcation diagram over the dilution rate
# ---------------------------------------------------------------------------
def fig_bifurcation():
    banner("Figure 4 / 6 : bifurcation diagram (this is the slow one, ~4 min)")
    # dense scan across the narrow band where all three microbes coexist
    Ds = np.linspace(0.02035, 0.02150, 78)
    xs, ys = [], []
    t0 = time.time()
    for i, D in enumerate(Ds):
        rc = cm.poincare_r(D, t_end=95_000, n_points=150_000, discard_frac=0.4)
        if rc.size:
            keep = rc[-80:]                      # last 80 section points
            xs.extend([D] * len(keep))
            ys.extend(keep)
        if (i + 1) % 20 == 0:
            print(f"   {i+1}/{len(Ds)} dilution rates  "
                  f"({time.time()-t0:.0f}s elapsed)")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(xs, np.array(ys) / 1e6, ".", ms=1.3, color="#222222", alpha=0.5)
    ax.axvline(cm.D_CHAOS, color=COL["r"], lw=1, ls="--",
               label=f"D = {cm.D_CHAOS} (chaotic run)")
    ax.set_xlabel("dilution rate  D  (1/hr)")
    ax.set_ylabel("rod density at Poincaré section  (10$^6$ cells/cc)")
    ax.set_title("Bifurcation diagram: chaotic bands threaded by periodic windows")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "04_bifurcation.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Sensitivity to initial conditions
# ---------------------------------------------------------------------------
def fig_sensitivity():
    banner("Figure 5 / 6 : sensitivity to initial conditions")
    D = cm.D_CHAOS
    # land on the attractor, then split into two trajectories a hair apart
    t0, Y0 = cm.simulate(D, t_end=8_000, n_points=4_000)
    ya = Y0[:, -1].copy()
    yb = ya.copy()
    yb[0] += 1e-8           # perturb nutrient by 1e-8 mg/cc

    ta, Ya = cm.simulate(D, t_end=30_000, n_points=30_000, y0=ya)
    tb, Yb = cm.simulate(D, t_end=30_000, n_points=30_000, y0=yb)
    td = ta / 24.0

    scale = np.array([0.05, 1e6, 1e6, 3e3])
    sep = np.linalg.norm(((Ya - Yb).T / scale), axis=1)

    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    ax[0].plot(td, Ya[1], color=COL["r"], lw=0.7, label="trajectory A")
    ax[0].plot(td, Yb[1], color="#6a2a0a", lw=0.7, ls="--", label="trajectory B")
    ax[0].set_ylabel("rods (cells/cc)")
    ax[0].set_title("Two runs differing by 1e-8 mg/cc in initial nutrient")
    ax[0].legend(loc="upper right", fontsize=8)

    ax[1].semilogy(td, sep, color="#333333", lw=0.8)
    ax[1].set_ylabel("state separation\n(normalised)")
    ax[1].set_xlabel("time (days)")
    ax[1].set_title("Exponential divergence until saturation "
                    "— the practical forecast horizon", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "05_sensitivity.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Lyapunov-exponent convergence + numeric diagnostics
# ---------------------------------------------------------------------------
def fig_lyapunov():
    banner("Figure 6 / 6 : Lyapunov exponent + diagnostics")
    lam, hist = cm.largest_lyapunov(cm.D_CHAOS, t_run=150_000,
                                    return_history=True)
    lyap_time = 1.0 / lam
    print(f"   largest Lyapunov exponent  = {lam:.5f} / hr")
    print(f"   Lyapunov time (1/lambda)   = {lyap_time:.0f} hr  "
          f"= {lyap_time/24:.0f} days")

    # 0-1 test on the same run for an independent confirmation
    t, Y = cm.simulate(cm.D_CHAOS, t_end=200_000, n_points=200_000,
                       discard_frac=0.4)
    K = cm.zero_one_test(Y[1])
    print(f"   0-1 test statistic K       = {K:.3f}  (K~1 => chaos)")

    steps = np.arange(1, len(hist) + 1) * 50 / 24.0  # days
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, hist, color="#4b3b7a", lw=1.2)
    ax.axhline(lam, color=COL["r"], ls="--", lw=1,
               label=f"converged $\\lambda$ = {lam:.4f}/hr")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("integration time (days)")
    ax.set_ylabel("running estimate of  $\\lambda_{max}$  (1/hr)")
    ax.set_title(f"Positive Lyapunov exponent confirms chaos "
                 f"(0-1 test K = {K:.2f})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "06_lyapunov.png"))
    plt.close(fig)
    return lam, lyap_time, K


if __name__ == "__main__":
    start = time.time()
    fig_time_series()
    fig_attractor()
    fig_regimes()
    fig_sensitivity()
    lam, lyap_time, K = fig_lyapunov()
    fig_bifurcation()   # slowest, run last
    banner(f"Done in {time.time()-start:.0f}s. Figures written to ./figures/")
