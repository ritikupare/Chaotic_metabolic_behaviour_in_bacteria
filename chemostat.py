"""
chemostat.py
============
Four-component microbial chemostat of Molz, Faybishenko & Agarwal (2019),
which reproduces the deterministic chaos measured experimentally by
Becks et al. (2005, Nature).

State variables
---------------
    n : nutrient concentration            (mg / cc)
    r : rod  (bacterium) concentration    (cells / cc)   -- strong nutrient competitor
    c : coccus (bacterium) concentration  (cells / cc)   -- weaker competitor
    p : ciliate predator concentration    (cells / cc)   -- eats both, prefers rods

The model is Eq. (1.10) of the LBNL report (LBNL-2001172).  Each microbe is
tracked in cell numbers; multiplying by a mean cell mass (m_r, m_c, m_p) converts
to biomass so that the nutrient mass balance is consistent.  Two features are what
make the system chaotic rather than merely oscillatory:

  1. The predator's preference for rods over cocci *increases with rod density*
     (the (m1*r + i1) factor and the extra m2*r term in the half-saturation).
  2. Dead predator biomass is recycled back into nutrient at efficiency EF.

Reference
---------
F. Molz, B. Faybishenko, D. Agarwal (2019). "A Broad Exploration of Coupled
Nonlinear Dynamics in Microbial Systems Motivated by Chemostat Experiments
Producing Deterministic Chaos." LBNL Report LBNL-2001172.
https://escholarship.org/uc/item/9wr5396s
"""

from dataclasses import dataclass, asdict
import numpy as np
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Parameters (Table 1.1 of the report).  Rates are per hour, concentrations mg/cc.
# ---------------------------------------------------------------------------
@dataclass
class Params:
    n0: float = 0.15          # inflowing nutrient concentration (mg/cc)

    mu_rn: float = 0.1873     # max specific growth rate, rods on nutrient (1/hr)
    Y_rn: float = 0.4         # yield, rods on nutrient
    K_rn: float = 0.009       # half-saturation, rods on nutrient (mg/cc)

    mu_cn: float = 0.1248     # max specific growth rate, cocci on nutrient (1/hr)
    Y_cn: float = 0.4
    K_cn: float = 0.009

    mu_pr: float = 0.05117    # max specific growth rate, predator on rods (1/hr)
    Y_pr: float = 0.6
    K_pr: float = 0.009

    mu_pc: float = 0.05117    # max specific growth rate, predator on cocci (1/hr)
    Y_pc: float = 0.6
    K_pc: float = 0.009

    delta_p: float = 0.00416  # predator intrinsic death rate (1/hr)

    m_r: float = 1.6e-9       # mean rod mass    (mg)
    m_c: float = 8.2e-9       # mean coccus mass (mg)
    m_p: float = 3.2e-6       # mean predator mass (mg)

    EF: float = 0.5           # efficiency of dead-predator recycling to nutrient
    m1: float = 1.579e-6      # slope of predator rod-preference change (cc/cell)
    i1: float = 0.8421        # intercept of predator rod-preference change
    m2: float = 6.6e-9        # extra half-saturation term for the preference change


# Canonical dilution rates (1/hr) that select qualitatively different regimes.
# Note: the chaotic band is threaded with narrow periodic windows, so exactly
# D = 0.0208 (the value used by Becks et al.) lands on a period-N cycle here;
# neighbouring values such as 0.0207 sit firmly on the strange attractor.
D_CHAOS = 0.0207          # deterministic chaos, all three microbes coexist
D_COCCI_DIE = 0.01875     # lower dilution -> cocci wash out, steady state
D_RODS_DIE = 0.0375       # higher dilution -> rods wash out, steady state

# Initial condition used throughout (Section 1.3 of the report).
Y0 = np.array([0.03, 4.2e6, 1.0e6, 3000.0])


# ---------------------------------------------------------------------------
# Right-hand side of the ODE system
# ---------------------------------------------------------------------------
def rhs(t, y, D, P: Params):
    """dy/dt for the four-component chemostat (Eq. 1.10)."""
    n, r, c, p = y
    # clamp tiny negative excursions from the stiff solver
    n = n if n > 0.0 else 0.0
    r = r if r > 0.0 else 0.0
    c = c if c > 0.0 else 0.0
    p = p if p > 0.0 else 0.0

    mr, mc, mp = P.m_r, P.m_c, P.m_p

    # Monod fractions for nutrient uptake
    fn_r = n / (P.K_rn + n)
    fn_c = n / (P.K_cn + n)

    # predator functional responses on rod / coccus biomass,
    # with the density-dependent rod preference folded in
    mu_pr_eff = P.mu_pr * (P.m1 * r + P.i1)
    f_r = (mr * r) / (P.K_pr + P.m2 * r + mr * r)
    f_c = (mc * c) / (P.K_pc + mc * c)

    dn = (D * (P.n0 - n)
          - (P.mu_rn / P.Y_rn) * fn_r * (mr * r)
          - (P.mu_cn / P.Y_cn) * fn_c * (mc * c)
          + P.EF * P.delta_p * (mp * p))            # recycling of dead predators

    dr = (P.mu_rn * fn_r * r
          - (mu_pr_eff / P.Y_pr) * f_r * (mp * p) / mr
          - D * r)

    dc = (P.mu_cn * fn_c * c
          - (P.mu_pc / P.Y_pc) * f_c * (mp * p) / mc
          - D * c)

    dp = (mu_pr_eff * f_r * p
          + P.mu_pc * f_c * p
          - D * p
          - P.delta_p * p)

    return [dn, dr, dc, dp]


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------
def simulate(D, P: Params = None, t_end=200_000, n_points=200_000,
             y0=None, discard_frac=0.0):
    """Integrate the system and return (t, Y) with Y shape (4, N).

    discard_frac trims the initial transient so what remains lies on the attractor.
    """
    if P is None:
        P = Params()
    if y0 is None:
        y0 = Y0.copy()
    t_eval = np.linspace(0, t_end, n_points)
    sol = solve_ivp(rhs, [0, t_end], y0, args=(D, P), method="LSODA",
                    t_eval=t_eval, rtol=1e-9,
                    atol=[1e-11, 1e-2, 1e-2, 1e-4], max_step=3.0)
    if discard_frac > 0:
        k = int(len(sol.t) * discard_frac)
        return sol.t[k:], sol.y[:, k:]
    return sol.t, sol.y


def survivors(Y, tail_frac=0.3, floor=1.0):
    """Return a dict flagging which microbes are still alive at the end."""
    tail = Y[:, int(Y.shape[1] * (1 - tail_frac)):]
    return {
        "rods":   tail[1].max() > floor,
        "cocci":  tail[2].max() > floor,
        "preds":  tail[3].max() > floor,
    }


# ---------------------------------------------------------------------------
# Largest Lyapunov exponent  (tangent-linear / variational method)
# ---------------------------------------------------------------------------
def _f(y, D, P):
    return np.asarray(rhs(0, y, D, P))


def _jacobian(y, D, P, eps=1e-6):
    """Numerical Jacobian, columns scaled to the very different variable magnitudes."""
    J = np.zeros((4, 4))
    scale = np.array([0.05, 1e6, 1e6, 3e3])
    for j in range(4):
        dy = np.zeros(4)
        dy[j] = eps * scale[j]
        J[:, j] = (_f(y + dy, D, P) - _f(y - dy, D, P)) / (2 * dy[j])
    return J


def _coupled(t, Y, D, P):
    y, v = Y[:4], Y[4:8]
    return np.concatenate([_f(y, D, P), _jacobian(y, D, P) @ v])


def largest_lyapunov(D, P: Params = None, t_transient=20_000,
                     t_run=150_000, dt=50.0, return_history=False):
    """Largest Lyapunov exponent (1/hr) by propagating one tangent vector and
    renormalising every dt hours.  Positive => chaos; ~0 => limit cycle;
    negative => fixed point."""
    if P is None:
        P = Params()
    # settle onto the attractor
    s = solve_ivp(lambda t, y: _f(y, D, P), [0, t_transient], Y0.copy(),
                  method="LSODA", rtol=1e-10,
                  atol=[1e-12, 1e-3, 1e-3, 1e-5], max_step=3.0)
    y = s.y[:, -1].copy()
    v = np.array([1.0, 0.0, 0.0, 0.0])

    n_steps = int(t_run / dt)
    acc, t, history = 0.0, 0.0, []
    for k in range(n_steps):
        s = solve_ivp(_coupled, [t, t + dt], np.concatenate([y, v]),
                      args=(D, P), method="LSODA", rtol=1e-10, atol=1e-9,
                      max_step=3.0)
        y = s.y[:4, -1]
        v = s.y[4:8, -1]
        norm = np.linalg.norm(v)
        acc += np.log(norm)
        v = v / norm
        t += dt
        history.append(acc / t)
    lam = acc / (n_steps * dt)
    return (lam, np.array(history)) if return_history else lam


# ---------------------------------------------------------------------------
# 0-1 test for chaos (Gottwald & Melbourne)
# ---------------------------------------------------------------------------
def zero_one_test(series, n_c=100, subsample=15, seed=0):
    """Median K over random frequencies.  K ~ 1 => chaos, K ~ 0 => regular.
    `series` should be a scalar observable sampled along the trajectory."""
    rng = np.random.default_rng(seed)
    phi = series[::subsample].astype(float)
    phi = (phi - phi.mean()) / (phi.std() + 1e-30)
    N = len(phi)
    n_cut = N // 10
    idx = np.arange(1, N + 1)
    lags = np.arange(1, n_cut)
    Ks = []
    for _ in range(n_c):
        cc = rng.uniform(0.5, 2.5)
        pc = np.cumsum(phi * np.cos(cc * idx))
        qc = np.cumsum(phi * np.sin(cc * idx))
        M = np.array([np.mean((pc[l:] - pc[:-l]) ** 2 + (qc[l:] - qc[:-l]) ** 2)
                      for l in lags])
        Ks.append(np.corrcoef(lags, M)[0, 1])
    return float(np.median(Ks))


# ---------------------------------------------------------------------------
# Poincare section (used for the bifurcation diagram)
# ---------------------------------------------------------------------------
def poincare_r(D, P: Params = None, t_end=250_000, n_points=500_000,
               discard_frac=0.35):
    """Rod values sampled each time nutrient crosses its long-run mean upward.
    A limit cycle gives a few discrete values; chaos gives a filled interval."""
    t, Y = simulate(D, P, t_end=t_end, n_points=n_points,
                    discard_frac=discard_frac)
    n, r = Y[0], Y[1]
    if r.max() < 1.0:
        return np.array([])
    nbar = n.mean()
    crossings = np.where((n[:-1] < nbar) & (n[1:] >= nbar))[0]
    return r[crossings]
