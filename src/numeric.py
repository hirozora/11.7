"""Likelihood, ridge fitting, resampling and match indexing shared by the analyses."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit


Z = 9
STEPS_L = 5

def newton(base, y, group, n_groups, lam, iters=1000):
    """Ridge-penalised logistic offsets per group: maximise sum loglik(base + theta[group]) - lam/2 * theta^2.
    Steps are clipped to one logit unit: with extreme offsets an unclipped Newton step overshoots and oscillates."""
    theta = np.zeros(n_groups)
    for _ in range(iters):
        p = expit(base + theta[group])
        g = np.bincount(group, y - p, n_groups) - lam * theta
        h = np.bincount(group, p * (1 - p), n_groups) + lam
        step = np.clip(g / h, -1.0, 1.0)
        theta += step
        if np.abs(step).max() < 1e-9:
            return theta
    raise RuntimeError(f"Newton did not converge: max step {np.abs(step).max():.2e}")


def newton_slope(base, y, x, idx, n, lam, iters=1000):
    b = np.zeros(n)
    for _ in range(iters):
        p = expit(base + b[idx] * x)
        gr = np.bincount(idx, x * (y - p), n) - lam * b
        h = np.bincount(idx, x * x * p * (1 - p), n) + lam
        step = np.clip(gr / h, -1.0, 1.0)
        b += step
        if np.abs(step).max() < 1e-9:
            return b
    raise RuntimeError("Newton did not converge")


def holm(pvals, alpha):
    order = np.argsort(pvals)
    m = len(pvals)
    reject = np.zeros(m, bool)
    for rank, i in enumerate(order):
        if pvals[i] <= alpha / (m - rank):
            reject[i] = True
        else:
            break
    return reject


def pop_offset(y, group, zone, level, fit, a):
    G, L = int(group.max()) + 1, int(level.max()) + 1
    cell = (group * Z + zone) * L + level
    num = np.bincount(cell[fit], y[fit], G * Z * L).reshape(G, Z, L)
    den = np.bincount(cell[fit], None, G * Z * L).reshape(G, Z, L).astype(float)
    p_g = (num.sum((1, 2)) + 1.0) / (den.sum((1, 2)) + 2.0)
    p_gz = (num.sum(2) + a * p_g[:, None]) / (den.sum(2) + a)
    p_gzl = (num + a * p_gz[:, :, None]) / (den + a)
    return logit(p_gzl[group, zone, level])


def loglik(eta, y):
    return y * eta - np.logaddexp(0.0, eta)


def team_match_index(A):
    tm = A.drop_duplicates(["match_id", "team_id"]).sort_values(["team_id", "kickoff"])[["match_id", "team_id"]].copy()
    tm["k"] = tm.groupby("team_id").cumcount() + 1
    return A.merge(tm, on=["match_id", "team_id"], how="left")["k"].to_numpy()


def boot_ci(diff, match, rng, n):
    s = pd.Series(diff).groupby(match).agg(["sum", "size"])
    v, c = s["sum"].to_numpy(), s["size"].to_numpy()
    b = rng.integers(0, len(v), size=(n, len(v)))
    dist = 1000.0 * v[b].sum(1) / c[b].sum(1)
    return [float(np.quantile(dist, 0.025)), float(np.quantile(dist, 0.975))]


def cv_style(base, y, idx, n, fit, match, kick, grid):
    fm = sorted(np.unique(match[fit]), key=kick.get)
    fold = pd.Series(match).map({m: i % 2 for i, m in enumerate(fm)}).fillna(-1).to_numpy()
    best, best_ll = None, -np.inf
    for al in grid:
        ll = 0.0
        for f in (0, 1):
            tr, te = fit & (fold == 1 - f), fit & (fold == f)
            b = newton(base[tr], y[tr], idx[tr], n, al)
            ll += loglik(base[te] + b[idx[te]], y[te]).sum()
        if ll > best_ll:
            best, best_ll = al, ll
    return base + newton(base[fit], y[fit], idx[fit], n, best)[idx]
