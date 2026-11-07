"""Inverse-variance pooling, paired permutation pooling and stratified match bootstrap."""
import numpy as np
from scipy.stats import chi2


def glm_pool(recs, ref):
    ref = np.asarray(ref, float)
    betas = [np.asarray(r["beta"]) for r in recs]
    precs = [np.linalg.inv(np.asarray(r["cov"])) for r in recs]
    P = sum(precs)
    cov = np.linalg.inv(P)
    beta = cov @ sum(p @ b for p, b in zip(precs, betas))
    wald = float(beta @ P @ beta)
    df = len(beta)
    out = {"names": recs[0]["names"], "beta": beta.tolist(), "se": np.sqrt(np.diag(cov)).tolist(), "wald": wald, "df": df,
           "p": float(chi2.sf(wald, df)), "direction_ok": bool(beta @ ref > 0),
           "leagues": [{"beta": b.tolist(), "wald": r["wald"], "p": r["p"], "agrees": bool(b @ ref > 0)} for b, r in zip(betas, recs)]}
    if df == 1:
        se = float(np.sqrt(cov[0, 0]))
        out["or"], out["or_ci"] = float(np.exp(beta[0])), [float(np.exp(beta[0] - 1.96 * se)), float(np.exp(beta[0] + 1.96 * se))]
    out["consistent"] = sum(lg["agrees"] for lg in out["leagues"])
    return out


def vc_pool(vcs, perms):
    z, zb, num, den, leagues = [], [], 0.0, 0.0, []
    for v, perm in zip(vcs, perms):
        mean, sd = perm.mean(), perm.std(ddof=1)
        z.append((v["T"] - mean) / sd)
        zb.append((perm - mean) / sd)
        num, den = num + v["T"] - mean, den + v["V"]
        leagues.append({"tau": float(np.sqrt(max(0.0, (v["T"] - mean) / v["V"]))), "z": float(z[-1]),
                        "p": float((1 + (perm >= v["T"]).sum()) / (1 + len(perm)))})
    if len({len(p) for p in perms}) != 1:
        raise ValueError("leagues have different numbers of permutations")
    Z = float(np.sum(z) / np.sqrt(len(z)))
    Zb = np.sum(zb, axis=0) / np.sqrt(len(z))
    return {"Z": Z, "p": float((1 + (Zb >= Z).sum()) / (1 + len(Zb))), "tau": float(np.sqrt(max(0.0, num / den))),
            "direction_ok": Z > 0, "leagues": leagues, "consistent": int(sum(lg["z"] > 0 for lg in leagues))}


def forward_pool(ds, B, rng):
    """ds: per league {"sum", "n"} per match. Match bootstrap within league."""
    boot_s, boot_n, leagues = 0.0, 0.0, []
    for d in ds:
        idx = rng.integers(0, len(d["sum"]), (B, len(d["sum"])))
        s, n = d["sum"][idx].sum(1), d["n"][idx].sum(1)
        boot_s, boot_n = boot_s + s, boot_n + n
        g = 1000 * d["sum"].sum() / d["n"].sum()
        leagues.append({"gain": float(g), "ci": np.quantile(1000 * s / n, [0.025, 0.975]).tolist(), "agrees": bool(g > 0)})
    gain = 1000 * sum(d["sum"].sum() for d in ds) / sum(d["n"].sum() for d in ds)
    dist = 1000 * boot_s / boot_n
    return {"gain": float(gain), "ci": np.quantile(dist, [0.025, 0.975]).tolist(), "p": float((1 + (dist <= 0).sum()) / (1 + B)),
            "direction_ok": bool(gain > 0), "leagues": leagues, "consistent": int(sum(lg["agrees"] for lg in leagues))}


def ratio_of(t2_type, t2_none):
    """tau_type / tau_none with tau = sqrt(max(0, tau^2)); infinite when tau_none^2 <= 0 (counted as >= 1)."""
    return np.where(t2_none > 0, np.sqrt(np.maximum(t2_type, 0.0) / np.where(t2_none > 0, t2_none, 1.0)), np.inf)


def p1_profile(acts, opps):
    rate = np.asarray(acts) / np.asarray(opps)
    if (rate[:, [0, 4]] <= 0).any():
        raise ValueError("a fine role has no involvement at steps 0 or 6+")
    return np.log(rate[:, 4] / rate[:, 0])


def fmt_est(r):
    if r["kind"] == "glm":
        return f"OR {r['or']:.3f} [{r['or_ci'][0]:.3f}, {r['or_ci'][1]:.3f}]" if r["df"] == 1 else f"Wald {r['wald']:.1f} (df {r['df']})"
    if r["kind"] == "vc":
        return f"tau {r['tau']:.3f}, Z {r['Z']:.2f}"
    if r["kind"] == "forward":
        return f"gain {r['gain']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]"
    if r["kind"] == "iut":
        return (f"type {r['type_gain']['gain']:+.3f}, type - individual {r['type_minus_individual']['gain']:+.3f}, "
                f"ratio {r['ratio']:.2f} [{r['ratio_ci'][0]:.2f}, {r['ratio_ci'][1]:.2f}]")
    return f"rho {r['rho']:.2f}"
