"""Conditional logit over the candidates of each target, with a fixed offset and a ridge penalty per parameter.

Rows are (target, candidate) pairs: group[r] is the target, y[r] = 1 for the actor (exactly one per fitted target),
offset[r] the fixed score, X the sparse design. Score s = offset + X theta; the negative log-likelihood is
  sum_g [logsumexp_{r in g} s_r - s_{actor(g)}] + 1/2 sum_j lam_j theta_j^2.
Fitted with scipy's trust-region Newton-CG on exact gradients and Hessian-vector products, followed by full Newton
steps; a fit whose remaining Newton step exceeds the tolerance raises. Targets without an actor among their rows
(unreachable) must be removed beforehand.
"""
import numpy as np
from scipy import optimize, sparse
from scipy.sparse import linalg as splinalg


class Problem:
    def __init__(self, X, offset, y, group, lam):
        self.X = sparse.csr_matrix(X, dtype=np.float64)
        self.XT = self.X.T.tocsr()
        self.offset = np.asarray(offset, np.float64)
        self.y = np.asarray(y, np.float64)
        self.group = np.asarray(group, np.int64)
        self.G = int(self.group.max()) + 1
        self.lam = np.broadcast_to(np.asarray(lam, np.float64), (self.X.shape[1],)).copy()
        per_group = np.bincount(self.group, self.y, self.G)
        if not np.all(per_group == 1):
            raise ValueError("every target needs exactly one actor row")
        self._cache = (None, None)

    def scores(self, theta):
        return self.offset + self.X @ theta

    def probs(self, s):
        m = np.full(self.G, -np.inf)
        np.maximum.at(m, self.group, s)
        e = np.exp(s - m[self.group])
        z = np.bincount(self.group, e, self.G)
        return e / z[self.group], m + np.log(z)

    def _state(self, theta):
        if self._cache[0] is not None and np.array_equal(self._cache[0], theta):
            return self._cache[1]
        s = self.scores(theta)
        p, lse = self.probs(s)
        self._cache = (theta.copy(), (s, p, lse))
        return s, p, lse

    def fun(self, theta):
        s, p, lse = self._state(theta)
        return float(lse.sum() - (s * self.y).sum() + 0.5 * (self.lam * theta * theta).sum())

    def grad(self, theta):
        s, p, lse = self._state(theta)
        return self.XT @ (p - self.y) + self.lam * theta

    def hessp(self, theta, v):
        s, p, lse = self._state(theta)
        q = self.X @ v
        mean_q = np.bincount(self.group, p * q, self.G)
        return self.XT @ (p * (q - mean_q[self.group])) + self.lam * v

    def loglik_per_target(self, theta):
        """Log-likelihood of the actor in each target (length G)."""
        s, p, lse = self._state(theta)
        return np.bincount(self.group, s * self.y, self.G) - lse


def newton_step(problem, theta, maxiter=5000):
    """Remaining Newton step -H^-1 g at theta, by conjugate gradients on exact Hessian-vector products."""
    n = problem.X.shape[1]
    H = splinalg.LinearOperator((n, n), matvec=lambda v: problem.hessp(theta, v), dtype=np.float64)
    step, flag = splinalg.cg(H, -problem.grad(theta), rtol=1e-10, atol=0.0, maxiter=maxiter)
    if flag != 0:
        raise RuntimeError(f"conjugate gradients for the remaining Newton step did not converge (flag {flag})")
    return step


def fit(problem, theta0=None, step_tol=1e-6, maxiter=500, newton_steps=3):
    """Penalised maximum likelihood: trust-region Newton-CG, then up to newton_steps full Newton steps from its solution.
    Convergence is defined by the remaining Newton step: the fit raises unless every component of -H^-1 g at the
    solution is at most step_tol in absolute value. (An absolute gradient threshold does not scale with the number of
    targets: the gradient is a sum over targets and stops shrinking at the floating-point precision of the objective.
    For the same reason trust-ncg can stop a few 1e-6 from the optimum, when the change of the objective falls below
    its precision; the full Newton steps use gradients and Hessian-vector products only, and each must reduce the
    gradient norm.)"""
    theta0 = np.zeros(problem.X.shape[1]) if theta0 is None else np.asarray(theta0, np.float64)
    res = optimize.minimize(problem.fun, theta0, jac=problem.grad, hessp=problem.hessp, method="trust-ncg",
                            options={"gtol": 1e-12, "maxiter": maxiter})
    theta = res.x
    step = newton_step(problem, theta)
    taken = 0
    while np.abs(step).max() > step_tol and taken < newton_steps:
        norm = np.linalg.norm(problem.grad(theta))
        theta = theta + step
        if not np.linalg.norm(problem.grad(theta)) < norm:
            raise RuntimeError("a full Newton step did not reduce the gradient norm")
        step = newton_step(problem, theta)
        taken += 1
    largest = float(np.abs(step).max())
    if largest > step_tol:
        raise RuntimeError(f"conditional logit did not converge: {res.message}; remaining Newton step {largest:.3g}")
    return theta, {"nll": float(problem.fun(theta)), "iterations": int(res.nit), "newton_steps": taken,
                   "remaining_step": largest, "gradient_norm": float(np.linalg.norm(problem.grad(theta))),
                   "message": str(res.message)}


def softmax_loglik(scores, group, y):
    """Per-target log-likelihood of the actor for given scores (rows of fitted targets only)."""
    G = int(group.max()) + 1
    m = np.full(G, -np.inf)
    np.maximum.at(m, group, scores)
    lse = m + np.log(np.bincount(group, np.exp(scores - m[group]), G))
    return np.bincount(group, scores * y, G) - lse
