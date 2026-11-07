import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import optimize, sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cond_logit import Problem, fit, softmax_loglik


def synthetic(seed=0, G=400, P=12):
    """Targets with 3-11 candidates, a sparse one-hot plus numeric design and a known parameter vector."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(3, 12, G)
    group = np.repeat(np.arange(G), sizes)
    n = len(group)
    cols = rng.integers(0, P - 2, n)
    X = sparse.hstack([sparse.csr_matrix((np.ones(n), (np.arange(n), cols)), shape=(n, P - 2)),
                       sparse.csr_matrix(rng.normal(size=(n, 2)))]).tocsr()
    theta = rng.normal(0, 0.7, P)
    offset = rng.normal(0, 1, n)
    s = offset + X @ theta
    y = np.zeros(n)
    for g in range(G):
        rows = np.flatnonzero(group == g)
        p = np.exp(s[rows] - s[rows].max())
        y[rows[rng.choice(len(rows), p=p / p.sum())]] = 1
    return X, offset, y, group, theta


class ConditionalLogit(unittest.TestCase):
    def test_gradient_and_hessian_vector_product_match_finite_differences(self):
        X, offset, y, group, _ = synthetic()
        prob = Problem(X, offset, y, group, lam=np.linspace(0.5, 2.0, X.shape[1]))
        rng = np.random.default_rng(1)
        theta, v = rng.normal(size=X.shape[1]), rng.normal(size=X.shape[1])
        eps = 1e-6
        num = np.array([(prob.fun(theta + eps * e) - prob.fun(theta - eps * e)) / (2 * eps) for e in np.eye(len(theta))])
        np.testing.assert_allclose(prob.grad(theta), num, rtol=1e-5, atol=1e-6)
        hv = (prob.grad(theta + eps * v) - prob.grad(theta - eps * v)) / (2 * eps)
        np.testing.assert_allclose(prob.hessp(theta, v), hv, rtol=1e-5, atol=1e-6)

    def test_fit_agrees_with_an_independent_quasi_newton_solution(self):
        X, offset, y, group, _ = synthetic(seed=2)
        prob = Problem(X, offset, y, group, lam=1.0)
        theta, info = fit(prob, step_tol=1e-8)
        ref = optimize.minimize(prob.fun, np.zeros(X.shape[1]), jac=prob.grad, method="BFGS", options={"gtol": 1e-9})
        self.assertLess(np.linalg.norm(prob.grad(ref.x)), 1e-7)
        np.testing.assert_allclose(theta, ref.x, atol=1e-6)
        self.assertLess(abs(info["nll"] - ref.fun), 1e-9)

    def test_full_newton_steps_finish_a_stopped_trust_region_fit(self):
        X, offset, y, group, _ = synthetic(seed=5)
        prob = Problem(X, offset, y, group, lam=1.0)
        theta, _ = fit(prob, step_tol=1e-8)
        with self.assertRaises(RuntimeError):
            fit(prob, step_tol=1e-8, maxiter=1, newton_steps=0)
        early, info = fit(prob, step_tol=1e-8, maxiter=1, newton_steps=10)
        self.assertGreater(info["newton_steps"], 0)
        np.testing.assert_allclose(early, theta, atol=1e-7)

    def test_loglik_per_target_is_the_log_softmax_of_the_actor(self):
        X, offset, y, group, theta = synthetic(seed=3, G=50)
        prob = Problem(X, offset, y, group, lam=0.0)
        s = prob.scores(theta)
        ll = prob.loglik_per_target(theta)
        for g in range(50):
            rows = np.flatnonzero(group == g)
            ref = s[rows][y[rows] == 1][0] - np.log(np.exp(s[rows]).sum())
            self.assertAlmostEqual(ll[g], ref, places=10)
        np.testing.assert_allclose(softmax_loglik(s, group, y), ll)

    def test_targets_without_exactly_one_actor_are_refused(self):
        X, offset, y, group, _ = synthetic(seed=4, G=20)
        y = y.copy()
        y[group == 0] = 0
        with self.assertRaises(ValueError):
            Problem(X, offset, y, group, lam=1.0)


if __name__ == '__main__':
    unittest.main()
