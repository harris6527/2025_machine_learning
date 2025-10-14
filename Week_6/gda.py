import numpy as np


class GDA:
    """
    Gaussian Discriminant Analysis (binary).

    - If shared_cov=True, uses a shared covariance across classes (LDA boundary).
    - If shared_cov=False, uses class-specific covariances (QDA boundary).

    Implements from scratch without external ML libraries.
    """

    def __init__(self, shared_cov: bool = True, reg_eps: float = 1e-6):
        self.shared_cov = shared_cov
        self.reg_eps = reg_eps
        # learned params
        self.phi_ = None  # P(y=1)
        self.mu0_ = None
        self.mu1_ = None
        self.sigma_ = None  # shared cov if shared_cov=True
        self.sigma0_ = None
        self.sigma1_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        assert X.ndim == 2 and y.ndim == 1 and X.shape[0] == y.shape[0]
        n = X.shape[0]

        # class priors and means
        self.phi_ = y.mean()
        X0 = X[y == 0]
        X1 = X[y == 1]
        if X0.size == 0 or X1.size == 0:
            raise ValueError("Both classes must be present in training data.")
        self.mu0_ = X0.mean(axis=0)
        self.mu1_ = X1.mean(axis=0)

        # covariance(s)
        if self.shared_cov:
            # shared covariance: sum over both classes of (x - mu_y)(x - mu_y)^T / n
            diff0 = X0 - self.mu0_
            diff1 = X1 - self.mu1_
            sigma = (diff0.T @ diff0 + diff1.T @ diff1) / n
            # regularize slightly for numerical stability
            sigma += self.reg_eps * np.eye(X.shape[1])
            self.sigma_ = sigma
            self.sigma0_ = None
            self.sigma1_ = None
        else:
            # class-specific covariances (QDA)
            diff0 = X0 - self.mu0_
            diff1 = X1 - self.mu1_
            s0 = (diff0.T @ diff0) / max(1, (X0.shape[0] - 1))
            s1 = (diff1.T @ diff1) / max(1, (X1.shape[0] - 1))
            s0 += self.reg_eps * np.eye(X.shape[1])
            s1 += self.reg_eps * np.eye(X.shape[1])
            self.sigma0_ = s0
            self.sigma1_ = s1
            self.sigma_ = None
        return self

    def _log_gaussian(self, X, mu, Sigma):
        # log N(x | mu, Sigma)
        d = X.shape[1]
        # Cholesky for stability if possible; fallback to inv/det
        try:
            L = np.linalg.cholesky(Sigma)
            solve = np.linalg.solve(L, (X - mu).T)
            quad = np.sum(solve**2, axis=0)
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
        except np.linalg.LinAlgError:
            iSigma = np.linalg.inv(Sigma)
            diff = X - mu
            quad = np.einsum('ij,ij->i', diff @ iSigma, diff)
            logdet = np.log(np.linalg.det(Sigma) + 1e-30)
        return -0.5 * (quad + logdet + d * np.log(2.0 * np.pi))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.shared_cov:
            # linear discriminant: compare log-likelihood + log prior
            logp0 = self._log_gaussian(X, self.mu0_, self.sigma_) + np.log(1 - self.phi_ + 1e-15)
            logp1 = self._log_gaussian(X, self.mu1_, self.sigma_) + np.log(self.phi_ + 1e-15)
        else:
            logp0 = self._log_gaussian(X, self.mu0_, self.sigma0_) + np.log(1 - self.phi_ + 1e-15)
            logp1 = self._log_gaussian(X, self.mu1_, self.sigma1_) + np.log(self.phi_ + 1e-15)

        # log-sum-exp for stability
        m = np.maximum(logp0, logp1)
        p1 = np.exp(logp1 - m) / (np.exp(logp0 - m) + np.exp(logp1 - m))
        return np.vstack([1 - p1, p1]).T  # shape (n, 2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

