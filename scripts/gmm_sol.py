import numpy as np
from scipy import stats


class GaussianMixtureModel:
    def __init__(self, n_components=2, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.priors = None
        self.means = None
        self.covariances = None

    def initialize(self, X):
        """Initialize model parameters randomly"""
        n_samples, n_features = X.shape

        # Initialize priors uniformly
        self.priors = np.ones(self.n_components) / self.n_components

        # Initialize means by randomly selecting points from dataset
        random_indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[random_indices]

        # Initialize covariances as identity matrices
        self.covariances = np.array(
            [np.eye(n_features) for _ in range(self.n_components)]
        )

    def e_step(self, X):
        """Expectation step: compute responsibilities"""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        # Calculate probability density for each component
        for k in range(self.n_components):
            responsibilities[:, k] = self.priors[k] * stats.multivariate_normal.pdf(
                X, mean=self.means[k], cov=self.covariances[k]
            )

        # Normalize responsibilities
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        return responsibilities

    def m_step(self, X, responsibilities):
        """Maximization step: update parameters"""
        n_samples = X.shape[0]

        # Calculate effective number of points for each component
        N_k = responsibilities.sum(axis=0)

        # Update priors
        self.priors = N_k / n_samples

        # Update means
        self.means = np.dot(responsibilities.T, X) / N_k[:, np.newaxis]

        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / N_k[k]

            # Add small value to diagonal for numerical stability
            self.covariances[k] += 1e-6 * np.eye(X.shape[1])

    def compute_log_likelihood(self, X):
        """Compute log likelihood of the data"""
        n_samples = X.shape[0]
        likelihood = np.zeros(n_samples)

        for k in range(self.n_components):
            likelihood += self.priors[k] * stats.multivariate_normal.pdf(
                X, mean=self.means[k], cov=self.covariances[k]
            )

        return np.sum(np.log(likelihood))

    def fit(self, X):
        """Fit the model to the data using EM algorithm"""
        # Convert input to numpy array if needed
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Initialize parameters
        self.initialize(X)

        # EM algorithm
        old_log_likelihood = -np.inf

        for iteration in range(self.max_iter):
            # E-step
            responsibilities = self.e_step(X)

            # M-step
            self.m_step(X, responsibilities)

            # Check convergence
            log_likelihood = self.compute_log_likelihood(X)
            if abs(log_likelihood - old_log_likelihood) < self.tol:
                break

            old_log_likelihood = log_likelihood

        return {
            "priors": self.priors,
            "means": self.means,
            "covariances": self.covariances,
            "log_likelihood": log_likelihood,
            "n_iterations": iteration + 1,
        }

    def predict(self, X):
        """Predict cluster labels for new data"""
        X = np.array(X)
        responsibilities = self.e_step(X)
        return np.argmax(responsibilities, axis=1)
