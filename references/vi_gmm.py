import torch
import torch.nn as nn
import torch.distributions as dist


class VariationalGMM(nn.Module):
    def __init__(self, n_components, n_features, alpha_prior=1.0):
        super().__init__()
        self.n_components = n_components
        self.n_features = n_features
        self.alpha_prior = alpha_prior

        # Variational parameters for mixture weights (Dirichlet)
        self.alpha = nn.Parameter(torch.zeros(n_components) * alpha_prior)

        # Variational parameters for component means (Normal)
        self.mu_loc = nn.Parameter(torch.randn(n_components, n_features))
        self.mu_scale = nn.Parameter(torch.ones(n_components, n_features))

    def forward(self, x):
        batch_size = x.shape[0]

        # Compute expected values under q(mu) and q(lambda)
        E_mu = self.mu_loc
        E_lambda = self.lambda_scale.unsqueeze(-1)

        # Compute responsibilities (variational distribution q(z))
        log_resp = self._compute_log_responsibilities(x, E_mu, E_lambda)
        resp = torch.softmax(log_resp, dim=-1)

        # Compute ELBO terms
        kl_z = self._kl_categorical(resp)
        kl_pi = self._kl_dirichlet()
        kl_mu_lambda = self._kl_normal_wishart()

        # Expected log-likelihood
        expected_log_likelihood = self._expected_log_likelihood(x, resp, E_mu, E_lambda)

        # ELBO = E[log p(x,z,θ)] - E[log q(z,θ)]
        elbo = expected_log_likelihood - (kl_z + kl_pi + kl_mu_lambda)

        return -elbo  # Return negative ELBO for minimization

    def _compute_log_responsibilities(self, x, E_mu, E_lambda):
        # x: [batch_size, n_features]
        # E_mu: [n_components, n_features]
        # E_lambda: [n_components, n_features]

        # Compute log p(x|z,μ,Λ) for each component
        log_prob = torch.zeros(x.shape[0], self.n_components)

        for k in range(self.n_components):
            # Using expected sufficient statistics
            precision = torch.diag(E_lambda[k])
            dist_k = dist.MultivariateNormal(loc=E_mu[k], precision_matrix=precision)
            log_prob[:, k] = dist_k.log_prob(x)

        # Add log mixture weights
        log_mix_weights = torch.digamma(self.alpha) - torch.digamma(self.alpha.sum())
        return log_prob + log_mix_weights

    def _kl_categorical(self, resp):
        # KL[q(z) || p(z|π)]
        log_mix_weights = torch.digamma(self.alpha) - torch.digamma(self.alpha.sum())
        return torch.sum(resp * (torch.log(resp + 1e-10) - log_mix_weights))

    def _kl_dirichlet(self):
        # KL[q(π) || p(π)]
        alpha_0 = self.alpha_prior * torch.ones_like(self.alpha)
        return dist.kl.kl_divergence(
            dist.Dirichlet(self.alpha), dist.Dirichlet(alpha_0)
        )

    def _kl_normal_wishart(self):
        # Simplified KL divergence for Normal-Wishart
        # This is a placeholder - implement full KL divergence here
        return torch.tensor(0.0)

    def _expected_log_likelihood(self, x, resp, E_mu, E_lambda):
        # E_q[log p(x|z,μ,Λ)]
        log_prob = torch.zeros(x.shape[0], self.n_components)

        for k in range(self.n_components):
            precision = torch.diag(E_lambda[k])
            dist_k = dist.MultivariateNormal(loc=E_mu[k], precision_matrix=precision)
            log_prob[:, k] = dist_k.log_prob(x)

        return torch.sum(resp * log_prob)


# Training example
def train_vi_gmm(model, data, n_iterations=1000, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for iter in range(n_iterations):
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()

        if iter % 100 == 0:
            print(f"Iteration {iter}, ELBO: {-loss.item():.2f}")


# Usage example
if __name__ == "__main__":
    # Generate some synthetic data
    n_samples = 10_000
    n_features = 2
    n_components = 3

    # Create synthetic mixture of Gaussians
    true_means = torch.tensor([[-2, -2], [0, 0], [2, 2]], dtype=torch.float32)
    true_covs = torch.stack([torch.eye(2)] * 3) * 0.5
    weights = torch.tensor([0.5, 0.2, 0.3])

    # Generate data
    data = []
    for i in range(n_samples):
        k = torch.multinomial(weights, 1).item()
        sample = dist.MultivariateNormal(true_means[k], true_covs[k]).sample()
        data.append(sample)

    data = torch.stack(data)

    # Create and train model
    model = VariationalGMM(n_components=3, n_features=2)
    train_vi_gmm(model, data)
