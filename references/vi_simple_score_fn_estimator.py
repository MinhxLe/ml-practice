import torch
import torch.nn as nn
import torch.distributions as dist


class BetaVariational(nn.Module):
    def __init__(self):
        super().__init__()
        # Parameters for Beta distribution (alpha, beta)
        # Using softplus to ensure they're positive
        self.log_alpha = nn.Parameter(torch.tensor(1.0))
        self.log_beta = nn.Parameter(torch.tensor(1.0))

    def forward(self):
        # Return Beta distribution with learned parameters
        alpha = torch.exp(self.log_alpha)
        beta = torch.exp(self.log_beta)
        return dist.Beta(alpha, beta)


# Generate Bernoulli samples with p=0.7
def generate_target_samples(n_samples=1000):
    dist_true = dist.Bernoulli(probs=0.7)
    return dist_true.sample((n_samples,))


def train_vi_reinforce(target_samples, n_iterations=2000, n_samples=10):
    q = BetaVariational()
    optimizer = torch.optim.Adam(q.parameters(), lr=0.01)

    losses = []
    for i in range(n_iterations):
        optimizer.zero_grad()

        total_loss = 0
        for _ in range(n_samples):
            # Sample p from Beta distribution
            q_dist = q()
            p = q_dist.sample()

            # Log probability under variational distribution
            log_q = q_dist.log_prob(p)

            # Log likelihood of the data given p
            bernoulli = dist.Bernoulli(probs=p)
            log_likelihood = bernoulli.log_prob(target_samples).sum()

            # Add prior (assume Beta(1,1) prior, which is uniform)
            prior = dist.Beta(1.0, 1.0)
            log_prior = prior.log_prob(p)

            # f(z) = log p(x|z) + log p(z) - log q(z)
            f_z = (log_likelihood + log_prior - log_q).detach()

            # REINFORCE estimator
            loss = -f_z * log_q

            total_loss += loss

        avg_loss = total_loss / n_samples
        avg_loss.backward()
        optimizer.step()

        losses.append(avg_loss.item())

        if i % 200 == 0:
            with torch.no_grad():
                q_dist = q()
                alpha = torch.exp(q.log_alpha)
                beta = torch.exp(q.log_beta)
                mean = alpha / (alpha + beta)
                print(
                    f"Iteration {i}, Loss: {avg_loss.item():.3f}, "
                    f"Mean: {mean.item():.3f}, "
                    f"Alpha: {alpha.item():.3f}, Beta: {beta.item():.3f}"
                )

    return q, losses


# Generate data and train
target_samples = generate_target_samples(1000)
q_trained, losses = train_vi_reinforce(target_samples)

# Print final distribution parameters
with torch.no_grad():
    q_dist = q_trained()
    alpha = torch.exp(q_trained.log_alpha)
    beta = torch.exp(q_trained.log_beta)
    mean = alpha / (alpha + beta)
    variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    print(f"\nFinal Parameters:")
    print(f"Mean: {mean.item():.3f}")
    print(f"Variance: {variance.item():.3f}")
    print(f"Alpha: {alpha.item():.3f}")
    print(f"Beta: {beta.item():.3f}")
