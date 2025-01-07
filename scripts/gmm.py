"""
[TODO] this is not working
gaussian mixture model implementation

https://www.ritchievink.com/blog/2019/05/24/algorithm-breakdown-expectation-maximization/
"""

import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import stats


class Config:
    seed: int = 69
    # dataset configuration
    n_classes: int = 3
    n_dim = 2
    n_samples = 200

    # training configuration
    n_epoch = 100


@dataclass
class DatasetParam:
    weights: np.ndarray
    mus: np.ndarray
    covs: np.ndarray

    @property
    def n_classes(self):
        return len(self.mus)


def generate_dataset(
    params: DatasetParam,
    n_samples: int,
):
    """Generate a random categorical distribution with random weights per class

    Args:
        n_samples: Number of samples to generate

    Returns:
        Tuple of (samples, weights) where:
            samples: Array of shape (n_samples,) with category indices
            weights: Array of shape (n_classes,) with class probabilities
    """

    # Generate samples from categorical distribution
    categories = np.random.choice(params.n_classes, size=n_samples, p=params.weights)
    return np.stack(
        [
            np.random.multivariate_normal(params.mus[c], params.covs[c])
            for c in categories
        ],
        axis=0,
    )


np.random.seed(Config.seed)
params = DatasetParam(
    weights=np.array([0.4, 0.3, 0.3]),
    mus=np.array([[10, 10], [-10, -10], [7, -5]]),
    covs=np.array(
        [
            [
                [1, -0.75],
                [-0.75, 1],
            ],
            [
                [10, 9],
                [9, 10],
            ],
            [
                [10, 0],
                [0, 10],
            ],
        ]
    ),
)
x = generate_dataset(params, Config.n_samples)


def plot_experiment(dataset, mus, covs, grid_points=100, n_std=3, alpha=0.3):
    """
    Plot multiple 2D multivariate Gaussian distributions.

    Args:
        mus: numpy array of shape (C, 2) containing means for C Gaussians
        covs: numpy array of shape (C, 2, 2) containing covariance matrices
        grid_points: number of points per dimension for visualization grid
        n_std: number of standard deviations to include in plot
        alpha: transparency for filled contours
    """
    # Determine plot bounds
    std_devs = np.sqrt(np.diagonal(covs, axis1=1, axis2=2))  # Shape (C, 2)
    x_min = np.min(mus[:, 0] - n_std * std_devs[:, 0])
    x_max = np.max(mus[:, 0] + n_std * std_devs[:, 0])
    y_min = np.min(mus[:, 1] - n_std * std_devs[:, 1])
    y_max = np.max(mus[:, 1] + n_std * std_devs[:, 1])

    # Create grid
    x = np.linspace(x_min, x_max, grid_points)
    y = np.linspace(y_min, y_max, grid_points)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))  # Shape (grid_points, grid_points, 2)

    # Plot setup
    plt.figure(figsize=(10, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(mus)))

    # Plot each Gaussian
    for i, (mu, cov, color) in enumerate(zip(mus, covs, colors)):
        # Calculate PDF values
        rv = stats.multivariate_normal(mu, cov)
        Z = rv.pdf(pos)

        # Plot contours
        plt.contourf(X, Y, Z, levels=10, alpha=alpha, colors=[color])
        plt.contour(X, Y, Z, levels=5, colors=[color], alpha=0.8)

        # Plot mean point
        plt.plot(mu[0], mu[1], "o", color=color, markersize=10, label=f"Gaussian {i+1}")

        # Plot covariance ellipse
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))

        for n_sig in [1, 2]:
            ell = plt.matplotlib.patches.Ellipse(
                mu,
                2 * n_sig * np.sqrt(eigenvals[0]),
                2 * n_sig * np.sqrt(eigenvals[1]),
                angle=angle,
                fill=False,
                color=color,
                alpha=0.3,
            )
            plt.gca().add_patch(ell)

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("GMM")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.scatter(dataset[:, 0], dataset[:, 1])

    return plt.gcf()


# plt.scatter(x[:, 0], x[:, 1])
# plt.show()
def is_invertible_rank(matrix):
    matrix = np.array(matrix)
    return (
        matrix.shape[0] == matrix.shape[1]
        and np.linalg.matrix_rank(matrix) == matrix.shape[0]
    )


class EM:
    def __init__(self, n_classes: int):
        self.n_classes = n_classes

        # parameters
        self.z = None
        self.mus = None
        self.covs = None
        self.p_z = None

    def init_parameters(self, x):
        n_samples, n_dim = x.shape
        n_classes = self.n_classes
        self.z = np.ones((n_samples, n_classes)) * 1 / n_classes

        # Initialize priors uniformly
        self.p_z = np.ones(self.n_classes) / self.n_classes

        # Initialize means by randomly selecting points from dataset
        random_indices = np.random.choice(n_samples, self.n_classes, replace=False)
        self.mus = x[random_indices]

        # Initialize covariances as identity matrices
        self.covs = np.array([np.eye(n_dim) for _ in range(self.n_classes)])

    def expectation_step(self, x):
        # # for each datapoint, what is the expected posterior(?))
        z_unormalized = np.stack(
            [
                stats.multivariate_normal(self.mus[i], self.covs[i]).pdf(x)
                for i in range(self.n_classes)
            ],
            axis=1,
        )  # n_samples, n_classes
        z_unormalized = z_unormalized * self.p_z[None, :]  # n_samples, n_classes
        self.z = z_unormalized / (
            z_unormalized.sum(axis=1, keepdims=True)
        )  # n_samples, n_classes

    def maximization_step(self, x):
        n, n_dim = x.shape
        self.p_z = self.z.mean(axis=0)
        self.mus = np.einsum("nd,nc->cd", x, self.z) / (self.z.sum(axis=0)[:, None])
        # self.mus = np.einsum("nd,nc->cd", x, self.z) / (self.z.sum(axis=0)[:, None])
        # self.mus - np.einsum("cd,c->cd", self.mus, self.z.sum(axis=0)**-1)
        x_mu = x[None, :, :] - self.mus[:, None, :]  # n_classes, n_samples, n_dim
        x_mu_2 = np.einsum("cnd,cne->cnde", x_mu, x_mu)
        self.covs = (
            np.einsum("cnde,nc->cde", x_mu_2, self.z)
            / (self.z.sum(axis=0)[:, None, None])
        ) + (np.eye(n_dim) * 1e-10)  # n_classes, n_dim, n_dim

        # self.covs = np.einsum("cnde,nc->cde", x_mu_2, self.z)
        # self.covs = np.einsum("cde,c->cde", self.covs, self.z.sum(axis=0) ** -1)
        # # add a small amount to make sure it is still valid covariacne matrix
        # self.covs += np.eye(n_dim) * 1e-10  # n_classes, n_dim, n_dim

    def fit(self, x):
        self.init_parameters(x)
        for i in range(Config.n_epoch):
            print(f"step {i}")
            self.expectation_step(x)
            self.maximization_step(x)
            # TODO add convergence check


model = EM(Config.n_classes)
model.fit(x)
plot_experiment(x, model.mus, model.covs).show()
