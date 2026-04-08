import numpy as np
from pymoo.util.ref_dirs import get_reference_directions


def generate_layer_energy_weights(layers: list, reward_dim: int = 3) -> list[float]:
    weights = get_reference_directions('layer-energy', reward_dim, layers)
    return weights


def generate_das_dennis_weights(num_weights: int, reward_dim: int = 2) -> list[float]:
    weights = get_reference_directions('das-dennis', reward_dim, n_points=num_weights)
    return weights


def generate_dirichlet_weights(num_weights: int, reward_dim: int, alpha: float) -> list[float]:
    weights = np.random.dirichlet(alpha * np.ones(reward_dim), size=num_weights)
    return weights.tolist()
