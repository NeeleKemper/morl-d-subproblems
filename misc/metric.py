import numpy as np
from copy import deepcopy
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from typing import Callable, Union, Optional


def hypervolume(ref_point: np.ndarray, points: list[np.ndarray] | np.ndarray) -> float:
    return HV(ref_point=ref_point * -1)(np.array(points) * -1)


def igd(known_front: list[np.ndarray], current_estimate: list[np.ndarray]) -> float:
    ind = IGD(np.array(known_front))
    return ind(np.array(current_estimate))


def sparsity(front: list[np.ndarray] | np.ndarray) -> float:
    if len(front) < 2:
        return 0.0

    sparsity_value = 0.0
    m = len(front[0])
    front = np.array(front)
    for dim in range(m):
        objs_i = np.sort(deepcopy(front.T[dim]))
        for i in range(1, len(objs_i)):
            sparsity_value += np.square(objs_i[i] - objs_i[i - 1])
    sparsity_value /= len(front) - 1

    return sparsity_value


def expected_utility(front: list[np.ndarray] | np.ndarray, weights_set: list[np.ndarray],
                     utility: Callable = np.dot) -> float:
    maxs = []
    for weights in weights_set:
        scalarized_front = np.array([utility(weights, point) for point in front])
        maxs.append(np.max(scalarized_front))

    return np.mean(np.array(maxs), axis=0)


def cardinality(front: list[np.ndarray] | np.ndarray) -> float:
    return len(front)


def maximum_utility_loss(front: list[np.ndarray], reference_set: list[np.ndarray], weights_set: np.ndarray,
                         utility: Callable = np.dot) -> float:
    max_scalarized_values_ref = [np.max([utility(weight, point) for point in reference_set]) for weight in weights_set]
    max_scalarized_values = [np.max([utility(weight, point) for point in front]) for weight in weights_set]
    utility_losses = [max_scalarized_values_ref[i] - max_scalarized_values[i] for i in
                      range(len(max_scalarized_values))]
    return np.max(utility_losses)


def get_non_pareto_dominated_inds(
        candidates: Union[np.ndarray, list],
        remove_duplicates: bool = True,
        maximize: bool = True,
) -> np.ndarray:
    """
    Return a boolean mask of nondominated rows (Pareto front) for a maximization problem.

    A point j dominates point i (for maximization) iff:
      - x_j >= x_i  elementwise, and
      - x_j >  x_i  in at least one objective.
    """
    P = np.asarray(candidates, dtype=float)
    if P.ndim != 2:
        raise ValueError('candidates must be a 2D array [n_points, n_objectives].')
    n = P.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    if n == 1:
        return np.ones(1, dtype=bool)

    # Convert to maximization (if you had a minimization, you’d negate)
    A = P if maximize else -P

    # Pairwise comparisons: result shape [n, n]
    # ge[i, j] = True iff A[j] >= A[i] elementwise
    ge = (A[None, :, :] >= A[:, None, :]).all(axis=2)
    # gt[i, j] = True iff A[j] > A[i] in at least one objective
    gt = (A[None, :, :] > A[:, None, :]).any(axis=2)

    # j dominates i  <=>  ge[i, j] and gt[i, j]
    dominates = ge & gt  # [n, n], entry (i, j)
    dominated = dominates.any(axis=1)  # [n], True if some j dominates i
    keep = ~dominated  # nondominated mask

    if remove_duplicates:
        # Keep only first occurrence of each duplicate point
        _, first_idx = np.unique(P, axis=0, return_index=True)
        dup_keep = np.zeros(n, dtype=bool)
        dup_keep[first_idx] = True
        keep &= dup_keep

    return keep


def filter_pareto_dominated(candidates: Union[np.ndarray, list], remove_duplicates: bool = True, maximize: bool = True
                            ) -> np.ndarray:
    P = np.asarray(candidates)
    if P.ndim != 2 or len(P) < 2:
        return P
    mask = get_non_pareto_dominated_inds(P, remove_duplicates=remove_duplicates, maximize=maximize)
    return P[mask]
