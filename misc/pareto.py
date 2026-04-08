"""Pareto utilities."""

from copy import deepcopy
from typing import Union

import numpy as np
from scipy.spatial import ConvexHull

from misc.metric import filter_pareto_dominated


def get_non_pareto_dominated_inds(candidates: Union[np.ndarray, list], remove_duplicates: bool = True) -> np.ndarray:
    candidates = np.array(candidates)
    uniques, indcs, invs, counts = np.unique(candidates, return_index=True, return_inverse=True, return_counts=True,
                                             axis=0)

    res_eq = np.all(candidates[:, None, None] <= candidates, axis=-1).squeeze()
    res_g = np.all(candidates[:, None, None] < candidates, axis=-1).squeeze()
    c1 = np.sum(res_eq, axis=-1) == counts[invs]
    c2 = np.any(~res_g, axis=-1)
    if remove_duplicates:
        to_keep = np.zeros(len(candidates), dtype=bool)
        to_keep[indcs] = 1
    else:
        to_keep = np.ones(len(candidates), dtype=bool)

    return np.logical_and(c1, c2) & to_keep


def filter_convex_dominated(candidates: Union[np.ndarray, list]) -> np.ndarray:
    candidates = np.array(candidates)
    if len(candidates) > 2:
        hull = ConvexHull(candidates)
        ccs = candidates[hull.vertices]
    else:
        ccs = candidates
    return filter_pareto_dominated(ccs)


def get_non_dominated(candidates: set) -> set:
    candidates = np.array(list(candidates))
    candidates = candidates[candidates.sum(1).argsort()[::-1]]
    for i in range(candidates.shape[0]):
        n = candidates.shape[0]
        if i >= n:
            break
        non_dominated = np.ones(candidates.shape[0], dtype=bool)
        non_dominated[i + 1:] = np.any(candidates[i + 1:] > candidates[i], axis=1)
        candidates = candidates[non_dominated]

    non_dominated = set()
    for candidate in candidates:
        non_dominated.add(tuple(candidate))

    return non_dominated


def get_non_dominated_inds(solutions: np.ndarray) -> np.ndarray:
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(solutions[is_efficient] > c, axis=1)
            is_efficient[i] = 1
    return is_efficient


def calculate_crowding_distance(evals: list[np.ndarray], extreme: bool = True) -> np.ndarray:
    if len(evals) == 0:
        return np.array([])
    arr = np.vstack(evals)
    n, m = arr.shape
    distances = np.zeros(n, dtype=float)

    for obj in range(m):
        vals = arr[:, obj]
        order = np.argsort(vals)
        norm = vals[order[-1]] - vals[order[0]] or 1.0
        for i in range(1, n - 1):
            prev_v = vals[order[i - 1]]
            next_v = vals[order[i + 1]]
            distances[order[i]] += (next_v - prev_v) / norm
        if extreme and n > 1:
            distances[order[0]] = np.inf
            distances[order[-1]] = np.inf

    return distances


class ParetoArchive:
    def __init__(self, convex_hull: bool = False, max_size: int = None):
        self.convex_hull = convex_hull
        self.max_size = max_size
        self.individuals: list = []
        self.evaluations: list[np.ndarray] = []
        self.z_star = None
        self.z_nadir = None
        self._selected_hist: set[tuple] = set()

    def _truncate_by_crowding_distance(self) -> None:
        """Trim archive in-place to max_size, keeping the most spread-out solutions."""
        dists = calculate_crowding_distance(self.evaluations, extreme=True)
        keep_idx = np.argsort(-dists)[:self.max_size]
        self.evaluations = [self.evaluations[i] for i in keep_idx]
        self.individuals = [self.individuals[i] for i in keep_idx]

    def add(self, candidate, evaluation):
        self.evaluations.append(evaluation)
        self.individuals.append(deepcopy(candidate))

        if self.convex_hull:
            nd_candidates = {tuple(x) for x in filter_convex_dominated(self.evaluations)}
        else:
            nd_candidates = {tuple(x) for x in filter_pareto_dominated(self.evaluations)}

        non_dominated_evals = []
        non_dominated_evals_tuples = []
        non_dominated_individuals = []
        for e, i in zip(self.evaluations, self.individuals):
            if tuple(e) in nd_candidates and tuple(e) not in non_dominated_evals_tuples:
                non_dominated_evals.append(e)
                non_dominated_evals_tuples.append(tuple(e))
                non_dominated_individuals.append(i)
        self.evaluations = non_dominated_evals
        self.individuals = non_dominated_individuals

        if self.max_size is not None and len(self.evaluations) > self.max_size:
            self._truncate_by_crowding_distance()

    def filter_by_crowding_distance(self, num_select_solutions: int) -> list:
        """Return a diverse subset of individuals without modifying the archive."""
        if not self.evaluations:
            return []
        dists = calculate_crowding_distance(self.evaluations, extreme=True)
        select_idx = np.argsort(-dists)[:num_select_solutions]
        return [deepcopy(self.individuals[i]) for i in select_idx]
