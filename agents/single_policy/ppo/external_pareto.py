import numpy as np
from copy import deepcopy


def check_dominated(obj_batch: np.ndarray, obj: np.ndarray) -> bool:
    return (np.logical_and((obj_batch >= obj).all(axis=1), (obj_batch > obj).any(axis=1))).any()


# return sorted indices of nondominated objs
def get_ep_indices(obj_batch_input: np.ndarray) -> list:
    if len(obj_batch_input) == 0:
        return np.array([])
    obj_batch = np.array(obj_batch_input)

    # Remove duplicates: keep only the first occurrence of each unique row
    _, unique_indices = np.unique(obj_batch.round(decimals=6), axis=0, return_index=True)

    sorted_indices = np.argsort(obj_batch[unique_indices].T[0])
    ep_indices = []
    for idx in unique_indices[sorted_indices]:
        if not check_dominated(obj_batch, obj_batch[idx]):
            ep_indices.append(idx)
    return ep_indices


class ExternalPareto:
    """
    Maintains an ever-updating set of Pareto-optimal samples (and their objectives).
    If the buffer grows too large, it truncates by crowding distance.
    Also keeps a history of selected objective tuples to avoid re-selecting the same solution.
    """

    def __init__(self, num_select_solutions: int, policy_buffer_size: int):
        # Current Pareto buffer: list of samples and corresponding objective arrays
        self.obj_batch = np.array([])
        self.sample_batch = np.array([])

        # Store historical selected objective values to avoid repeated selection
        self.obj_hist = []

        # Store selected solutions for next extension iteration
        self.selected_batch = np.array([])
        self.selected_obj_batch = np.array([])

        self.policy_buffer_size = policy_buffer_size
        self.num_select_solutions = num_select_solutions

    def crowding_distance_index(self, indices: list) -> None:
        self.selected_obj_batch = self.obj_batch[indices]
        self.selected_batch = self.sample_batch[indices]

    def index(self, indices: list, inplace: bool = True):
        if inplace:
            self.obj_batch, self.sample_batch = \
                map(lambda batch: batch[np.array(indices, dtype=int)], [self.obj_batch, self.sample_batch])
        else:
            return map(lambda batch: deepcopy(batch[np.array(indices, dtype=int)]), [self.obj_batch, self.sample_batch])

    def update(self, sample_batch: list) -> None:
        self.sample_batch = np.append(self.sample_batch, np.array(deepcopy(sample_batch)))
        for sample in sample_batch:
            self.obj_batch = np.vstack([self.obj_batch, sample.objs]) if len(self.obj_batch) > 0 else np.array(
                [sample.objs])
        if len(self.obj_batch) == 0:
            return

        ep_indices = get_ep_indices(self.obj_batch)
        self.index(ep_indices)
        if len(self.sample_batch) > self.policy_buffer_size:
            crowding_distances = self.calculate_crowding_distance(self.obj_batch, True)
            sorted_indices = np.argsort(-crowding_distances)  # Get indices sorted by crowding distance
            pareto_index = []
            for idx in sorted_indices:
                if len(pareto_index) < self.policy_buffer_size:
                    pareto_index.append(idx)
            self.sample_batch = self.sample_batch[pareto_index]
            self.obj_batch = self.obj_batch[pareto_index]

        self.filter_by_crowding_distance(self.num_select_solutions)  # number of elite policies

    def calculate_crowding_distance(self, obj_batch: np.ndarray, extreme: bool):
        if len(obj_batch) == 0:
            return np.array([])

        num_samples = obj_batch.shape[0]
        num_objectives = obj_batch.shape[1]
        crowding_distances = np.zeros(num_samples)
        # print(obj_batch)
        for i in range(num_objectives):
            obj_values = obj_batch[:, i]
            sorted_indices = np.argsort(obj_values)
            distances = np.zeros(num_samples)
            distances[1:-1] = (obj_values[sorted_indices[2:]] - obj_values[sorted_indices[:-2]]) / \
                              (obj_values[sorted_indices[-1]] - obj_values[sorted_indices[0]])

            # Set the boundary points to infinity or a large number to ensure they are selected
            if extreme:
                distances[0] = distances[-1] = np.inf
            crowding_distances[sorted_indices] += distances

        return crowding_distances

    def filter_by_crowding_distance(self, num_select_solutions: int) -> None:
        crowding_distances = self.calculate_crowding_distance(self.obj_batch, True)
        sorted_indices = np.argsort(-crowding_distances)  # Get indices sorted by crowding distance

        # Select the top num_select_solutions policies, ensuring we don't select duplicates based on objective values
        new_indices = []
        for idx in sorted_indices:
            # Convert objectives to a tuple to use as a hashable type for comparison
            obj_tuple = tuple(self.obj_batch[idx])

            if obj_tuple not in self.obj_hist and len(new_indices) < num_select_solutions:
                new_indices.append(idx)
                self.obj_hist.append(obj_tuple)

        # Filter samples by these new indices using the existing self.index method
        self.crowding_distance_index(new_indices)
