import os
import json
import torch
import random
import numpy as np
from typing import Optional
from dataclasses import dataclass, field

BASE_DIR = 'results'


@dataclass
class ParetoFrontStore:
    path: str
    buf: dict[str, np.ndarray] = field(default_factory=dict)

    def add(self, front: np.ndarray, global_step: int,
            weights: Optional[np.ndarray] = None) -> None:
        key = f't_{global_step:08d}'
        self.buf[key] = np.asarray(front, dtype=np.float32)  # [N, R]
        if weights is not None:
            self.buf[f'{key}_w'] = np.asarray(weights, dtype=np.float32)  # [N, R]

    def flush(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        np.savez_compressed(self.path, **self.buf)

    def clear(self) -> None:
        self.buf.clear()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_env_config(path: str) -> dict:
    with open(path) as json_data:
        d = json.load(json_data)
    return d


def read_algo_config(path: str) -> dict:
    with open(path) as json_data:
        d = json.load(json_data)
    return d

def setup_directories(log_dir: str, file_name: str, save_fronts: bool, save_models: bool) -> tuple[
    str, str, str, ParetoFrontStore | None]:
    runs_path = f'{BASE_DIR}/{log_dir}/runs'
    model_path = f'{BASE_DIR}/{log_dir}/models'
    front_path = f'{BASE_DIR}/{log_dir}/fronts'
    config_path = f'{BASE_DIR}/{log_dir}/config'
    pf_store = None
    os.makedirs(runs_path, exist_ok=True)
    if save_models:
        os.makedirs(model_path, exist_ok=True)
    if save_fronts:
        os.makedirs(front_path, exist_ok=True)
        pf_store = ParetoFrontStore(path=f'{front_path}/{file_name}.npz')
    os.makedirs(config_path, exist_ok=True)
    return runs_path, model_path, config_path, pf_store
