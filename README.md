# How Many Subproblems? A Controlled Study of Decomposition Size in Multi-Policy Multi-Objective Reinforcement Learning

**Neele Kemper, Jonathan Wurth, Michael Heider, Jörg Hähner**  
*Accepted at IJCNN 2026*  
[Paper](#) | [Appendix](https://zenodo.org/records/19471075)

---

## Abstract

Decomposition-based multi-objective reinforcement learning reduces a multi-objective problem to a set of scalar subproblems, each defined by a weight vector on the preference simplex. While this approach has proven effective, the number of weight vectors is typically chosen heuristically without systematic justification. This paper presents a controlled empirical study examining how the number of weight vectors affects Pareto front quality under fixed training budgets. A multi-policy decomposition framework is evaluated with Soft Actor-Critic and Proximal Policy Optimization, training one independent policy per weight vector via linear scalarization across configurations from two to twenty-one weight vectors and six MuJoCo tasks with two and three objectives. A Bayesian temporal ranking model quantifies uncertainty about which configuration is optimal throughout training, and the best-found configurations are compared against state-of-the-art baselines. The results show that the optimal decomposition size fundamentally depends on the algorithmic structure. Off-policy learning favors small decompositions, with four weight vectors serving as a robust default. On-policy learning requires larger decompositions that scale with the training budget. The common default of six weight vectors is suboptimal for both algorithms, and minimal corner weight configurations lead to significant performance degradation. With properly tuned configurations, the proposed implementations outperform all baselines.

---

## Installation

**Requirements:** Python 3.11, Conda

**1. Create and activate a conda environment:**
```bash
conda create -n morl-subproblems python=3.11
conda activate morl-subproblems
```

**2. Install PyTorch** (select your CUDA version at [pytorch.org](https://pytorch.org/get-started/locally/)):
```bash
pip install torch
```

**3. Install remaining dependencies:**
```bash
pip install -r requirements.txt
```

---

## Repository Structure
```
morl-d-subproblems/
├── agents/
│   ├── multi_policy/
│   │   ├── mo_sac.py                   # MO-SAC: decomposition-based multi-policy SAC
│   │   └── mo_ppo.py                   # MO-PPO: decomposition-based multi-policy PPO
│   ├── single_policy/
│   │   ├── ppo/
│   │   │   ├── ppo.py                  # PPO update logic
│   │   │   ├── ppo_worker.py           # Parallel rollout worker for MO-PPO
│   │   │   └── ...                     # GAE, rollout storage, vectorized envs
│   │   └── sac_continues_action.py     # SAC agent for continuous action spaces
│   └── utils/
│       ├── agent.py                    # Abstract base class for all agents
│       └── sac_net.py                  # Actor and critic network definitions for SAC
├── configs/
│   ├── environment_configs.json        # Environment IDs and Hypervolume reference points
│   └── multi_policy/
│       ├── mo_sac.json                 # MO-SAC hyperparameters
│       └── mo_ppo.json                 # MO-PPO hyperparameters
├── misc/                               # Shared utilities: Pareto archive, evaluation, weight generation
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt 
├── train_mo_sac.py                     # Entry point for MO-SAC training
└── train_mo_ppo.py                     # Entry point for MO-PPO training

```
---

## Usage

### MO-SAC

```bash
python train_mo_sac.py --env halfcheetah --seed 1 --num_subproblems 4 --total_timesteps 10500000
```

### MO-PPO

```bash
python train_mo_ppo.py --env halfcheetah --seed 1 --num_subproblems 12 --total_timesteps 10500000
```

### Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--env` | `halfcheetah` | Environment: `ant`, `halfcheetah`, `hopper`, `humanoid`, `swimmer`, `walker2d` |
| `--seed` | `1` | Random seed |
| `--num_subproblems` | `6` | Number of weight vectors $k$ |
| `--total_timesteps` | `10500000` | Total environment interaction steps |
| `--max_episode_steps` | `500` | Maximum steps per episode |
| `--init_w_sampling` | `uniform` | Weight sampling strategy: `uniform`, `dirichlet` |

Training logs are written to TensorBoard under:
```
{agent_name}/{env}/{init_w_sampling}/k_{num_subproblems:04d}/ws/s_{seed:04d}/
```

To monitor training:
```bash
tensorboard --logdir mo_sac/
```

---

## Acknowledgements

This repository builds on the following codebases:
- [MORL/D](https://github.com/LucasAlegre/morl-baselines) — MORL/D baseline and decomposition framework
- [C-MORL](https://github.com/RuohLiuq/C-MORL) — C-MORL baseline
- [PG-MORL](https://github.com/mit-gfx/PGMORL) — PG-MORL baseline
- [CAPQL](https://github.com/haoyelu/CAPQL) — CAPQL baseline

---

## Citation

```bibtex
@inproceedings{kemper2026subproblems,
  title     = {How Many Subproblems? A Controlled Study of Decomposition Size 
               in Multi-Policy Multi-Objective Reinforcement Learning},
  author    = {Kemper, Neele and Wurth, Jonathan and Heider, Michael and H{\"a}hner, J{\"o}rg},
  booktitle = {Proceedings of the International Joint Conference on Neural Networks (IJCNN)},
  year      = {2026}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.