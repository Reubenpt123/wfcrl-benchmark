## WFCRL Algorithms

This repository contains a comprehensive benchmarking suite for **Wind Farm Control using Reinforcement Learning (WFCRL)**. The benchmark includes multiple multi-agent RL algorithms designed to optimize wind farm performance by controlling individual wind turbines.

## Overview

The goal is to optimize wind farm performance by controlling individual wind turbines using RL techniques. The environments are either using **FLORIS** (steady-state simulator with low computational cost) or **FAST.Farm** (high-fidelity, dynamic simulator).

Built on the [WFCRL environment suite](https://github.com/ifpen/wfcrl-env) and adapted from [CleanRL](https://github.com/vwxyzjn/cleanrl).

## Algorithms Included

| **Algorithm** | **File** | **Description** |
|---------------|----------|-----------------|
| IDQN | `algorithms/baseline_idqn.py` | Independent Deep Q-Network |
| IDRQN | `algorithms/baseline_idrqn.py` | Independent Deep Recurrent Q-Network |
| IPPO | `algorithms/baseline_ippo.py` | Independent Proximal Policy Optimization - [Yu et. al](https://arxiv.org/abs/2103.01955) |
| MAPPO | `algorithms/baseline_mappo.py` | Multi-Agent Proximal Policy Optimization - [Yu et. al](https://arxiv.org/abs/2103.01955) |
| QMIX | `algorithms/baseline_qmix.py` | QMIX - [Rashid et. al](https://arxiv.org/abs/1803.11485) |
| IFAC | `algorithms/ifac.py` | Independent Fourier-basis Actor-Critic |
| IFPPO | `algorithms/ifppo.py` | Independent Fourier-basis Proximal Policy Optimization |

## Simulators

- **FLORIS**: A steady-state wind farm simulator that models the wake effects between turbines
- **FAST.Farm**: A high-fidelity, dynamic wind farm simulator that captures complex interactions

## Environments

The repository includes several pre-defined wind farm layouts:
- **Ablaincourt**: Main benchmark environment
- **Ormonde**: Additional test environment

## Installation

1. Install the dependencies:
```bash
pip install -r requirements.txt
```

2. For experiment tracking with Weights & Biases, add your API key in a `.env` file at the root:
```bash
WANDB_API_KEY=your_api_key
```

## Quick Start

### Single Algorithm Training

Launch an IPPO training experiment on the `Dec_Ablaincourt_Floris` environment:

```bash
python algorithms/baseline_ippo.py --seed 1 --env_id Dec_Ablaincourt_Floris --total_timesteps 100000
```

### Evaluation

Evaluate a trained model on the `Dec_Ablaincourt_Fastfarm` environment:

```bash
mpiexec -n 1 python algorithms/eval.py --seed 0 --algo ippo --env_id Dec_Ablaincourt_Fastfarm --num_episodes 1 --pretrained_models path/to/run
```

### Batch Training

Run all algorithms with the provided batch script:

```bash
bash scripts/ablaincourt_batch.sh
```

### Parameter Sweeps

Run systematic parameter sweeps across multiple configurations:

```bash
# Basic parameter sweep
python scripts/parameter_sweep.py

# Custom parameters with automatic shutdown
python scripts/parameter_sweep.py \
    --episode_lengths 50 100 200 400 \
    --total_timesteps 10000 20000 50000 100000 \
    --shutdown
```

## Wind Scenarios

Add `--scenario windrose` to train/eval on *Wind Scenario II* (wind rose evaluation):

```bash
python algorithms/baseline_ippo.py --seed 1 --env_id Dec_Ablaincourt_Floris --total_timesteps 1000000 --scenario windrose
```

## Scripts

The `scripts` directory contains:
- **`ablaincourt_batch.sh`**: Batch training script for all algorithms
- **`parameter_sweep.py`**: Python script for systematic parameter exploration
- **`eval_batch.sh`**: Batch evaluation script
- **`ormonde_batch.sh`**: Alternative environment batch script
