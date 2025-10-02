# WFCRL Benchmark Architecture Documentation

## Overview

The **WFCRL Benchmark** (Wind Farm Control Reinforcement Learning) is a comprehensive multi-agent reinforcement learning framework designed for optimizing wind farm control strategies. The system trains multiple RL agents to coordinate wind turbine yaw control to maximize power generation while minimizing structural loads.

## Project Structure

```
wfcrl-benchmark/
├── algos/                  # Core RL algorithm implementations
│   ├── baseline_ippo.py    # Independent Proximal Policy Optimization
│   ├── baseline_mappo.py   # Multi-Agent Proximal Policy Optimization
│   ├── baseline_qmix.py    # QMIX value decomposition method
│   ├── baseline_idqn.py    # Independent Deep Q-Network
│   ├── baseline_idrqn.py   # Independent Deep Recurrent Q-Network
│   ├── ifac.py            # Independent Actor-Critic with Fourier basis
│   ├── ifppo.py           # Independent PPO with Fourier features
│   ├── evaluate.py        # Unified evaluation framework
│   ├── extractors.py      # Observation and feature extractors
│   └── utils.py           # Utilities and helper functions
├── data/
│   └── smarteole.csv      # Real wind data for evaluation scenarios
├── scripts/               # Batch experiment scripts
│   ├── jobs.sbatch        # SLURM cluster job submission
│   └── run_bench_*.bat    # Windows batch training scripts
├── __simul__/             # Simulation outputs and logs
├── requirements.txt       # Python dependencies
└── README.md             # Basic usage instructions
```

## System Architecture

### 1. Environment Integration Layer

The benchmark integrates with the [WFCRL environment suite](https://github.com/ifpen/wfcrl-env) which provides:

- **Multiple Simulators**: Floris (fast wake modeling) and FastFarm (high-fidelity CFD)
- **Realistic Wind Farms**: Various layouts (3-turbine test, Ablaincourt, Horns Rev, etc.)
- **Multi-Agent Interface**: PettingZoo-compatible environment for independent turbine control
- **Physics-Based Rewards**: Power generation and structural load objectives

#### Environment Configurations
```python
# Available environments in benchmark
ENVIRONMENTS = {
    "Dec_Turb3_Row1_Floris": "3-turbine test case with Floris",
    "Dec_Ablaincourt_Floris": "Ablaincourt wind farm with Floris",
    "Dec_WMR_Floris": "Wind farm with Floris simulator",
    "Dec_HornsRev1_Floris": "Horns Rev wind farm with Floris",
    "*_Fastfarm": "High-fidelity CFD variants"
}
```

### 2. Observation Processing System

The `extractors.py` module provides three key components:

#### VectorExtractor
- **Purpose**: Converts dictionary observations to flat vectors
- **Features**: Handles both continuous (Box) and discrete (MultiDiscrete) spaces
- **Filtering**: Excludes irrelevant observations (pitch, torque by default)
- **Normalization**: Automatic scaling to [0,1] range for continuous spaces

```python
class VectorExtractor:
    def __init__(self, space, filter_out=["pitch", "torque"])
    def forward(self, dic) -> np.ndarray  # Dict -> Vector
    def make_dict(self, vector) -> dict   # Vector -> Dict
```

#### FourierExtractor
- **Purpose**: Generates Fourier basis features for continuous control
- **Applications**: Enhanced function approximation in continuous spaces
- **Configuration**: Adjustable order, dimensionality, and learning parameters
- **Advanced Features**: Hypernetwork support for adaptive basis generation

#### DfacSPaceExtractor
- **Purpose**: Specialized extractor combining local and global observations
- **Usage**: IFAC algorithm for decentralized control with global awareness

### 3. Core Algorithm Implementations

#### Policy-Based Methods

**IPPO (Independent PPO)** - `baseline_ippo.py`
- **Architecture**: Separate actor-critic networks per agent
- **Training**: Independent policy gradient optimization
- **Features**: Observation normalization, GAE, entropy regularization
- **Network Structure**: Configurable hidden layers (default: 64x64)

**MAPPO (Multi-Agent PPO)** - `baseline_mappo.py`
- **Architecture**: Shared critic with decentralized actors
- **Training**: Centralized training, decentralized execution (CTDE)
- **Global State**: Uses farm-wide observations for value estimation
- **Coordination**: Implicit through shared critic and experience

#### Value-Based Methods

**QMIX** - `baseline_qmix.py`
- **Architecture**: Individual Q-networks + mixing network
- **Training**: Centralized value decomposition
- **Memory**: Experience replay with episode-based sampling
- **Coordination**: Explicit through value mixing and global state

**IDQN/IDRQN** - `baseline_idqn.py`, `baseline_idrqn.py`
- **Architecture**: Independent Q-networks per agent
- **Variants**: Standard feedforward (IDQN) and recurrent (IDRQN)
- **Training**: Independent experience replay and target networks
- **Exploration**: Epsilon-greedy with linear decay

#### Actor-Critic Methods

**IFAC** - `ifac.py`
- **Architecture**: Simple online actor-critic
- **Features**: Fourier basis function approximation
- **Training**: Online policy gradient with baseline
- **Specialization**: Optimized for continuous control problems

### 4. Agent Network Architectures

#### Standard Policy Network (IPPO/MAPPO)
```python
class Agent(nn.Module):
    def __init__(self, observation_space, action_space, hidden_layers):
        # Actor network: obs -> action_mean
        self.actor = nn.Sequential(...)
        # Critic network: obs -> value
        self.critic = nn.Sequential(...)
        # Learnable action std
        self.log_std = nn.Parameter(...)
```

#### Q-Network (QMIX/IDQN)
```python
class QNetwork(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dims):
        # Input: concatenated [observation, last_action]
        # Recurrent processing (QMIX/IDRQN)
        self.rnn = nn.GRUCell(...)
        # Separate output heads per action dimension
        self.output_layers = nn.ModuleList(...)
```

#### QMIX Mixer Network
```python
class QMixer(nn.Module):
    def __init__(self, num_agents, observation_space, hidden_dim):
        # Hypernetworks for dynamic weight generation
        self.hyper_network_w1 = nn.Linear(...)
        self.hyper_network_b1 = nn.Linear(...)
        # Mixing: individual Q-values -> global Q-value
```

### 5. Training Pipeline

#### Common Training Loop Structure
1. **Environment Reset**: Initialize wind conditions and turbine states
2. **Experience Collection**: 
   - Multi-agent step iteration
   - Action selection via current policy
   - Environment state transition
   - Reward and observation collection
3. **Learning Updates**:
   - Policy gradient computation (PPO variants)
   - Q-value updates with target networks (DQN variants)
   - Experience replay sampling (value-based methods)
4. **Evaluation**: Periodic assessment on test scenarios

#### Multi-Agent Coordination Patterns

**Independent Training** (IPPO, IDQN):
```python
for agent_id, agent in enumerate(agents):
    obs = partial_observations[agent_id]
    action = agent.get_action(obs)
    # Each agent trains independently
```

**Centralized Training** (MAPPO, QMIX):
```python
global_state = env.get_global_state()
for agent_id, agent in enumerate(agents):
    local_obs = partial_observations[agent_id]
    action = agent.get_action(local_obs)
    value = shared_critic(global_state)  # MAPPO
    # OR
    global_q = mixer(individual_qs, global_state)  # QMIX
```

### 6. Evaluation Framework

The `evaluate.py` module provides unified evaluation across all algorithms:

#### Evaluation Modes

**Single Condition Testing**:
```python
env.reset(options={"wind_speed": 8, "wind_direction": 270})
score = eval_policies(env, trained_agents)
```

**Wind Rose Evaluation**:
```python
wind_rose = prepare_eval_windrose("data/smarteole.csv", num_bins=5)
weighted_score, episode_rewards, conditions = eval_wind_rose(
    env, trained_agents, wind_rose
)
```

#### Cross-Simulator Transfer
- **Training**: Typically on Floris (fast simulation)
- **Evaluation**: On FastFarm (high-fidelity simulation)
- **Purpose**: Test sim-to-sim generalization capabilities

### 7. Reward Structure

#### Multi-Objective Optimization
```python
# Power maximization (primary objective)
power_reward = sum(turbine_powers) / max_possible_power

# Load penalty (structural constraint)
load_penalty = load_coef * mean(abs(turbine_loads))

# Combined reward
total_reward = power_reward - load_penalty
```

#### Reward Shaping Options
- **StepPercentage**: Gradual reward based on episode progress
- **RewardShaper**: Customizable reward modifications
- **Load Coefficient**: Balances power vs. load trade-off

### 8. Configuration Management

#### Hyperparameter Configuration
All algorithms use `@dataclass` decorators with `tyro` for CLI integration:

```python
@dataclass
class Args:
    # Environment settings
    env_id: str = "Dec_Turb3_Row1_Floris"
    total_timesteps: int = int(1e5)
    
    # Algorithm hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    
    # Architecture settings
    hidden_layer_nn: tuple = (64, 64)
    
    # Evaluation settings
    scenario: str = "constant"  # or "windrose"
    wind_data: str = "data/smarteole.csv"
```

#### Runtime Configuration
```bash
# Basic training
python algos/baseline_ippo.py --env_id Dec_Turb3_Row1_Floris --total_timesteps 100000

# Wind rose scenario
python algos/baseline_ippo.py --scenario windrose --total_timesteps 1000000

# Hyperparameter tuning
python algos/baseline_mappo.py --learning_rate 1e-3 --hidden_layer_nn 128 128
```

### 9. Logging and Monitoring

#### Tensorboard Integration
- **Scalar Logging**: Rewards, losses, learning metrics
- **Farm Metrics**: Power generation, load indicators
- **Agent Metrics**: Individual turbine performance

#### Weights & Biases Support
- **Experiment Tracking**: Automatic hyperparameter logging
- **Visualization**: Real-time training curves
- **Model Artifacts**: Saved checkpoints and configurations

#### Custom CSV Logging
```python
class LocalSummaryWriter:
    def add_scalar(self, tag, value, step)  # Standard logging
    def add_config(self, args)              # Hyperparameter storage
    def save(self)                          # CSV export
```

### 10. Deployment and Experimentation

#### High-Performance Computing Support
- **SLURM Integration**: Cluster job submission via `jobs.sbatch`
- **Array Jobs**: Parallel seed experiments
- **GPU Support**: CUDA acceleration for neural networks

#### Batch Experimentation
```bash
# Example batch script structure
scripts/run_bench_0.bat:
  - Multiple algorithm comparisons
  - Different environment scales
  - Systematic seed variation
  - Cross-simulator evaluation
```

#### Model Persistence
- **Checkpoint Saving**: Automatic model state preservation
- **Cross-Evaluation**: Load models trained with one algorithm for evaluation
- **Transfer Learning**: Pre-trained model initialization

### 11. Dependencies and Requirements

#### Core Dependencies
```python
# Deep Learning
torch >= 2.0.1
stable-baselines3

# Environment
wfcrl @ git+https://github.com/ifpen/wfcrl-env.git

# Utilities
gymnasium
numpy
pandas
matplotlib
seaborn

# Experiment Management
tensorboard
wandb
tyro
python-dotenv
```

#### Development Tools
- **Configuration**: `tyro` for CLI argument parsing
- **Visualization**: `matplotlib`, `seaborn` for plotting
- **Data Processing**: `pandas` for wind data analysis

### 12. Extension Points

#### Adding New Algorithms
1. Create new file in `algos/` following naming convention
2. Implement `Agent` class with required methods
3. Add algorithm mapping in `evaluate.py`
4. Define hyperparameter dataclass
5. Implement training loop following established patterns

#### Custom Environments
1. Extend WFCRL environment suite
2. Update environment ID mapping
3. Adjust observation/action space handling
4. Modify reward structure if needed

#### Feature Engineering
1. Extend `extractors.py` with new observation processors
2. Implement custom feature transformation
3. Integrate with existing algorithm architectures
4. Test across multiple algorithms for consistency

## Best Practices

### 1. Reproducibility
- Always set random seeds across all libraries
- Use deterministic CUDA operations when available
- Log all hyperparameters and configurations
- Version control experiment scripts

### 2. Scalability
- Start with small environments (3-turbine) for development
- Scale to larger farms (HornsRev1) for final evaluation
- Use appropriate timestep budgets per environment size
- Monitor computational resource usage

### 3. Evaluation
- Always evaluate on multiple seeds (typically 5)
- Test both constant wind and wind rose scenarios
- Perform cross-simulator validation (Floris -> FastFarm)
- Compare against baseline algorithms

### 4. Debugging
- Enable debug logging for detailed step-by-step analysis
- Use smaller timestep limits during development
- Visualize training curves regularly
- Monitor individual agent behaviors

This architecture provides a robust foundation for multi-agent reinforcement learning research in wind farm control, with clear separation of concerns, extensibility, and comprehensive evaluation capabilities.