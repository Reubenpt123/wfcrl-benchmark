# WFCRL Benchmark - Development Notes

## Setup and Usage Guidelines

### Best Practices
- **Working Directory**: Always run scripts from the `wfcrl-benchmark` root folder for proper path resolution
- **Neural Network Consistency**: Ensure neural network architectures match between training and evaluation files
- **Minimum Training Requirements**: IPPO/MAPPO require ≈3000 timesteps minimum to save properly
  - Below this threshold: `ValueError: Shape of passed values is (7, 1), indices imply (7, 7)`
  - This can be useful for debugging runs

## Experimental Results

### Performance Comparison (10,000 iterations, episode length 100)

#### Poor Performers
- **IDQN**: Really bad performance
- **IDRQN**: Really bad performance
- **QMIX**: Really bad performance

#### Moderate Performers
- **IPPO**: Bad but better than Q-learning methods
- **MAPPO**: Bad but better than Q-learning methods

#### Strong Performers
- **IFAC**: Surprisingly good performance
  - Takes longer to run
  - Does not work in episodic way
  - Good performance by ≈5000 iterations
  - Produces optimal yaw angles

- **IFPPO**: Surprisingly good performance
  - Takes longer to run
  - Does not work in episodic way
  - Good performance by ≈5000 iterations
  - Maybe slightly higher yield than IFAC
  - Initially more unstable yaw outputs
  - Produces similar optimal yaws to IFAC

### Current Experiments
- Testing 10,000 iterations with episode length 200

## Algorithm Insights

### Episode Length Impact
- **Episode Length 100**: Clear performance hierarchy established
- **Episode Length 200**: Under investigation

### Fourier-Based Methods (IFAC/IFPPO)
- **Training Style**: Non-episodic approach
- **Convergence**: Requires more iterations but achieves better final performance
- **Computational Cost**: Higher than traditional methods
- **Output Quality**: More optimal yaw control strategies

## Technical Notes

### Common Issues
1. **Path Dependencies**: Scripts must be run from correct working directory
2. **Architecture Matching**: Training/eval network size consistency critical
3. **Minimum Timesteps**: Some algorithms have minimum training requirements

### Algorithm Categories
- **Q-Learning Methods**: IDQN, IDRQN, QMIX - Poor performance on wind farm control
- **Policy Gradient Methods**: IPPO, MAPPO - Moderate performance
- **Fourier-Based Methods**: IFAC, IFPPO - Superior performance with different training paradigm

