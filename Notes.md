# WFCRL Benchmark - Notes

## Best Practices
- **Working Directory**: Always run scripts from the `wfcrl-benchmark` root folder for proper path resolution
- **Neural Network Consistency**: Ensure neural network architectures match between training and evaluation files/inputs (i.e. --hidden_layer_nn 64 64)
- **Minimum Training Requirements**: IPPO/MAPPO require ≈3000 timesteps minimum to save properly
  - Below this threshold: `ValueError: Shape of passed values is (7, 1), indices imply (7, 7)`
- **Maximum Episode Length**: IPPO/MAPPO suffer with instability with longer episodes, maximum episode_length≈500
  - Above this: `/home/reuben/miniconda3/envs/wfcrl/lib/python3.11/site-packages/numpy/core/_methods.py:187: RuntimeWarning: overflow encountered in reduce
  ret = umr_sum(x, axis, dtype, out, keepdims=keepdims, where=where)`
- **MPI Exec**: Using "mpiexec -n 1" causes an MPI error in Linux, not tested on Windows.


## FAST.Farm
- Uses a 100s initialisation time (set in ablaincourt databases). After this has run it then does a further num_steps*dt second simulation. (dt the simulation timestep is 3s by default)
- **NEED TO INVESTIGATE WHAT AN APPROPRIATE SIMULATION LENGTH WOULD BE** Suggestions are that it could take 10mins for the simulation to develop properly.

## Experimental Results

### Performance Comparison (Episode lengths [50,200,600], Iterations [10 000, 50 000, 100 000])

#### Poor Performers
- **IDQN**: Very poor performance, regardless of episode length or iterations the yaws go to the extremes.
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