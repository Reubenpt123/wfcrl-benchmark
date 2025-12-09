#!/usr/bin/env python3
"""
Configuration file for parameter sweeps in WFCRL Benchmark
"""

# ================================
# ENVIRONMENT CONFIGURATIONS
# ================================

ENVIRONMENTS = {
    "ablaincourt": "Dec_Ablaincourt_Floris",
    "turb3": "Dec_Turb3_Row1_Floris",
    "ormonde": "Dec_Ormonde_Floris",
}

# ================================
# SWEEP PARAMETERS
# ================================

# Environment and algorithm selection
ENV_ID = ENVIRONMENTS["ablaincourt"]
ALGORITHMS = ["ifppo"] #, "ifppo"]
# Options: "ippo", "mappo", "ifac", "ifppo", "idqn", "idrqn", "qmix"

# Training parameters
EPISODE_LENGTHS = [100]  # [50, 100, 400]
TOTAL_TIMESTEPS = [300]  # [3000, 10000, 50000, 100000]
SEEDS = [1]  # [1, 2, 3]
WIND_SPEED = 9  # Wind speed in m/s
WIND_DIRECTION = 0  # Wind direction in degrees (meteorological convention: 0=N, 90=E, 180=S, 270=W)

# Plot parameters
PLOT_POWER_YLIM = (7.0, 9.5)  # (7,9.5) is appropriate for Ablaincourt
PLOT_LOAD_YLIM = (5.0, 8.0)   # (5,8) is appropriate for Ablaincourt

# VTK wind output (for FAST.Farm visualization)
VTK_WIND = True  # Set to True to generate VTK wind field files for ParaView visualization

# ================================
# EVALUATION PARAMETERS
# ================================

EVAL_EPISODE_LENGTH = 10
EVAL_SEED = 1
EVALUATION = "both"  # Options: "floris", "fastfarm", "both"

# ================================
# EXECUTION SETTINGS
# ================================

MAX_WORKERS = 2
RESUME = False  # If True, skip completed runs and reuse existing models; if False, train from scratch
DEBUG = True  # Set to True to print all training/evaluation output to terminal
TESTING = True  # If True, prepend "TEST_" to sweep directory name
AUTO_SHUTDOWN = False # If True, shut down machine after sweep completion
SHUTDOWN_DELAY_MINUTES = 5  # Minutes to wait before shutting down

# ================================
# ALGORITHM CONFIGURATIONS
# ================================

ALGORITHM_CONFIGS = {
    "ippo": {
        "script": "algorithms/baseline_ippo.py",
        "supports_plots": True,
        "supports_episode_length": True,
        "hidden_layer_nn": (64, 64),
    },
    "mappo": {
        "script": "algorithms/baseline_mappo.py",
        "supports_plots": True,
        "supports_episode_length": True,
        "hidden_layer_nn": (64, 64),
    },
    "ifac": {
        "script": "algorithms/ifac.py",
        "supports_plots": True,
        "supports_episode_length": False,  # One continuous episode
        "hidden_layer_nn": False,
    },
    "ifppo": {
        "script": "algorithms/ifppo.py",
        "supports_plots": True,
        "supports_episode_length": False,  # One continuous episode
        "hidden_layer_nn": False,
    },
    "idqn": {
        "script": "algorithms/baseline_idqn.py",
        "supports_plots": True,
        "supports_episode_length": True,
        "hidden_layer_nn": 64,
    },
    "idrqn": {
        "script": "algorithms/baseline_idrqn.py",
        "supports_plots": True,
        "supports_episode_length": True,
        "hidden_layer_nn": (64, 64),
    },
    "qmix": {
        "script": "algorithms/baseline_qmix.py",
        "supports_plots": True,
        "supports_episode_length": True,
        "hidden_layer_nn": 64,
    },
}
