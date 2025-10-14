#!/usr/bin/env python3
"""
WFCRL Benchmark - Ablaincourt Batch Training Script

Python script to run all benchmark algorithms on the Ablaincourt environment.
"""

import os
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
import sys


# Algorithm configurations
ALGORITHMS = {
    "IDQN": "baseline_idqn.py",
    "IDRQN": "baseline_idrqn.py", 
    "IPPO": "baseline_ippo.py",
    "MAPPO": "baseline_mappo.py",
    "QMIX": "baseline_qmix.py",
    "IFAC": "ifac.py",
    "IFPPO": "ifppo.py"
}


def log_message(message: str, log_file: Path = None):
    """Print message and optionally log to file"""
    print(message)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(message + '\n')


def get_most_recent_model_path(algorithm_name: str, base_dir: Path) -> Path:
    """Read the most recent model path from the tracking file"""
    path_file = base_dir / "scripts" / "most_recent_models" / f"{algorithm_name.lower()}_path.txt"
    
    if path_file.exists():
        with open(path_file, 'r') as f:
            return Path(f.read().strip())
    return None


def evaluate_algorithm(algorithm_name: str, run_path: Path, params: dict, base_dir: Path, log_file: Path = None) -> bool:
    """Run evaluation for a trained algorithm"""
    timestamp = datetime.now().strftime("%d/%m/%y %H:%M")
    log_message(f"[{timestamp}] Starting {algorithm_name} evaluation", log_file)
    
    # Build evaluation command
    cmd = [
        "python", "algorithms/evaluate.py",
        "--algorithm", algorithm_name.lower(),
        "--env_id", params["env_id"],
        "--pretrained_models", str(run_path),
        "--scenario", "constant",
        "--episode_length", str(params.get("eval_episode_length", 300))
    ]
    
    # Add hidden_layer_nn for algorithms with different architectures
    if algorithm_name in ["IDQN", "QMIX"]:
        # Single layer: (64,)
        cmd.extend(["--hidden_layer_nn", "64"])
    elif algorithm_name == "IDRQN":
        # Two layers: (64, 64) - matches training default
        cmd.extend(["--hidden_layer_nn", "64", "64"])
    elif algorithm_name in ["IFAC", "IFPPO"]:
        # No hidden layers (Fourier features only): False
        cmd.extend(["--hidden_layer_nn", "False"])
    
    try:
        # Run evaluation
        result = subprocess.run(cmd, check=True, timeout=1800, capture_output=True, text=True, cwd=base_dir)
        
        timestamp = datetime.now().strftime("%d/%m/%y %H:%M")
        log_message(f"[{timestamp}] {algorithm_name} evaluation completed successfully", log_file)
        return True
        
    except subprocess.CalledProcessError as e:
        timestamp = datetime.now().strftime("%d/%m/%y %H:%M")
        log_message(f"[{timestamp}] {algorithm_name} evaluation failed with exit code {e.returncode}", log_file)
        if e.stderr:
            log_message(f"Error output: {e.stderr[:500]}", log_file)
        if e.stdout:
            log_message(f"Standard output: {e.stdout[:500]}", log_file)
        return False
    except subprocess.TimeoutExpired as e:
        timestamp = datetime.now().strftime("%d/%m/%y %H:%M")
        log_message(f"[{timestamp}] {algorithm_name} evaluation timed out after 30 minutes", log_file)
        return False


def train_algorithm(algorithm_name: str, script_path: Path, params: dict, log_file: Path = None) -> bool:
    """Train a single algorithm"""
    timestamp = datetime.now().strftime("%d/%m/%y %H:%M")
    log_message(f"[{timestamp}] Starting {algorithm_name} training", log_file)
    
    # Build command
    cmd = [
        "python", str(script_path),
        "--seed", str(params["seed"]),
        "--env_id", params["env_id"], 
        "--total_timesteps", str(params["total_timesteps"]),
        "--wandb_project_name", params["wandb_project_name"]
    ]
    
    # Add episode length for baseline algorithms
    if algorithm_name in ["IDQN", "IDRQN", "IPPO", "MAPPO", "QMIX"]:
        cmd.extend(["--episode_length", str(params["episode_length"])])
    
    # Add optional flags
    if params.get("track"):
        cmd.append("--track")
    else:
        cmd.append("--no-track")
        
    if params.get("save_model", True):
        cmd.append("--save_model")
    
    try:
        # Run training
        result = subprocess.run(cmd, check=True, timeout=7200)  # 2 hour timeout
        
        timestamp = datetime.now().strftime("%d/%m/%y %H:%M")
        log_message(f"[{timestamp}] {algorithm_name} completed successfully", log_file)
        return True
        
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        timestamp = datetime.now().strftime("%d/%m/%y %H:%M")
        log_message(f"[{timestamp}] {algorithm_name} failed: {str(e)}", log_file)
        return False


def main():
    parser = argparse.ArgumentParser(description="WFCRL Benchmark - Simplified Batch Training")
    
    # Core parameters
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--total_timesteps", type=int, default=4000, help="Training timesteps")
    parser.add_argument("--episode_length", type=int, default=200, help="Episode length")
    parser.add_argument("--eval_episode_length", type=int, default=300, help="Evaluation episode length")
    parser.add_argument("--track", action="store_true", help="Enable wandb tracking")
    
    # Select algorithms to run
    parser.add_argument("--algorithms", nargs="+", choices=list(ALGORITHMS.keys()),
                       default=list(ALGORITHMS.keys()), help="Algorithms to train")
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path("/home/reuben/code/wfcrl-benchmark")
    algorithms_dir = base_dir / "algorithms"
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Create log file
    timestamp = datetime.now().strftime("%d%m%y_%H%M%S")
    log_file = logs_dir / f"batch_run_{timestamp}.log"
    
    # Training parameters
    params = {
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "episode_length": args.episode_length,
        "eval_episode_length": args.eval_episode_length,
        "env_id": "Dec_Ablaincourt_Floris",
        "wandb_project_name": "benchmark-wfcrl-test",
        "track": args.track,
        "save_model": True
    }
    
    # Start batch training
    start_time = datetime.now().strftime("%d/%m/%y %H:%M")
    log_message("=" * 50, log_file)
    log_message("WFCRL BENCHMARK BATCH TRAINING", log_file)
    log_message("=" * 50, log_file)
    log_message(f"Started: {start_time}", log_file)
    log_message(f"Algorithms: {', '.join(args.algorithms)}", log_file)
    log_message(f"Parameters: {params}", log_file)
    log_message("", log_file)
    
    # Train each algorithm
    successful = 0
    total = len(args.algorithms)
    trained_models = {}  # Store model paths for later evaluation
    
    for i, algorithm_name in enumerate(args.algorithms, 1):
        log_message(f"Training {i}/{total}: {algorithm_name}", log_file)
        
        script_path = algorithms_dir / ALGORITHMS[algorithm_name]
        if script_path.exists():
            if train_algorithm(algorithm_name, script_path, params, log_file):
                successful += 1
                
                # Get the model path
                model_path = get_most_recent_model_path(algorithm_name, base_dir)
                if model_path:
                    trained_models[algorithm_name] = model_path
                    log_message(f"Model saved to: {model_path}", log_file)
                    
                    # Run evaluation on Floris with 300 timesteps
                    log_message(f"Running evaluation for {algorithm_name}...", log_file)
                    evaluate_algorithm(algorithm_name, model_path, params, base_dir, log_file)
                else:
                    log_message(f"Warning: Could not find model path for {algorithm_name}", log_file)
        else:
            log_message(f"Script not found: {script_path}", log_file)
    
    # Summary
    end_time = datetime.now().strftime("%d/%m/%y %H:%M") 
    log_message("", log_file)
    log_message("=" * 50, log_file)
    log_message(f"Completed: {end_time}", log_file)
    log_message(f"Results: {successful}/{total} algorithms successful", log_file)
    log_message(f"Log file: {log_file}", log_file)
    
    # Exit with appropriate code
    if successful == total:
        print(f"\nAll {total} algorithms completed successfully!")
        sys.exit(0)
    else:
        print(f"\n{successful}/{total} algorithms completed successfully")
        sys.exit(1)


if __name__ == "__main__":
    main()
