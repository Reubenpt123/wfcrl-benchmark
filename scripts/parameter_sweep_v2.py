#!/usr/bin/env python3
"""
Comprehensive Parameter Sweep Script for WFCRL Benchmark
"""

import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import tyro


# ================================
# DEFAULT PARAMETERS
# ================================

# Default environment configurations
DEFAULT_ENVS = {
    "ablaincourt": "Dec_Ablaincourt_Floris",
    "turb3": "Dec_Turb3_Row1_Floris",
    "ormonde": "Dec_Ormonde_Floris",
}

DEBUG = False  # Set to True to print all training/evaluation output to terminal

# Default environment and algorithm selection
DEFAULT_ENV_ID = DEFAULT_ENVS["ablaincourt"]
DEFAULT_ALGORITHMS = ["ippo", "mappo", "idqn", "idrqn", "qmix"]
# Options: "ippo", "mappo", "ifac", "ifppo", "idqn", "idrqn", "qmix"

# Default training parameters
DEFAULT_EPISODE_LENGTHS = [50]#, 100, 400]
DEFAULT_TOTAL_TIMESTEPS = [500]#0, 10000, 50000, 100000]
DEFAULT_SEEDS = [1]#, 2, 3]

# Default plot parameters
DEFAULT_PLOT_POWER_YLIM = (7.0, 9.5)  # (7,9.5) is appropriate for Ablaincourt
DEFAULT_PLOT_LOAD_YLIM = (5.0, 8.0)   # (5,8) is appropriate for Ablaincourt

# Evaluation parameters
DEFAULT_EVAL_EPISODE_LENGTH = 1000
DEFAULT_EVAL_SEED = 0

# Parallel execution settings
DEFAULT_MAX_WORKERS = 4

# Algorithm-specific configurations
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


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class RunConfig:
    """Configuration for a single training run."""

    algorithm: str
    env_id: str
    episode_length: int
    total_timesteps: int
    seed: int
    plot_power_ylim: Optional[tuple[float, float]] = None
    plot_load_ylim: Optional[tuple[float, float]] = None

    def get_run_id(self) -> str:
        """Generate unique identifier for this run."""
        return f"{self.algorithm}_{self.env_id}_el{self.episode_length}_tt{self.total_timesteps}_s{self.seed}"


@dataclass
class SweepConfig:
    """Configuration for the entire parameter sweep."""

    algorithms: list[str]
    env_id: str
    episode_lengths: list[int] = field(default_factory=lambda: DEFAULT_EPISODE_LENGTHS)
    total_timesteps: list[int] = field(default_factory=lambda: DEFAULT_TOTAL_TIMESTEPS)
    seeds: list[int] = field(default_factory=lambda: DEFAULT_SEEDS)
    plot_power_ylim: Optional[tuple[float, float]] = DEFAULT_PLOT_POWER_YLIM
    plot_load_ylim: Optional[tuple[float, float]] = DEFAULT_PLOT_LOAD_YLIM
    eval_episode_length: int = DEFAULT_EVAL_EPISODE_LENGTH
    eval_seed: int = DEFAULT_EVAL_SEED
    max_workers: int = DEFAULT_MAX_WORKERS
    resume: bool = True
    output_dir: Optional[Path] = None
    debug: bool = DEBUG
    auto_shutdown: bool = False
    """Automatically shutdown the system after sweep completes"""
    shutdown_delay_minutes: int = 5
    """Minutes to wait before shutting down (allows time to cancel if needed)"""


# =============================================================================
# MAIN PARAMETER SWEEP CLASS
# =============================================================================


class ParameterSweep:
    """Manages parameter sweep execution with resume capability."""

    def __init__(self, config: SweepConfig):
        self.config = config
        
        # Determine sweep directory
        if config.output_dir:
            # User specified a directory - use it (for resume or custom location)
            self.sweep_dir = config.output_dir
            self.sweep_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if resuming an existing sweep
            existing_config = self.sweep_dir / "sweep_config.json"
            if existing_config.exists():
                # Resuming - load the original timestamp
                with open(existing_config, "r") as f:
                    saved_config = json.load(f)
                    self.timestamp = saved_config.get("timestamp", datetime.now().strftime("%H-%M-%S_%d-%m-%y"))
                print(f"📂 Resuming existing sweep from: {self.sweep_dir}")
            else:
                # New sweep in custom directory
                self.timestamp = datetime.now().strftime("%H-%M-%S_%d-%m-%y")
        else:
            # No directory specified - create new one with timestamp
            self.timestamp = datetime.now().strftime("%H-%M-%S_%d-%m-%y")
            self.sweep_dir = Path("parameter_sweeps") / f"{self.timestamp}_parameter_sweep"
            self.sweep_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.progress_file = self.sweep_dir / "progress.json"
        self.completed_runs = self._load_progress()

        # Save sweep configuration
        self._save_config()

    def _load_progress(self) -> set[str]:
        """Load completed runs from progress file."""
        if not self.progress_file.exists():
            return set()

        try:
            with open(self.progress_file, "r") as f:
                data = json.load(f)
                return set(data.get("completed_runs", []))
        except (json.JSONDecodeError, IOError):
            print(f"Warning: Could not load progress from {self.progress_file}")
            return set()

    def _save_progress(self, run_id: str):
        """Save progress after completing a run with file locking."""
        import fcntl
        
        # Use a lock file to ensure atomic updates
        lock_file = self.progress_file.parent / f"{self.progress_file.name}.lock"
        
        # Acquire lock
        with open(lock_file, "w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            
            try:
                # Re-read the progress file to get latest state
                if self.progress_file.exists():
                    with open(self.progress_file, "r") as f:
                        data = json.load(f)
                        completed_runs = set(data.get("completed_runs", []))
                else:
                    completed_runs = set()
                
                # Add new run
                completed_runs.add(run_id)
                
                # Save updated progress
                data = {
                    "completed_runs": list(completed_runs),
                    "last_updated": datetime.now().isoformat(),
                }
                with open(self.progress_file, "w") as f:
                    json.dump(data, f, indent=2)
                
                # Update in-memory state
                self.completed_runs = completed_runs
            finally:
                # Lock is automatically released when the with block exits
                pass

    def _save_config(self):
        """Save sweep configuration to JSON file."""
        config_file = self.sweep_dir / "sweep_config.json"
        config_dict = asdict(self.config)
        # Convert Path objects to strings for JSON serialization
        if config_dict.get("output_dir"):
            config_dict["output_dir"] = str(config_dict["output_dir"])
        config_dict["sweep_dir"] = str(self.sweep_dir)
        config_dict["timestamp"] = self.timestamp

        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2)

    def generate_run_configs(self) -> list[RunConfig]:
        """Generate all run configurations for the sweep."""
        configs = []
        for algo in self.config.algorithms:
            for episode_length in self.config.episode_lengths:
                for total_timesteps in self.config.total_timesteps:
                    for seed in self.config.seeds:
                        run_config = RunConfig(
                            algorithm=algo,
                            env_id=self.config.env_id,
                            episode_length=episode_length,
                            total_timesteps=total_timesteps,
                            seed=seed,
                            plot_power_ylim=self.config.plot_power_ylim,
                            plot_load_ylim=self.config.plot_load_ylim,
                        )
                        configs.append(run_config)
        return configs

    def run_training(self, run_config: RunConfig) -> dict:
        """Execute a single training run."""
        run_id = run_config.get_run_id()

        # Check if already completed in progress.json
        if self.config.resume and run_id in self.completed_runs:
            print(f"⏭️  Skipping completed run: {run_id}")
            return {"status": "skipped", "run_id": run_id}

        # Check if a model already exists in parameter_sweep directories (fully trained models only)
        if self.config.resume:
            existing_run_path = self._find_existing_run_path(run_config)
            if existing_run_path:
                print(f"🔍 Found existing model (not in progress): {run_id}")
                print(f"   Located at: {existing_run_path}")
                
                # Check if model is external and needs to be copied
                run_path_obj = Path(existing_run_path).resolve()
                sweep_dir_obj = self.sweep_dir.resolve()
                
                if sweep_dir_obj in run_path_obj.parents:
                    # Already in sweep directory
                    final_path = existing_run_path
                    print(f"   ✓ Model already in sweep directory")
                else:
                    # External model, copy it
                    print(f"   📋 Copying model to sweep directory...")
                    organised_path = self._organise_run(existing_run_path, run_config, copy_only=True)
                    final_path = organised_path if organised_path else existing_run_path
                
                # Mark as completed in progress
                self._save_progress(run_id)
                
                return {
                    "status": "success",
                    "run_id": run_id,
                    "run_path": final_path,
                    "duration": 0,
                    "reused": True,
                }

        print(f"🚀 Starting training: {run_id}")
        start_time = time.time()

        # Get algorithm script path
        algo_config = ALGORITHM_CONFIGS.get(run_config.algorithm)
        if not algo_config:
            return {
                "status": "error",
                "run_id": run_id,
                "error": f"Unknown algorithm: {run_config.algorithm}",
            }

        script_path = Path(algo_config["script"])
        if not script_path.exists():
            return {
                "status": "error",
                "run_id": run_id,
                "error": f"Script not found: {script_path}",
            }

        # Build command
        cmd = [
            "python",
            str(script_path),
            "--env_id",
            run_config.env_id,
             "--total_timesteps",
            str(run_config.total_timesteps),
            "--seed",
            str(run_config.seed),
        ]
        
        # Add episode_length only if algorithm supports it
        if algo_config.get("supports_episode_length", True):
            cmd.extend(["--episode_length", str(run_config.episode_length)])
        
        # Add hidden_layer_nn if specified
        if "hidden_layer_nn" in algo_config:
            hidden_layers = algo_config["hidden_layer_nn"]
            if hidden_layers is False:
                # Pass False as a parameter
                cmd.extend(["--hidden_layer_nn", "False"])
            elif isinstance(hidden_layers, tuple):
                # For tuple like (64, 64), pass as separate arguments like shell: --hidden_layer_nn 64 64
                cmd.append("--hidden_layer_nn")
                cmd.extend(str(x) for x in hidden_layers)
            else:
                # For single int like 64
                cmd.extend(["--hidden_layer_nn", str(hidden_layers)])
    
        # Add plot parameters if supported
        if algo_config["supports_plots"]:
            if run_config.plot_power_ylim:
                cmd.extend(
                    [
                        "--plot_power_ylim",
                        str(run_config.plot_power_ylim[0]),
                        str(run_config.plot_power_ylim[1]),
                    ]
                )
            if run_config.plot_load_ylim:
                cmd.extend(
                    [
                        "--plot_load_ylim",
                        str(run_config.plot_load_ylim[0]),
                        str(run_config.plot_load_ylim[1]),
                    ]
                )

        # Execute training
        try:
            # Debug: print command being executed
            if self.config.debug:
                print(f"🐛 Debug - Executing command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=not self.config.debug,  # Don't capture if debug (stream to terminal)
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            # Debug: print output
            if self.config.debug and result.stdout:
                print(f"🐛 Debug - STDOUT for {run_id}:\n{result.stdout}")
            if self.config.debug and result.stderr:
                print(f"🐛 Debug - STDERR for {run_id}:\n{result.stderr}")

            if result.returncode != 0:
                error_msg = f"Training failed with exit code {result.returncode}"
                print(f"❌ {error_msg}: {run_id}")
                if not self.config.debug:
                    print(f"STDERR: {result.stderr[-500:]}")  # Last 500 chars
                return {"status": "failed", "run_id": run_id, "error": error_msg}

            # Extract run path from stdout (models saved in /runs/{run_name}/)
            run_path = self._extract_run_path(result.stdout)

            duration = time.time() - start_time
            print(f"✅ Completed training: {run_id} ({duration:.1f}s)")

            # Organise run into sweep directory
            organised_path = self._organise_run(run_path, run_config)

            # Save progress
            self._save_progress(run_id)

            return {
                "status": "success",
                "run_id": run_id,
                "run_path": organised_path if organised_path else run_path,
                "duration": duration,
            }

        except subprocess.TimeoutExpired:
            print(f"⏱️  Training timeout: {run_id}")
            return {"status": "timeout", "run_id": run_id}
        except Exception as e:
            print(f"❌ Training error: {run_id} - {e}")
            return {"status": "error", "run_id": run_id, "error": str(e)}

    def _extract_run_path(self, stdout: str) -> Optional[str]:
        """Extract the run path from training output."""
        # Look for patterns like "runs/{run_name}" or "Saving to: ..."
        for line in stdout.split("\n"):
            if "runs/" in line:
                # Simple heuristic - extract path containing "runs/"
                parts = line.split()
                for part in parts:
                    if "runs/" in part:
                        path = part.strip(",").strip("'").strip('"')
                        # If path includes a filename, return just the directory
                        if path.endswith('.cleanrl_model') or path.endswith('.pt') or path.endswith('.pth'):
                            path = str(Path(path).parent)
                        return path
        return None

    def _organise_run(self, run_path: Optional[str], run_config: RunConfig, copy_only: bool = False) -> Optional[str]:
        """Move or copy a completed run directory into the sweep directory with a readable name.
        
        Args:
            run_path: Path to the run directory
            run_config: Configuration for this run
            copy_only: If True, copy instead of move (for existing runs from other sweeps)
        """
        if not run_path:
            return None
        
        import shutil
        
        source_path = Path(run_path)
        if not source_path.exists():
            print(f"⚠️  Warning: Run path not found for organisation: {run_path}")
            return None
        
        # Create runs subdirectory in sweep directory with algorithm folder
        runs_subdir = self.sweep_dir / "runs" / run_config.algorithm
        runs_subdir.mkdir(parents=True, exist_ok=True)
        
        # Create readable name: seed{seed}_el{episode_length}_tt{total_timesteps}
        readable_name = (
            f"seed{run_config.seed}_"
            f"el{run_config.episode_length}_"
            f"tt{run_config.total_timesteps}"
        )
        
        dest_path = runs_subdir / readable_name
        
        # Handle name collision (shouldn't happen but just in case)
        counter = 1
        while dest_path.exists():
            dest_path = runs_subdir / f"{readable_name}_{counter}"
            counter += 1
        
        try:
            if copy_only:
                shutil.copytree(str(source_path), str(dest_path))
                action = "Copied"
            else:
                shutil.move(str(source_path), str(dest_path))
                action = "Moved"
            
            # Use absolute paths to avoid relative_to issues
            dest_path_abs = dest_path.resolve()
            try:
                display_path = dest_path_abs.relative_to(Path.cwd().resolve())
            except ValueError:
                # If relative_to fails, just use the absolute path
                display_path = dest_path_abs
            print(f"📁 {action} run to: {display_path}")
            return str(dest_path)
        except Exception as e:
            print(f"⚠️  Warning: Could not {'copy' if copy_only else 'move'} run {run_config.get_run_id()}: {e}")
            return None

    def _find_existing_run_path(self, run_config: RunConfig) -> Optional[str]:
        """Find the run path for a previously completed training run.
        
        Searches in exact pattern: parameter_sweeps/{*}/runs/{algorithm}/seed{X}_el{Y}_tt{Z}
        
        Args:
            run_config: Configuration for the run to find
        
        Returns:
            Path to the run directory if found, None otherwise
        """
        # Build exact directory name pattern: seed{seed}_el{episode_length}_tt{total_timesteps}
        target_dir_name = (
            f"seed{run_config.seed}_"
            f"el{run_config.episode_length}_"
            f"tt{run_config.total_timesteps}"
        )
        
        # Search pattern: parameter_sweeps/{*}/runs/{algorithm}/{target_dir_name}
        parameter_sweeps_dir = Path("parameter_sweeps")
        if not parameter_sweeps_dir.exists():
            return None
        
        # Iterate through all sweep directories (most recent first)
        for sweep_dir in sorted(parameter_sweeps_dir.iterdir(), reverse=True):
            if not sweep_dir.is_dir():
                continue
            
            # Check exact path: parameter_sweeps/{sweep_dir}/runs/{algorithm}/{target_dir_name}
            target_path = sweep_dir / "runs" / run_config.algorithm / target_dir_name
            
            if target_path.exists() and target_path.is_dir():
                # Verify it has model files
                model_files = list(target_path.glob("*model*"))
                if model_files:
                    # Additional validation: check the path contains the algorithm name
                    # to avoid picking up wrong models
                    if run_config.algorithm in str(target_path):
                        print(f"   📂 Found model in: {target_path}")
                        print(f"   📂 Model files: {[f.name for f in model_files[:3]]}")  # Show first 3
                        return str(target_path)
                    else:
                        print(f"   ⚠️  Skipping {target_path} - algorithm mismatch")
        
        return None

    def run_evaluation(self, run_config: RunConfig, run_path: str) -> dict:
        """Execute evaluation for a trained model."""
        print(f"📊 Evaluating: {run_config.get_run_id()}")
        print(f"   Using model from: {run_path}")

        if not run_path or not Path(run_path).exists():
            return {
                "status": "error",
                "run_id": run_config.get_run_id(),
                "error": "Run path not found",
            }

        cmd = [
            "mpiexec",
            "-n",
            "1",
            "python",
            "algorithms/evaluate.py",
            "--seed",
            str(self.config.eval_seed),
            "--algorithm",
            run_config.algorithm,
            "--env_id",
            run_config.env_id,
            "--pretrained_models",
            run_path,
            "--episode_length",
            str(self.config.eval_episode_length),
            "--training_timesteps",
            str(run_config.total_timesteps),
            "--training_episode_length",
            str(run_config.episode_length),
            "--training_seed",
            str(run_config.seed),
        ]

        # Add hidden_layer_nn if specified in algorithm config
        algo_config = ALGORITHM_CONFIGS.get(run_config.algorithm)
        if algo_config and "hidden_layer_nn" in algo_config:
            hidden_layers = algo_config["hidden_layer_nn"]
            if hidden_layers is False:
                # Pass False as a parameter
                cmd.extend(["--hidden_layer_nn", "False"])
            elif isinstance(hidden_layers, tuple):
                # For tuple like (64, 64), pass as separate arguments like shell: --hidden_layer_nn 64 64
                cmd.append("--hidden_layer_nn")
                cmd.extend(str(x) for x in hidden_layers)
            else:
                # For single int like 64
                cmd.extend(["--hidden_layer_nn", str(hidden_layers)])
        
        # Add plot parameters
        if run_config.plot_power_ylim:
            cmd.extend(
                [
                    "--plot_power_ylim",
                    str(run_config.plot_power_ylim[0]),
                    str(run_config.plot_power_ylim[1]),
                ]
            )
        if run_config.plot_load_ylim:
            cmd.extend(
                [
                    "--plot_load_ylim",
                    str(run_config.plot_load_ylim[0]),
                    str(run_config.plot_load_ylim[1]),
                ]
            )

        try:
            # Debug: print command being executed
            if self.config.debug:
                print(f"🐛 Debug - Eval command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=not self.config.debug,  # Don't capture if debug (stream to terminal)
                text=True,
                timeout=600,  # 10 minute timeout
            )
            
            # Debug: print output
            if self.config.debug and result.stdout:
                print(f"🐛 Debug - Eval STDOUT for {run_config.get_run_id()}:\n{result.stdout}")
            if self.config.debug and result.stderr:
                print(f"🐛 Debug - Eval STDERR for {run_config.get_run_id()}:\n{result.stderr}")

            if result.returncode != 0:
                print(f"❌ Evaluation failed: {run_config.get_run_id()}")
                if not self.config.debug:
                    print(f"   Error: {result.stderr[-500:]}")  # Last 500 chars of stderr
                return {
                    "status": "failed",
                    "run_id": run_config.get_run_id(),
                    "error": f"Exit code {result.returncode}: {result.stderr[-200:]}",
                }

            print(f"✅ Evaluation complete: {run_config.get_run_id()}")
            return {"status": "success", "run_id": run_config.get_run_id()}

        except Exception as e:
            print(f"❌ Evaluation error: {run_config.get_run_id()} - {e}")
            return {
                "status": "error",
                "run_id": run_config.get_run_id(),
                "error": str(e),
            }

    def run_sweep(self):
        """Execute the entire parameter sweep."""
        print(f"\n{'='*60}")
        print(f"Starting Parameter Sweep")
        print(f"Output directory: {self.sweep_dir}")
        print(f"{'='*60}\n")

        # Generate all run configurations
        run_configs = self.generate_run_configs()
        total_runs = len(run_configs)
        print(f"Total runs to execute: {total_runs}")
        print(f"Algorithms: {', '.join(self.config.algorithms)}")
        print(f"Environment: {self.config.env_id}")
        print(
            f"Episode lengths: {', '.join(map(str, self.config.episode_lengths))}"
        )
        print(
            f"Total timesteps: {', '.join(map(str, self.config.total_timesteps))}"
        )
        print(f"Seeds: {', '.join(map(str, self.config.seeds))}")
        print(f"Max parallel workers: {self.config.max_workers}")
        print(f"Resume mode: {'enabled' if self.config.resume else 'disabled'}")
        print(f"\n{'='*60}\n")

        # Track results
        training_results = []
        evaluation_results = []

        # Execute training runs (parallel)
        print("Phase 1: Training")
        print("-" * 60)
        training_results_dict = {}  # Map run_id to result
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_config = {}
            for i, config in enumerate(run_configs):
                # Add 1 second delay between submitting each job to ensure unique timestamps
                if i > 0:
                    time.sleep(1)
                future_to_config[executor.submit(self.run_training, config)] = config

            for future in as_completed(future_to_config):
                config = future_to_config[future]
                try:
                    result = future.result()
                    training_results_dict[config.get_run_id()] = result
                except Exception as e:
                    print(f"❌ Unexpected error for {config.get_run_id()}: {e}")
                    training_results_dict[config.get_run_id()] = {
                        "status": "error",
                        "run_id": config.get_run_id(),
                        "error": str(e),
                    }
        
        # Build training_results in the same order as run_configs
        training_results = [training_results_dict[config.get_run_id()] for config in run_configs]

        # Execute evaluations (parallel)
        print("\nPhase 2: Evaluation")
        print("-" * 60)
        
        # Prepare evaluation tasks
        eval_tasks = []
        for result, config in zip(training_results, run_configs):
            if result["status"] == "success" and result.get("run_path"):
                eval_tasks.append((config, result["run_path"]))
            elif result["status"] == "skipped":
                # For skipped runs, try to find the model path from the runs directory
                run_path = self._find_existing_run_path(config)
                if run_path:
                    print(f"🔍 Found existing model for skipped run: {config.get_run_id()}")
                    
                    # Check if model is already in the current sweep directory
                    run_path_obj = Path(run_path).resolve()
                    sweep_dir_obj = self.sweep_dir.resolve()
                    
                    if sweep_dir_obj in run_path_obj.parents:
                        # Already in sweep directory, use as-is
                        print(f"✓ Model already in sweep directory")
                        eval_tasks.append((config, run_path))
                    else:
                        # Model is external (different sweep or main runs/), copy it
                        print(f"📋 Copying external model to sweep directory...")
                        organised_path = self._organise_run(run_path, config, copy_only=True)
                        if organised_path:
                            eval_tasks.append((config, organised_path))
                        else:
                            print(f"⚠️  Could not copy model, using original path")
                            eval_tasks.append((config, run_path))
                else:
                    print(f"⚠️  Could not find model for skipped run: {config.get_run_id()}")
                    evaluation_results.append(
                        {"status": "model_not_found", "run_id": result["run_id"]}
                    )
            else:
                evaluation_results.append(
                    {
                        "status": "skipped_due_to_training_failure",
                        "run_id": result["run_id"],
                    }
                )
        
        # Run evaluations in parallel
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_task = {
                executor.submit(self.run_evaluation, config, run_path): (config, run_path)
                for config, run_path in eval_tasks
            }
            
            for future in as_completed(future_to_task):
                config, run_path = future_to_task[future]
                try:
                    eval_result = future.result()
                    evaluation_results.append(eval_result)
                except Exception as e:
                    print(f"❌ Unexpected evaluation error for {config.get_run_id()}: {e}")
                    evaluation_results.append(
                        {
                            "status": "error",
                            "run_id": config.get_run_id(),
                            "error": str(e),
                        }
                    )

        # Save summary
        self._save_summary(training_results, evaluation_results)

        # Print final summary
        self._print_summary(training_results, evaluation_results)
        
        # Auto-shutdown if requested
        if self.config.auto_shutdown:
            self._schedule_shutdown()

    def _save_summary(self, training_results: list, evaluation_results: list):
        """Save sweep results summary to JSON."""
        config_dict = asdict(self.config)
        # Convert Path objects to strings for JSON serialization
        if config_dict.get("output_dir"):
            config_dict["output_dir"] = str(config_dict["output_dir"])
        
        summary = {
            "sweep_config": config_dict,
            "timestamp": self.timestamp,
            "training_results": training_results,
            "evaluation_results": evaluation_results,
            "statistics": {
                "total_runs": len(training_results),
                "successful_training": sum(
                    1 for r in training_results if r["status"] == "success"
                ),
                "failed_training": sum(
                    1 for r in training_results if r["status"] == "failed"
                ),
                "skipped_training": sum(
                    1 for r in training_results if r["status"] == "skipped"
                ),
                "successful_evaluation": sum(
                    1 for r in evaluation_results if r["status"] == "success"
                ),
            },
        }

        summary_file = self.sweep_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n📝 Summary saved to: {summary_file}")

    def _print_summary(self, training_results: list, evaluation_results: list):
        """Print final summary statistics."""
        print(f"\n{'='*60}")
        print("Parameter Sweep Complete!")
        print(f"{'='*60}")
        print(f"Total runs: {len(training_results)}")
        print(
            f"Successful training: {sum(1 for r in training_results if r['status'] == 'success')}"
        )
        print(
            f"Failed training: {sum(1 for r in training_results if r['status'] == 'failed')}"
        )
        print(
            f"Skipped training: {sum(1 for r in training_results if r['status'] == 'skipped')}"
        )
        print(
            f"Successful evaluation: {sum(1 for r in evaluation_results if r['status'] == 'success')}"
        )
        print(f"\nResults saved to: {self.sweep_dir}")
        print(f"{'='*60}\n")

    def _schedule_shutdown(self):
        """Schedule system shutdown after a delay."""
        import subprocess
        
        delay_minutes = self.config.shutdown_delay_minutes
        print(f"\n{'='*60}")
        print(f"🔴 AUTO-SHUTDOWN SCHEDULED")
        print(f"{'='*60}")
        print(f"System will shutdown in {delay_minutes} minutes")
        print(f"To cancel, run: sudo shutdown -c")
        print(f"{'='*60}\n")
        
        try:
            # Schedule shutdown using the system shutdown command
            subprocess.run(
                ["sudo", "shutdown", "-h", f"+{delay_minutes}"],
                check=True,
                capture_output=True,
                text=True
            )
            print(f"✅ Shutdown scheduled successfully for +{delay_minutes} minutes")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Warning: Could not schedule shutdown: {e.stderr}")
            print(f"   You may need to configure passwordless sudo for shutdown.")
            print(f"   See scripts/shutdown.sh for setup instructions.")
        except Exception as e:
            print(f"⚠️  Warning: Unexpected error scheduling shutdown: {e}")


# =============================================================================
# CLI INTERFACE
# =============================================================================


def main(
    algorithms: list[str] = DEFAULT_ALGORITHMS,
    env_id: str = DEFAULT_ENV_ID,
    episode_lengths: list[int] = DEFAULT_EPISODE_LENGTHS,
    total_timesteps: list[int] = DEFAULT_TOTAL_TIMESTEPS,
    seeds: list[int] = DEFAULT_SEEDS,
    plot_power_ylim: Optional[tuple[float, float]] = DEFAULT_PLOT_POWER_YLIM,
    plot_load_ylim: Optional[tuple[float, float]] = DEFAULT_PLOT_LOAD_YLIM,
    eval_episode_length: int = DEFAULT_EVAL_EPISODE_LENGTH,
    eval_seed: int = DEFAULT_EVAL_SEED,
    max_workers: int = DEFAULT_MAX_WORKERS,
    resume: bool = True,
    output_dir: Optional[str] = None,
    auto_shutdown: bool = False,
    shutdown_delay_minutes: int = 5,
):
    """
    Run a comprehensive parameter sweep for WFCRL algorithms.

    Args:
        algorithms: List of algorithms to run. Default: ippo, mappo, ifac.
                   Options: ippo, mappo, ifac, ifppo, idqn, idrqn, qmix
        env_id: Environment ID (e.g., Dec_Ablaincourt_Floris, Dec_Turb3_Row1_Floris)
        episode_lengths: List of episode lengths to sweep over
        total_timesteps: List of total timesteps to sweep over
        seeds: List of random seeds to use
        plot_power_ylim: Y-axis limits for power plots (min, max)
        plot_load_ylim: Y-axis limits for load plots (min, max)
        eval_episode_length: Episode length for evaluation
        eval_seed: Random seed for evaluation
        max_workers: Maximum number of parallel training processes
        resume: Enable resume capability (skip completed runs and reuse existing models)
        output_dir: Path to sweep directory. If exists, will RESUME that sweep.
                   If new, creates sweep there. If None, creates new timestamped directory.
        auto_shutdown: Automatically shutdown the system after sweep completes (default: False)
        shutdown_delay_minutes: Minutes to wait before shutting down (default: 5, allows time to cancel)

    Examples:
        # Start a new sweep (creates parameter_sweeps/<timestamp>_parameter_sweep/)
        python parameter_sweep_v2.py --algorithms ippo mappo

        # Resume an interrupted sweep (continues from progress.json)
        python parameter_sweep_v2.py \\
            --output_dir parameter_sweeps/10-00-00_16-10-25_parameter_sweep

        # Long-running sweep with auto-shutdown
        python parameter_sweep_v2.py \\
            --algorithms ippo mappo idqn idrqn qmix \\
            --total_timesteps 50000 100000 \\
            --seeds 1 2 3 4 5 \\
            --auto_shutdown True \\
            --shutdown_delay_minutes 10

        # Multiple algorithms with custom parameters
        python parameter_sweep_v2.py --algorithms ippo mappo ifac \\
            --env_id Dec_Turb3_Row1_Floris \\
            --episode_lengths 100 200 300 \\
            --total_timesteps 10000 50000 \\
            --seeds 1 2 3 4 5

        # Disable resume to force re-training even if models exist
        python parameter_sweep_v2.py --algorithms ifppo \\
            --resume False
    
    Note on auto-shutdown:
        Requires passwordless sudo for shutdown command. To set up:
        1. Run: sudo visudo
        2. Add line: your_username ALL=(ALL) NOPASSWD: /sbin/shutdown
        3. Save and exit
        To cancel shutdown: sudo shutdown -c
    """
    # Validate algorithms
    valid_algos = set(ALGORITHM_CONFIGS.keys())
    invalid_algos = [a for a in algorithms if a not in valid_algos]
    if invalid_algos:
        raise ValueError(
            f"Invalid algorithms: {invalid_algos}. Valid options: {sorted(valid_algos)}"
        )

    # Create sweep configuration
    config = SweepConfig(
        algorithms=algorithms,
        env_id=env_id,
        episode_lengths=episode_lengths,
        total_timesteps=total_timesteps,
        seeds=seeds,
        plot_power_ylim=plot_power_ylim,
        plot_load_ylim=plot_load_ylim,
        eval_episode_length=eval_episode_length,
        eval_seed=eval_seed,
        max_workers=max_workers,
        resume=resume,
        output_dir=Path(output_dir) if output_dir else None,
        auto_shutdown=auto_shutdown,
        shutdown_delay_minutes=shutdown_delay_minutes,
    )

    # Run the sweep
    sweep = ParameterSweep(config)
    sweep.run_sweep()


if __name__ == "__main__":
    tyro.cli(main)
