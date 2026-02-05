#!/usr/bin/env python3
"""
Comprehensive Parameter Sweep Script for WFCRL Benchmark
"""

import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import tyro

# Get script directory and project root
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()

# Add parent directory to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

# Import configuration from separate config file
from scripts.sweep_config import (
    ALGORITHMS,
    ALGORITHM_CONFIGS,
    AUTO_SHUTDOWN,
    ENV_ID,
    ENVIRONMENTS,
    EPISODE_LENGTHS,
    EVAL_EPISODE_LENGTH,
    EVAL_SEED,
    EVALUATION,
    FILTER_OUTPUT,
    FILTER_PHRASES,
    MAX_WORKERS,
    PLOT_LOAD_YLIM,
    PLOT_POWER_YLIM,
    RESUME,
    SEEDS,
    SHUTDOWN_DELAY_MINUTES,
    TESTING,
    TOTAL_ITERATIONS,
    USE_WINDROSE,
    WIND_DIRECTION,
    WIND_SPEED,
    WINDROSE_DATA_PATH,
    WINDROSE_NUM_BINS,
    VTK_WIND,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class RunConfig:
    """Configuration for a single training run."""

    algorithm: str
    env_id: str
    episode_length: int
    total_iterations: int
    seed: int
    wind_speed: float = 8
    wind_direction: float = 270
    plot_power_ylim: Optional[tuple[float, float]] = None
    plot_load_ylim: Optional[tuple[float, float]] = None
    vtk_wind: bool = False

    def get_run_id(self) -> str:
        """Generate unique identifier for this run."""
        if ALGORITHM_CONFIGS[self.algorithm].get("supports_episode_length", True):
            return f"{self.algorithm}_{self.env_id}_el{self.episode_length}_ti{self.total_iterations}_s{self.seed}"
        else:
            return f"{self.algorithm}_{self.env_id}_ti{self.total_iterations}_s{self.seed}"
    
    def get_eval_run_id(self, eval_env_id: str) -> str:
        """Generate unique identifier for an evaluation run.
        
        Args:
            eval_env_id: Environment ID used for evaluation
        
        Returns:
            Unique ID that includes the evaluation environment
        """
        if ALGORITHM_CONFIGS[self.algorithm].get("supports_episode_length", True):
            return f"{self.algorithm}_{self.env_id}_el{self.episode_length}_ti{self.total_iterations}_s{self.seed}_eval_{eval_env_id}"
        else:
            return f"{self.algorithm}_{self.env_id}_ti{self.total_iterations}_s{self.seed}_eval_{eval_env_id}"


@dataclass
class SweepConfig:
    """Configuration for the entire parameter sweep."""

    algorithms: list[str]
    env_id: str
    episode_lengths: list[int] = field(default_factory=lambda: EPISODE_LENGTHS)
    total_iterations: list[int] = field(default_factory=lambda: TOTAL_ITERATIONS)
    seeds: list[int] = field(default_factory=lambda: SEEDS)
    wind_speed: float = WIND_SPEED
    wind_direction: float = WIND_DIRECTION
    plot_power_ylim: Optional[tuple[float, float]] = PLOT_POWER_YLIM
    plot_load_ylim: Optional[tuple[float, float]] = PLOT_LOAD_YLIM
    vtk_wind: bool = VTK_WIND
    """Generate VTK wind field files during evaluation for FAST.Farm ParaView visualization"""
    eval_episode_length: int = EVAL_EPISODE_LENGTH
    eval_seed: int = EVAL_SEED
    evaluation: str = EVALUATION
    """Evaluation mode: "floris" (FLORIS only), "fastfarm" (FAST.Farm only), or "both" (both simulators)"""
    use_windrose: bool = USE_WINDROSE
    """If True, use windrose for both training and evaluation"""
    windrose_data_path: str = WINDROSE_DATA_PATH
    """Path to wind data CSV file for windrose"""
    windrose_num_bins: int = WINDROSE_NUM_BINS
    """Number of bins for windrose evaluation"""
    max_workers: int = MAX_WORKERS
    resume: bool = RESUME
    """If True, skip completed runs and reuse existing models; if False, train everything from scratch"""
    output_dir: Optional[Path] = None
    testing: bool = TESTING
    """If True, prepend 'TEST_' to sweep directory name"""
    filter_output: bool = FILTER_OUTPUT
    """If True, filter unwanted console messages during evaluation"""
    filter_phrases: list[str] = field(default_factory=lambda: FILTER_PHRASES)
    """List of phrases to filter from evaluation output"""
    auto_shutdown: bool = AUTO_SHUTDOWN
    """Automatically shutdown the system after sweep completes"""
    shutdown_delay_minutes: int = SHUTDOWN_DELAY_MINUTES
    """Minutes to wait before shutting down (allows time to cancel if needed)"""


# =============================================================================
# MAIN PARAMETER SWEEP CLASS
# =============================================================================


class ParameterSweep:
    """Manages parameter sweep execution with resume capability."""

    def __init__(self, config: SweepConfig):
        self.config = config
        
        # Convert evaluation mode to environment IDs
        self.eval_env_ids = self._get_eval_env_ids()
        
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
                    self.timestamp = saved_config.get("timestamp", datetime.now().strftime("%d-%m-%y______%H-%M-%S______"))
                print(f"📂 Resuming existing sweep from: {self.sweep_dir}")
            else:
                # New sweep in custom directory
                self.timestamp = datetime.now().strftime("%d-%m-%y______%H-%M-%S______")
        else:
            # No directory specified - create new one with timestamp
            self.timestamp = datetime.now().strftime("%d-%m-%y______%H-%M-%S______")
            prefix = "TEST_" if config.testing else ""
            self.sweep_dir = PROJECT_ROOT / "parameter_sweeps" / f"{prefix}{self.timestamp}_parameter_sweep"
            self.sweep_dir.mkdir(parents=True, exist_ok=True)

        # Progress tracking
        self.progress_file = self.sweep_dir / "progress.json"
        self.completed_runs, self.completed_evaluations = self._load_progress()

        # Save sweep configuration
        self._save_config()

    def _get_eval_env_ids(self) -> list[str]:
        """Convert evaluation mode string to list of environment IDs.
        
        Returns:
            List of environment IDs to evaluate on
        """
        eval_mode = self.config.evaluation.lower()
        training_env = self.config.env_id
        
        # Determine the base environment name (without _Floris or _Fastfarm suffix)
        if "_Floris" in training_env:
            base_env = training_env.replace("_Floris", "")
        elif "_Fastfarm" in training_env:
            base_env = training_env.replace("_Fastfarm", "")
        else:
            # Assume it's a base name already
            base_env = training_env
        
        # Convert evaluation mode to environment IDs
        if eval_mode == "floris":
            return [f"{base_env}_Floris"]
        elif eval_mode == "fastfarm":
            return [f"{base_env}_Fastfarm"]
        elif eval_mode == "both":
            return [f"{base_env}_Floris", f"{base_env}_Fastfarm"]
        else:
            raise ValueError(
                f"Invalid evaluation mode: '{self.config.evaluation}'. "
                "Must be 'floris', 'fastfarm', or 'both'"
            )

    def _load_progress(self) -> tuple[set[str], set[str]]:
        """Load completed runs and evaluations from progress file.
        
        Returns:
            Tuple of (completed_runs, completed_evaluations) as sets of run_ids
        """
        if not self.progress_file.exists():
            return set(), set()

        try:
            with open(self.progress_file, "r") as f:
                data = json.load(f)
                completed_runs = set(data.get("completed_runs", []))
                completed_evaluations = set(data.get("completed_evaluations", []))
                return completed_runs, completed_evaluations
        except (json.JSONDecodeError, IOError):
            print(f"Warning: Could not load progress from {self.progress_file}")
            return set(), set()

    def _save_progress(self, run_id: str, progress_type: str = "training"):
        """Save progress after completing a training run or evaluation with file locking.
        
        Args:
            run_id: Unique identifier for the run
            progress_type: Either "training" or "evaluation"
        """
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
                        completed_evaluations = set(data.get("completed_evaluations", []))
                else:
                    completed_runs = set()
                    completed_evaluations = set()
                
                # Add new run or evaluation
                if progress_type == "training":
                    completed_runs.add(run_id)
                    self.completed_runs = completed_runs
                elif progress_type == "evaluation":
                    completed_evaluations.add(run_id)
                    self.completed_evaluations = completed_evaluations
                else:
                    raise ValueError(f"Invalid progress_type: {progress_type}. Must be 'training' or 'evaluation'")
                
                # Save updated progress
                data = {
                    "completed_runs": list(completed_runs),
                    "completed_evaluations": list(completed_evaluations),
                    "last_updated": datetime.now().isoformat(),
                }
                with open(self.progress_file, "w") as f:
                    json.dump(data, f, indent=2)
                
            finally:
                # Lock is automatically released when the with block exits
                pass

    def _save_config(self):
        """Save sweep configuration to JSON file."""
        config_file = self.sweep_dir / "sweep_config.json"
        config_dict = asdict(self.config)
        # Convert Path objects to strings for JSON serialisation
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
                for total_iterations in self.config.total_iterations:
                    for seed in self.config.seeds:
                        run_config = RunConfig(
                            algorithm=algo,
                            env_id=self.config.env_id,
                            episode_length=episode_length,
                            total_iterations=total_iterations,
                            seed=seed,
                            wind_speed=self.config.wind_speed,
                            wind_direction=self.config.wind_direction,
                            plot_power_ylim=self.config.plot_power_ylim,
                            plot_load_ylim=self.config.plot_load_ylim,
                            vtk_wind=self.config.vtk_wind,
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

        # Check if a model already exists in previous sweeps (only if resume is enabled)
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

        # Calculate output directory - files will be created directly here
        # Create runs subdirectory in sweep directory with algorithm folder
        runs_subdir = self.sweep_dir / "runs" / run_config.algorithm
        runs_subdir.mkdir(parents=True, exist_ok=True)
        
        # Create readable name: seed{seed}_el{episode_length}_ti{total_iterations}
        readable_name = (
            f"seed{run_config.seed}_"
            f"el{run_config.episode_length}_"
            f"ti{run_config.total_iterations}"
        )
        output_dir = runs_subdir / readable_name

        # Get algorithm script path
        algo_config = ALGORITHM_CONFIGS.get(run_config.algorithm)
        if not algo_config:
            return {
                "status": "error",
                "run_id": run_id,
                "error": f"Unknown algorithm: {run_config.algorithm}",
            }

        script_path = PROJECT_ROOT / algo_config["script"]
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
            "--total_iterations",
            str(run_config.total_iterations),
            "--seed",
            str(run_config.seed),
            "--wind_speed",
            str(run_config.wind_speed),
            "--wind_direction",
            str(run_config.wind_direction),
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
        
        # Add output directory to create files directly in sweep directory
        cmd.extend(["--output_dir", str(output_dir)])
        
        # Add windrose parameters for training if enabled
        if self.config.use_windrose:
            cmd.extend(["--scenario", "windrose"])
            cmd.extend(["--wind_data", str(PROJECT_ROOT / self.config.windrose_data_path)])

        # Execute training
        try:
            result = subprocess.run(
                cmd,
                capture_output=False,  # Stream output to terminal
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode != 0:
                error_msg = f"Training failed with exit code {result.returncode}"
                print(f"❌ {error_msg}: {run_id}")
                return {"status": "failed", "run_id": run_id, "error": error_msg}

            # Verify the output directory was created
            run_path = str(output_dir)
            if not output_dir.exists():
                print(f"⚠️  Warning: Expected output directory does not exist: {output_dir}")
                run_path = None

            duration = time.time() - start_time
            print(f"✅ Completed training: {run_id} ({duration:.1f}s)")

            # Save progress
            self._save_progress(run_id)

            return {
                "status": "success",
                "run_id": run_id,
                "run_path": run_path,
                "duration": duration,
            }

        except subprocess.TimeoutExpired:
            print(f"⏱️  Training timeout: {run_id}")
            return {"status": "timeout", "run_id": run_id}
        except Exception as e:
            print(f"❌ Training error: {run_id} - {e}")
            return {"status": "error", "run_id": run_id, "error": str(e)}

    def _organise_run(self, run_path: Optional[str], run_config: RunConfig, copy_only: bool = False) -> Optional[str]:
        """Copy a completed run directory into the sweep directory with a readable name.
        
        Note: New training runs are created directly in the sweep directory via --output_dir.
        This method is now primarily used for copying existing models from other sweeps.
        
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
        
        # Create readable name: seed{seed}_el{episode_length}_ti{total_iterations}
        readable_name = (
            f"seed{run_config.seed}_"
            f"el{run_config.episode_length}_"
            f"ti{run_config.total_iterations}"
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
        # Build exact directory name pattern: seed{seed}_el{episode_length}_ti{total_iterations}
        target_dir_name = (
            f"seed{run_config.seed}_"
            f"el{run_config.episode_length}_"
            f"ti{run_config.total_iterations}"
        )
        
        # Search pattern: parameter_sweeps/{*}/runs/{algorithm}/{target_dir_name}
        parameter_sweeps_dir = PROJECT_ROOT / "parameter_sweeps"
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
                        return str(target_path)
                    else:
                        print(f"   ⚠️  Skipping {target_path} - algorithm mismatch")
        
        return None

    def _check_and_copy_existing_evaluation(self, run_path: str, eval_env_id: str) -> bool:
        """Check if evaluation already exists in the run path and is complete.
        
        Evaluations are stored in subdirectories like 'floris_evaluation' or 'fastfarm_evaluation'
        within the model directory. This method checks if the evaluation exists and has results.
        
        Args:
            run_path: Path to the trained model directory
            eval_env_id: Environment ID being evaluated (e.g., Dec_Ablaincourt_Floris)
        
        Returns:
            True if evaluation exists and is complete, False otherwise
        """
        # Determine evaluation subdirectory name based on environment
        if "Floris" in eval_env_id:
            eval_subdir = "floris_evaluation"
        elif "Fastfarm" in eval_env_id:
            eval_subdir = "fastfarm_evaluation"
        else:
            # Unknown environment type, can't determine subdirectory
            return False
        
        eval_path = Path(run_path) / eval_subdir
        
        # Check if evaluation directory exists
        if not eval_path.exists():
            return False
        
        # Check for results files (windrose_scores.csv or other evaluation outputs)
        result_files = list(eval_path.glob("*.csv")) + list(eval_path.glob("*.png"))
        
        if result_files:
            print(f"   ✅ Found existing evaluation results in: {eval_path}")
            print(f"   📊 Result files: {[f.name for f in result_files[:5]]}")  # Show first 5
            return True
        
        return False

    def run_evaluation(self, run_config: RunConfig, run_path: str, eval_env_id: Optional[str] = None) -> dict:
        """Execute evaluation for a trained model.
        
        Args:
            run_config: Configuration for the training run
            run_path: Path to the trained model
            eval_env_id: Environment ID to evaluate on (if None, uses training env_id)
        
        Returns:
            Dictionary with evaluation results
        """
        # Use training env_id if eval_env_id not specified
        if eval_env_id is None:
            eval_env_id = run_config.env_id
        
        eval_run_id = run_config.get_eval_run_id(eval_env_id)
        
        print(f"📊 Evaluating: {eval_run_id}")
        print(f"   Training env: {run_config.env_id}")
        print(f"   Evaluation env: {eval_env_id}")
        print(f"   Using model from: {run_path}")
        
        # Record start time for filtering FAST.Farm files
        import time
        eval_start_time = time.time()

        if not run_path or not Path(run_path).exists():
            return {
                "status": "error",
                "run_id": eval_run_id,
                "error": "Run path not found",
            }
        
        # Check if evaluation already exists in the model directory
        if self._check_and_copy_existing_evaluation(run_path, eval_env_id):
            print(f"   ⏭️  Using existing evaluation results")
            # Mark as completed in progress
            self._save_progress(eval_run_id, progress_type="evaluation")
            return {
                "status": "success",
                "run_id": eval_run_id,
                "eval_env_id": eval_env_id,
                "reused": True,
            }

        cmd = [
            "python",
            str(PROJECT_ROOT / "algorithms" / "evaluate.py"),
            "--seed",
            str(self.config.eval_seed),
            "--algorithm",
            run_config.algorithm,
            "--env_id",
            eval_env_id,  # Use eval environment
            "--pretrained_models",
            run_path,
            "--episode_length",
            str(self.config.eval_episode_length),
            "--training_iterations",
            str(run_config.total_iterations),
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
        
        # Add VTK wind generation if enabled
        if run_config.vtk_wind:
            cmd.append("--vtk_wind")
        
        # Add windrose parameters for evaluation if enabled
        if self.config.use_windrose:
            cmd.extend(["--scenario", "windrose"])
            cmd.extend(["--wind_data", str(PROJECT_ROOT / self.config.windrose_data_path)])
        
        # For FAST.Farm evaluations, specify output directory directly
        # This avoids needing to move files after evaluation
        fastfarm_output_dir = None
        if "Fastfarm" in eval_env_id:
            fastfarm_output_dir = str(Path(run_path) / "fastfarm_evaluation")
            cmd.extend(["--fastfarm_output_dir", fastfarm_output_dir])

        try:
            # When filtering is on, capture and filter, otherwise just stream
            if self.config.filter_output:
                # Use Popen with real-time filtering
                import sys
                import threading
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
                def read_output():
                    last_line_blank = False
                    try:
                        for line in iter(process.stdout.readline, ''):
                            if not line:
                                break
                            # Filter and print line by line
                            if not any(phrase in line for phrase in self.config.filter_phrases):
                                # Skip consecutive blank lines
                                is_blank = line.strip() == ''
                                if not (is_blank and last_line_blank):
                                    print(line, end='')
                                    sys.stdout.flush()
                                last_line_blank = is_blank
                    finally:
                        process.stdout.close()
                
                # Read output in a thread to prevent blocking
                reader_thread = threading.Thread(target=read_output)
                reader_thread.daemon = True
                reader_thread.start()
                
                # Wait for process to complete
                try:
                    return_code = process.wait(timeout=7200)
                    reader_thread.join(timeout=5)  # Give reader thread time to finish
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    reader_thread.join(timeout=5)
                    print(f"⏱️  Evaluation timeout: {eval_run_id}")
                    return {
                        "status": "failed",
                        "run_id": eval_run_id,
                        "eval_env_id": eval_env_id,
                        "error": "Evaluation timed out after 7200 seconds (2 hours)",
                    }
                
                if return_code != 0:
                    print(f"❌ Evaluation failed: {eval_run_id}")
                    return {
                        "status": "failed",
                        "run_id": eval_run_id,
                        "eval_env_id": eval_env_id,
                        "error": f"Exit code {return_code}",
                    }
            else:
                # Filtering off - just stream everything
                result = subprocess.run(
                    cmd,
                    capture_output=False,
                    text=True,
                    timeout=7200,
                )
                
                if result.returncode != 0:
                    print(f"❌ Evaluation failed: {eval_run_id}")
                    return {
                        "status": "failed",
                        "run_id": eval_run_id,
                        "eval_env_id": eval_env_id,
                        "error": f"Exit code {result.returncode}",
                    }

            # Wait for FAST.Farm to finish writing files after MPI communication ends
            # This delay ensures all output files (including VTK) are fully written
            if "Fastfarm" in eval_env_id:
                time.sleep(12.0)
            
            print(f"✅ Evaluation complete: {eval_run_id}")
            
            # Save evaluation progress
            self._save_progress(eval_run_id, progress_type="evaluation")
            
            return {
                "status": "success",
                "run_id": eval_run_id,
                "eval_env_id": eval_env_id,
            }

        except subprocess.TimeoutExpired:
            print(f"⏱️  Evaluation timeout: {eval_run_id}")
            return {
                "status": "failed",
                "run_id": eval_run_id,
                "eval_env_id": eval_env_id,
                "error": "Evaluation timed out after 7200 seconds (2 hours)",
            }
        except Exception as e:
            print(f"❌ Evaluation error: {eval_run_id} - {e}")
            return {
                "status": "error",
                "run_id": eval_run_id,
                "eval_env_id": eval_env_id,
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
            f"Total iterations: {', '.join(map(str, self.config.total_iterations))}"
        )
        print(f"Seeds: {', '.join(map(str, self.config.seeds))}")
        print(f"Max parallel workers: {self.config.max_workers}")
        print(f"Resume mode: {'enabled' if self.config.resume else 'disabled'}")
        print(f"Evaluation mode: {self.config.evaluation}")
        print(f"Windrose mode: {'enabled' if self.config.use_windrose else 'disabled'}")
        if self.config.use_windrose:
            print(f"  - Wind data: {self.config.windrose_data_path}")
            print(f"  - Number of bins: {self.config.windrose_num_bins}")
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
        
        # Determine evaluation environments
        eval_env_ids = self.eval_env_ids
        print(f"Evaluation mode: {self.config.evaluation}")
        print(f"Evaluation environments: {', '.join(eval_env_ids)}")
        
        # Prepare evaluation tasks
        eval_tasks = []
        for result, config in zip(training_results, run_configs):
            training_run_id = config.get_run_id()
            
            # Get the run path (either from successful training or from finding existing model)
            run_path = None
            if result["status"] == "success" and result.get("run_path"):
                run_path = result["run_path"]
            elif result["status"] == "skipped":
                # For skipped runs, try to find the model path (only if resume enabled)
                if self.config.resume:
                    run_path = self._find_existing_run_path(config)
                    if run_path:
                        print(f"🔍 Found existing model for skipped run: {config.get_run_id()}")
                        
                        # Check if model is already in the current sweep directory
                        run_path_obj = Path(run_path).resolve()
                        sweep_dir_obj = self.sweep_dir.resolve()
                        
                        if sweep_dir_obj in run_path_obj.parents:
                            # Already in sweep directory, use as-is
                            print(f"✓ Model already in sweep directory")
                        else:
                            # Model is external (different sweep or main runs/), copy it
                            print(f"📋 Copying external model to sweep directory...")
                            organised_path = self._organise_run(run_path, config, copy_only=True)
                            if organised_path:
                                run_path = organised_path
                            else:
                                print(f"⚠️  Could not copy model, using original path")
                else:
                    # resume is False, don't search for external models
                    run_path = None
            
            # If we have a run path, create evaluation tasks for each eval environment
            if run_path:
                for eval_env_id in eval_env_ids:
                    eval_run_id = config.get_eval_run_id(eval_env_id)
                    
                    # Check if this specific evaluation already completed (in progress.json)
                    if self.config.resume and eval_run_id in self.completed_evaluations:
                        print(f"⏭️  Skipping completed evaluation: {eval_run_id}")
                        evaluation_results.append({
                            "status": "skipped",
                            "run_id": eval_run_id,
                            "eval_env_id": eval_env_id,
                        })
                        continue
                    
                    # Check if evaluation already exists in the model directory
                    if self.config.resume and self._check_and_copy_existing_evaluation(run_path, eval_env_id):
                        print(f"⏭️  Found existing evaluation, marking as complete: {eval_run_id}")
                        self._save_progress(eval_run_id, progress_type="evaluation")
                        evaluation_results.append({
                            "status": "success",
                            "run_id": eval_run_id,
                            "eval_env_id": eval_env_id,
                            "reused": True,
                        })
                        continue
                    
                    # Add to evaluation tasks
                    eval_tasks.append((config, run_path, eval_env_id))
            else:
                # No model found
                if result["status"] == "skipped":
                    print(f"⚠️  Could not find model for skipped run: {training_run_id}")
                    for eval_env_id in eval_env_ids:
                        evaluation_results.append({
                            "status": "model_not_found",
                            "run_id": config.get_eval_run_id(eval_env_id),
                            "eval_env_id": eval_env_id,
                        })
                else:
                    # Training failed
                    for eval_env_id in eval_env_ids:
                        evaluation_results.append({
                            "status": "skipped_due_to_training_failure",
                            "run_id": config.get_eval_run_id(eval_env_id),
                            "eval_env_id": eval_env_id,
                        })
        
        # Run evaluations in parallel
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_task = {
                executor.submit(self.run_evaluation, config, run_path, eval_env_id): (config, run_path, eval_env_id)
                for config, run_path, eval_env_id in eval_tasks
            }
            
            for future in as_completed(future_to_task):
                config, run_path, eval_env_id = future_to_task[future]
                try:
                    eval_result = future.result()
                    evaluation_results.append(eval_result)
                except Exception as e:
                    eval_run_id = config.get_eval_run_id(eval_env_id)
                    print(f"❌ Unexpected evaluation error for {eval_run_id}: {e}")
                    evaluation_results.append({
                        "status": "error",
                        "run_id": eval_run_id,
                        "eval_env_id": eval_env_id,
                        "error": str(e),
                    })

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
                    1 for r in training_results if r["status"] in ["failed", "error"]
                ),
                "skipped_training": sum(
                    1 for r in training_results if r["status"] == "skipped"
                ),
                "successful_evaluation": sum(
                    1 for r in evaluation_results if r["status"] == "success"
                ),
                "skipped_evaluation": sum(
                    1 for r in evaluation_results if r["status"] == "skipped"
                ),
                "failed_evaluation": sum(
                    1 for r in evaluation_results if r["status"] in ["failed", "error"]
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
            f"Failed training: {sum(1 for r in training_results if r['status'] in ['failed', 'error'])}"
        )
        print(
            f"Skipped training: {sum(1 for r in training_results if r['status'] == 'skipped')}"
        )
        print(
            f"Successful evaluation: {sum(1 for r in evaluation_results if r['status'] == 'success')}"
        )
        print(
            f"Skipped evaluation: {sum(1 for r in evaluation_results if r['status'] == 'skipped')}"
        )
        print(
            f"Failed evaluation: {sum(1 for r in evaluation_results if r['status'] in ['failed', 'error'])}"
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
    algorithms: list[str] = ALGORITHMS,
    env_id: str = ENV_ID,
    episode_lengths: list[int] = EPISODE_LENGTHS,
    total_iterations: list[int] = TOTAL_ITERATIONS,
    seeds: list[int] = SEEDS,
    wind_speed: float = WIND_SPEED,
    wind_direction: float = WIND_DIRECTION,
    plot_power_ylim: Optional[tuple[float, float]] = PLOT_POWER_YLIM,
    plot_load_ylim: Optional[tuple[float, float]] = PLOT_LOAD_YLIM,
    vtk_wind: bool = VTK_WIND,
    eval_episode_length: int = EVAL_EPISODE_LENGTH,
    eval_seed: int = EVAL_SEED,
    evaluation: str = EVALUATION,
    max_workers: int = MAX_WORKERS,
    resume: bool = RESUME,
    output_dir: Optional[str] = None,
    testing: bool = TESTING,
    filter_output: bool = FILTER_OUTPUT,
    filter_phrases: list[str] = FILTER_PHRASES,
    auto_shutdown: bool = AUTO_SHUTDOWN,
    shutdown_delay_minutes: int = SHUTDOWN_DELAY_MINUTES,
):
    """
    Run a comprehensive parameter sweep for WFCRL algorithms.

    Args:
        algorithms: List of algorithms to run. Default: ippo, mappo, ifac.
                   Options: ippo, mappo, ifac, ifppo, idqn, idrqn, qmix
        env_id: Environment ID for training (e.g., Dec_Ablaincourt_Floris, Dec_Turb3_Row1_Floris)
        episode_lengths: List of episode lengths to sweep over
        total_iterations: List of total iterations to sweep over
        seeds: List of random seeds to use
        wind_speed: Wind speed in m/s (default: 8)
        wind_direction: Wind direction in degrees, meteorological convention (default: 270)
        plot_power_ylim: Y-axis limits for power plots (min, max)
        plot_load_ylim: Y-axis limits for load plots (min, max)
        vtk_wind: Enable VTK wind field file generation during evaluation for ParaView visualization (default: False)
        eval_episode_length: Episode length for evaluation
        eval_seed: Random seed for evaluation
        evaluation: Evaluation mode - "floris" (FLORIS only), "fastfarm" (FAST.Farm only), 
                   or "both" (evaluate on both simulators). Default: "floris"
        max_workers: Maximum number of parallel training processes
        resume: If True, skip completed runs in progress.json and reuse existing models from 
                previous sweeps. If False, train everything from scratch (ignore progress.json 
                and don't search for existing models).
        output_dir: Path to sweep directory. If exists, will RESUME that sweep.
                   If new, creates sweep there. If None, creates new timestamped directory.
        testing: If True, prepend 'TEST_' to sweep directory name (default: False)
        filter_output: If True, filter unwanted console messages from evaluation output (default: True)
        filter_phrases: List of phrases to filter from output (default: warnings and library messages)
        auto_shutdown: Automatically shutdown the system after sweep completes (default: False)
        shutdown_delay_minutes: Minutes to wait before shutting down (default: 5, allows time to cancel)

    Examples:
        # Start a new sweep (creates parameter_sweeps/<timestamp>_parameter_sweep/)
        python parameter_sweep_v2.py --algorithms ippo mappo

        # Cross-simulator evaluation (train on FLORIS, evaluate on both FLORIS and FAST.Farm)
        python parameter_sweep_v2.py \\
            --algorithms ippo \\
            --env_id Dec_Ablaincourt_Floris \\
            --evaluation both
        
        # Train on FLORIS, evaluate only on FAST.Farm
        python parameter_sweep_v2.py \\
            --algorithms ippo \\
            --env_id Dec_Ablaincourt_Floris \\
            --evaluation fastfarm

        # Resume an interrupted sweep (continues from progress.json)
        python parameter_sweep_v2.py \\
            --output_dir parameter_sweeps/10-00-00_16-10-25_parameter_sweep

        # Long-running sweep with auto-shutdown
        python parameter_sweep_v2.py \\
            --algorithms ippo mappo idqn idrqn qmix \\
            --total_iterations 50000 100000 \\
            --seeds 1 2 3 4 5 \\
            --auto_shutdown True \\
            --shutdown_delay_minutes 10

        # Multiple algorithms with custom parameters
        python parameter_sweep_v2.py --algorithms ippo mappo ifac \\
            --env_id Dec_Turb3_Row1_Floris \\
            --episode_lengths 100 200 300 \\
            --total_iterations 10000 50000 \\
            --seeds 1 2 3 4 5

        # Disable resume to force re-training even if models exist
        python parameter_sweep_v2.py --algorithms ifppo \\
            --resume False
        
        # Testing mode (creates TEST_<timestamp>_parameter_sweep directory)
        python parameter_sweep_v2.py --algorithms ippo \\
            --testing True
    
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

    # Validate evaluation mode
    if evaluation.lower() not in ["floris", "fastfarm", "both"]:
        raise ValueError(
            f"Invalid evaluation mode: '{evaluation}'. Must be 'floris', 'fastfarm', or 'both'"
        )
    
    # Create sweep configuration
    config = SweepConfig(
        algorithms=algorithms,
        env_id=env_id,
        episode_lengths=episode_lengths,
        total_iterations=total_iterations,
        seeds=seeds,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        plot_power_ylim=plot_power_ylim,
        plot_load_ylim=plot_load_ylim,
        vtk_wind=vtk_wind,
        eval_episode_length=eval_episode_length,
        eval_seed=eval_seed,
        evaluation=evaluation,
        max_workers=max_workers,
        resume=resume,
        output_dir=Path(output_dir) if output_dir else None,
        testing=testing,
        filter_output=filter_output,
        filter_phrases=filter_phrases,
        auto_shutdown=auto_shutdown,
        shutdown_delay_minutes=shutdown_delay_minutes,
    )

    # Run the sweep
    sweep = ParameterSweep(config)
    sweep.run_sweep()


if __name__ == "__main__":
    tyro.cli(main)