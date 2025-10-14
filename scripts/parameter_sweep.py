#!/usr/bin/env python3
"""
Parameter Sweep Script for WFCRL Benchmark

This script runs the ablaincourt batch script with different combinations of
episode_length and total_timesteps parameters, then organises the results
into structured directories.
"""

import os
import subprocess
import shutil
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import argparse


class ParameterSweep:
    def __init__(self, base_dir: str = "/home/reuben/code/wfcrl-benchmark"):
        self.base_dir = Path(base_dir)
        self.script_path = self.base_dir / "scripts" / "ablaincourt_batch.sh"
        self.runs_dir = self.base_dir / "runs"
        self.logs_dir = self.base_dir / "logs"
        self.most_recent_models_dir = self.base_dir / "scripts" / "most_recent_models"
        
        # Create sweep results directory with readable name
        timestamp = datetime.now().strftime("%d-%m-%y_%H-%M-%S")
        self.sweep_dir = self.base_dir / "parameter_sweeps" / f"parameter_sweep_{timestamp}"
        self.sweep_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialise run tracking log
        self.run_log_file = self.sweep_dir / "run_locations.json"
        self.run_log: Dict[str, List[str]] = {}
        
        print(f"Parameter sweep results will be saved to: {self.sweep_dir}")
    
    def log_run_locations(self, run_key: str):
        """
        Read run locations from most_recent_models directory and log them.
        Creates a snapshot of the path files to preserve them before they get overwritten.
        
        Args:
            run_key: Identifier for this set of runs
        """
        run_paths = []
        
        # Create a snapshot directory for this run's path files
        snapshot_dir = self.sweep_dir / "path_snapshots" / run_key
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # Read all algorithm path files and create snapshots
        for path_file in self.most_recent_models_dir.glob("*_path.txt"):
            try:
                with open(path_file, 'r') as f:
                    run_path = f.read().strip()
                    if run_path and Path(run_path).exists():
                        run_paths.append(run_path)
                        print(f"  Logged run: {run_path}")
                        
                        # Create snapshot of this path file
                        snapshot_file = snapshot_dir / path_file.name
                        shutil.copy2(str(path_file), str(snapshot_file))
                    elif run_path:
                        print(f"  WARNING: Path exists in {path_file.name} but directory not found: {run_path}")
            except Exception as e:
                print(f"  WARNING: Failed to read {path_file}: {e}")
        
        # Store in log
        self.run_log[run_key] = run_paths
        
        # Save to file
        with open(self.run_log_file, 'w') as f:
            json.dump(self.run_log, f, indent=2)
        
        print(f"  Snapshot saved to: {snapshot_dir}")
        print(f"  Total runs logged: {len(run_paths)}")
        
        return run_paths
    
    def run_evaluation(self, run_path: str, env_id: str, episode_length: int = 1000) -> Tuple[bool, str]:
        """
        Run evaluation for a trained model
        
        Args:
            run_path: Path to the trained model directory
            env_id: Environment ID to evaluate on
            episode_length: Episode length for evaluation (default: 1000)
            
        Returns:
            Tuple of (success, message)
        """
        # Determine algorithm from path
        algorithm = None
        for algo in ["ippo", "mappo", "idqn", "idrqn", "qmix", "ifac", "ifppo"]:
            if algo in run_path:
                algorithm = algo
                break
        
        if not algorithm:
            return False, f"Could not determine algorithm from path: {run_path}"
        
        print(f"  Running evaluation for {algorithm} on {env_id} (episode_length={episode_length})...")
        
        try:
            # Build evaluation command
            cmd = [
                "python", "algorithms/evaluate.py",
                "--algorithm", algorithm,
                "--env_id", env_id,
                "--pretrained_models", run_path,
                "--scenario", "constant",
                "--episode_length", str(episode_length)
            ]
            
            # Add hidden_layer_nn for algorithms with different architectures
            if algorithm in ["idqn", "qmix"]:
                # Single layer: (64,)
                cmd.extend(["--hidden_layer_nn", "64"])
            elif algorithm == "idrqn":
                # Two layers: (64, 64) - matches training default
                cmd.extend(["--hidden_layer_nn", "64", "64"])
            elif algorithm in ["ifac", "ifppo"]:
                # No hidden layers (Fourier features only): False
                cmd.extend(["--hidden_layer_nn", "False"])
            
            # Run evaluation script
            result = subprocess.run(
                cmd,
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                timeout=None
            )
            
            if result.returncode == 0:
                print(f"  Evaluation completed successfully for {algorithm}")
                return True, "Evaluation successful"
            else:
                error_msg = f"Evaluation failed with return code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nError: {result.stderr[:500]}"
                print(f"  {error_msg}")
                return False, error_msg
                
        except Exception as e:
            error_msg = f"Error running evaluation: {str(e)}"
            print(f"  {error_msg}")
            return False, error_msg
    
    def run_batch_script(self, episode_length: int, total_timesteps: int, 
                        additional_params: dict = None) -> Tuple[bool, str]:
        """
        Run the batch script with specified parameters
        
        Args:
            episode_length: Length of each episode in steps
            total_timesteps: Total timesteps for training
            additional_params: Additional parameters to modify in the script
            
        Returns:
            Tuple of (success, log_message)
        """
        print(f"\n{'='*60}")
        print(f"Running batch script with:")
        print(f"  Episode Length: {episode_length}")
        print(f"  Total Timesteps: {total_timesteps}")
        if additional_params:
            for key, value in additional_params.items():
                print(f"  {key}: {value}")
        print(f"{'='*60}")
        
        # Create a temporary modified script with the new parameters
        temp_script = self.base_dir / "scripts" / "temp_ablaincourt_batch.sh"
        
        try:
            # Read the original script
            with open(self.script_path, 'r') as f:
                script_content = f.read()
            
            # Modify parameters with validation
            original_episode = "episode_length=200"
            original_timesteps = "total_timesteps=10000"
            
            if original_episode not in script_content:
                raise ValueError(f"Could not find '{original_episode}' in batch script")
            if original_timesteps not in script_content:
                raise ValueError(f"Could not find '{original_timesteps}' in batch script")
            
            script_content = script_content.replace(
                original_episode, 
                f"episode_length={episode_length}"
            )
            script_content = script_content.replace(
                original_timesteps, 
                f"total_timesteps={total_timesteps}"
            )
            
            # Apply additional parameters if provided
            if additional_params:
                for param, value in additional_params.items():
                    # Find and replace parameter lines
                    lines = script_content.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip().startswith(f"{param}="):
                            lines[i] = f"{param}={value}"
                            break
                    script_content = '\n'.join(lines)
            
            # Write temporary script
            with open(temp_script, 'w') as f:
                f.write(script_content)
            
            # Make it executable
            os.chmod(temp_script, 0o755)
            
            # Run the script with real-time output
            start_time = time.time()
            print("Starting batch script execution...")
            result = subprocess.run(
                ["bash", str(temp_script)],
                cwd=self.base_dir,
                capture_output=False,  # Show output in real-time
                text=True,
                timeout=14400  # 4 hours timeout
            )
            end_time = time.time()
            duration = end_time - start_time
            
            success = result.returncode == 0
            log_message = f"Run completed in {duration:.1f}s. Return code: {result.returncode}"
            
            if not success:
                print(f"Script failed with return code: {result.returncode}")
            else:
                print(f"Script completed successfully in {duration:.1f}s")
            
            return success, log_message
            
        except subprocess.TimeoutExpired:
            return False, "Script timed out after 4 hours"
        except Exception as e:
            return False, f"Error running script: {str(e)}"
        finally:
            # Clean up temporary script
            if temp_script.exists():
                temp_script.unlink()
    
    def move_results(self, run_key: str, episode_length: int, total_timesteps: int, 
                    run_success: bool, additional_params: dict = None):
        """
        Move logged run results to organised directories
        
        Args:
            run_key: Key to look up runs in the log
            episode_length: Episode length parameter
            total_timesteps: Total timesteps parameter
            run_success: Whether the run was successful
            additional_params: Additional parameters used
        """
        # Create parameter-specific directory with readable names
        param_str = f"episode_length_{episode_length}_total_timesteps_{total_timesteps}"
        if additional_params:
            for key, value in additional_params.items():
                # Convert parameter names to readable format
                readable_key = key.replace('_', ' ').title().replace(' ', '_')
                param_str += f"_{readable_key}_{value}"
        
        param_dir = self.sweep_dir / param_str
        param_dir.mkdir(exist_ok=True)
        
        # Get runs from log
        run_paths = self.run_log.get(run_key, [])
        
        # Move runs using logged paths
        runs_moved = 0
        for run_path_str in run_paths:
            run_path = Path(run_path_str)
            if not run_path.exists():
                print(f"  WARNING: Run path no longer exists: {run_path}")
                continue
                
            try:
                dest_run_dir = param_dir / "runs" / run_path.name
                dest_run_dir.parent.mkdir(exist_ok=True)
                shutil.move(str(run_path), str(dest_run_dir))
                runs_moved += 1
                print(f"  Moved run: {run_path.name}")
            except Exception as e:
                print(f"  WARNING: Failed to move run {run_path.name}: {e}")
        
        # Move log files (still look for recent logs as they aren't tracked in most_recent_models)
        recent_time = time.time() - 1800  # 30 minutes ago
        logs_moved = 0
        for log_file in self.logs_dir.glob("batch_run_*.log"):
            if log_file.stat().st_mtime > recent_time:
                try:
                    dest_log_dir = param_dir / "logs"
                    dest_log_dir.mkdir(exist_ok=True)
                    shutil.move(str(log_file), str(dest_log_dir / log_file.name))
                    logs_moved += 1
                    print(f"  Moved log: {log_file.name}")
                except Exception as e:
                    print(f"  WARNING: Failed to move log {log_file.name}: {e}")
        
        # Create a summary file
        summary_file = param_dir / "run_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Parameter Sweep Run Summary\n")
            f.write(f"{'='*30}\n")
            f.write(f"Episode Length: {episode_length}\n")
            f.write(f"Total Timesteps: {total_timesteps}\n")
            if additional_params:
                for key, value in additional_params.items():
                    f.write(f"{key}: {value}\n")
            f.write(f"Success: {run_success}\n")
            f.write(f"Runs moved: {runs_moved}\n")
            f.write(f"Logs moved: {logs_moved}\n")
            f.write(f"Run paths logged: {len(run_paths)}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%d-%m-%y %H:%M:%S')}\n")
        
        print(f"  Results organised in: {param_dir}")
        print(f"  Summary saved to: {summary_file}")
    
    def run_sweep(self, episode_lengths: List[int], total_timesteps_list: List[int],
                  additional_params_list: List[dict] = None):
        """
        Run a parameter sweep with all combinations
        """
        if additional_params_list is None:
            additional_params_list = [{}]
        
        total_runs = len(episode_lengths) * len(total_timesteps_list) * len(additional_params_list)
        print(f"\nStarting parameter sweep with {total_runs} total runs")
        
        # Create overall summary
        overall_summary = self.sweep_dir / "sweep_summary.txt"
        
        run_count = 0
        successful_runs = 0
        
        for episode_length in episode_lengths:
            for total_timesteps in total_timesteps_list:
                for additional_params in additional_params_list:
                    run_count += 1
                    print(f"\nRun {run_count}/{total_runs}")
                    
                    # Create a unique run key for tracking
                    run_key = f"run_{run_count}_ep{episode_length}_ts{total_timesteps}"
                    
                    # Run the batch script
                    success, log_msg = self.run_batch_script(
                        episode_length, total_timesteps, additional_params
                    )
                    
                    if success:
                        successful_runs += 1
                        
                        # Log run locations after successful training
                        print("\nLogging run locations...")
                        run_paths = self.log_run_locations(run_key)
                        
                        # Run evaluation for each trained model
                        # Use longer episode length (1000) for better evaluation statistics
                        print("\nRunning evaluations on FLORIS...")
                        eval_results = []
                        for run_path in run_paths:
                            # Use FLORIS for fast evaluation (same as training environment)
                            eval_env = "Dec_Ablaincourt_Floris"  # Use FLORIS for speed
                            
                            eval_success, eval_msg = self.run_evaluation(run_path, eval_env, episode_length=1000)
                            eval_results.append({
                                "path": run_path,
                                "success": eval_success,
                                "message": eval_msg
                            })
                    
                    # Move results using logged paths
                    self.move_results(run_key, episode_length, total_timesteps, success, additional_params)
                    
                    # Update overall summary
                    with open(overall_summary, 'a') as f:
                        f.write(f"Run {run_count}: Episode_Length_{episode_length}_Total_Timesteps_{total_timesteps}")
                        if additional_params:
                            for key, value in additional_params.items():
                                readable_key = key.replace('_', ' ').title().replace(' ', '_')
                                f.write(f"_{readable_key}_{value}")
                        f.write(f" - {'SUCCESS' if success else 'FAILED'}\n")
                        if not success:
                            f.write(f"  Error: {log_msg}\n")
                        elif success and 'eval_results' in locals():
                            f.write(f"  Evaluations completed: {sum(1 for r in eval_results if r['success'])}/{len(eval_results)}\n")
        
        # Final summary
        print(f"\nParameter sweep completed!")
        print(f"Successful runs: {successful_runs}/{total_runs}")
        print(f"Results saved to: {self.sweep_dir}")
        
        with open(overall_summary, 'a') as f:
            f.write(f"\nSweep Summary:\n")
            f.write(f"Total runs: {total_runs}\n")
            f.write(f"Successful: {successful_runs}\n")
            f.write(f"Failed: {total_runs - successful_runs}\n")
            f.write(f"Completed: {datetime.now().strftime('%d-%m-%y %H:%M:%S')}\n")
    
    def shutdown_system(self, delay_minutes: int = 2):
        print(f"\nSystem will shutdown in {delay_minutes} minutes...")
        try:
            subprocess.run(["sudo", "shutdown", "-h", f"+{delay_minutes}"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to shutdown: {e}")
            print("You may need to run: sudo shutdown -h now")


def main():
    parser = argparse.ArgumentParser(description="Run parameter sweep for WFCRL benchmark")
    parser.add_argument("--episode_lengths", nargs="+", type=int, default=[600],
                       help="List of episode lengths to test")
    parser.add_argument("--total_timesteps", nargs="+", type=int, default=[50000],
                       help="List of total timesteps to test") 
    parser.add_argument("--base_dir", type=str, default="/home/reuben/code/wfcrl-benchmark",
                       help="Base directory of the WFCRL benchmark")
    parser.add_argument("--shutdown", action="store_true",
                       help="Shutdown the computer after completing all runs")
    parser.add_argument("--shutdown_delay", type=int, default=2,
                       help="Minutes to wait before shutdown (default: 2)")
    
    args = parser.parse_args()
    
    # Create parameter sweep instance
    sweep = ParameterSweep(args.base_dir)
    
    # Example additional parameters (uncomment and modify as needed)
    # additional_params_list = [
    #     {},  # Default parameters
    #     {"seed": 42},  # Different seed
    #     {"learning_rate": "3e-4"},  # Different learning rate
    # ]
    
    # Run the sweep
    sweep.run_sweep(
        episode_lengths=args.episode_lengths,
        total_timesteps_list=args.total_timesteps,
        # additional_params_list=additional_params_list  # Uncomment to use
    )
    
    # Shutdown if requested
    if args.shutdown:
        sweep.shutdown_system(args.shutdown_delay)


if __name__ == "__main__":
    main()