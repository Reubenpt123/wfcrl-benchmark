#!/usr/bin/env python3
"""
Parameter Sweep Script for WFCRL Benchmark

This script runs the ablaincourt batch script with different combinations of
episode_length and total_timesteps parameters, then organizes the results
into structured directories.
"""

import os
import subprocess
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import argparse


class ParameterSweep:
    def __init__(self, base_dir: str = "/home/reuben/code/wfcrl-benchmark"):
        self.base_dir = Path(base_dir)
        self.script_path = self.base_dir / "scripts" / "ablaincourt_batch.sh"
        self.runs_dir = self.base_dir / "runs"
        self.logs_dir = self.base_dir / "logs"
        
        # Create sweep results directory with readable name
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.sweep_dir = self.base_dir / "parameter_sweeps" / f"parameter_sweep_{timestamp}"
        self.sweep_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Parameter sweep results will be saved to: {self.sweep_dir}")
    
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
            
            # Modify parameters
            script_content = script_content.replace(
                f"episode_length=200", 
                f"episode_length={episode_length}"
            )
            script_content = script_content.replace(
                f"total_timesteps=10000", 
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
                timeout=7200  # 2 hour timeout
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
            return False, "Script timed out after 2 hours"
        except Exception as e:
            return False, f"Error running script: {str(e)}"
        finally:
            # Clean up temporary script
            if temp_script.exists():
                temp_script.unlink()
    
    def move_results(self, episode_length: int, total_timesteps: int, 
                    run_success: bool, additional_params: dict = None):
        """
        Move the most recent run results to organized directories
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
        
        # Get the most recent runs (created in the last few minutes)  
        recent_time = time.time() - 300  # 5 minutes ago
        recent_runs = []
        
        for run_dir in self.runs_dir.iterdir():
            if run_dir.is_dir() and run_dir.stat().st_mtime > recent_time:
                recent_runs.append(run_dir)
        
        # Get the most recent log files
        recent_logs = []
        for log_file in self.logs_dir.glob("batch_run_*.log"):
            if log_file.stat().st_mtime > recent_time:
                recent_logs.append(log_file)
        
        # Move runs
        runs_moved = 0
        for run_dir in recent_runs:
            try:
                dest_run_dir = param_dir / "runs" / run_dir.name
                dest_run_dir.parent.mkdir(exist_ok=True)
                shutil.move(str(run_dir), str(dest_run_dir))
                runs_moved += 1
                print(f"  Moved run: {run_dir.name}")
            except Exception as e:
                print(f"  WARNING: Failed to move run {run_dir.name}: {e}")
        
        # Move logs
        logs_moved = 0
        for log_file in recent_logs:
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
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"  Results organized in: {param_dir}")
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
                    
                    # Run the batch script
                    success, log_msg = self.run_batch_script(
                        episode_length, total_timesteps, additional_params
                    )
                    
                    if success:
                        successful_runs += 1
                    
                    # Move results
                    self.move_results(episode_length, total_timesteps, success, additional_params)
                    
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
        
        # Final summary
        print(f"\nParameter sweep completed!")
        print(f"Successful runs: {successful_runs}/{total_runs}")
        print(f"Results saved to: {self.sweep_dir}")
        
        with open(overall_summary, 'a') as f:
            f.write(f"\nSweep Summary:\n")
            f.write(f"Total runs: {total_runs}\n")
            f.write(f"Successful: {successful_runs}\n")
            f.write(f"Failed: {total_runs - successful_runs}\n")
            f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def shutdown_system(self, delay_minutes: int = 2):
        """
        Shutdown the system after a delay
        
        Args:
            delay_minutes: Minutes to wait before shutdown
        """
        print(f"\nSystem will shutdown in {delay_minutes} minutes...")
        print(f"WARNING: Save any unsaved work now!")
        
        # Give user time to cancel if needed
        for i in range(delay_minutes * 60, 0, -10):
            minutes = i // 60
            seconds = i % 60
            print(f"Shutdown in {minutes:02d}:{seconds:02d}... (Ctrl+C to cancel)")
            time.sleep(10)
        
        print("Shutting down system now...")
        try:
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Failed to shutdown: {e}")
            print("You may need to run: sudo shutdown -h now")


def main():
    parser = argparse.ArgumentParser(description="Run parameter sweep for WFCRL benchmark")
    parser.add_argument("--episode_lengths", nargs="+", type=int, default=[50, 100, 200, 400],
                       help="List of episode lengths to test")
    parser.add_argument("--total_timesteps", nargs="+", type=int, default=[10000, 20000, 50000, 100000],
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