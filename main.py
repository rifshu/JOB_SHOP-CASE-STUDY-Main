# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 23:39:01 2025

@author: SHAIK RIFSHU
"""

import argparse
import sys
import os
import numpy as np
import random
import statistics
from typing import Callable, List, Dict, Any

# --- LAZY IMPORTS ---
from JSSP_Simulation_Environment import (
    NUM_MACHINES, JOBS_PER_EPISODE, NEW_JOB_ARRIVAL_RATE, MAX_SIM_TIME
)

# --- CONFIGURATION ---
DEFAULT_FIXED_SEED = 55 # Used as the base seed

class JSSPManager:
    """
    Object-Oriented structure for managing the execution modes of the 
    Job Shop Scheduling Problem (JSSP) project.
    """
    
    def __init__(self):
        pass

    def _handle_import_error(self, name: str):
        print("-" * 60)
        print(f"Execution Failed: Could not import '{name}'.")
        print(f"Please ensure '{name}.py' is in the same directory.")
        print("-" * 60)
        sys.exit(1)

    # =================================================================
    # STATISTICAL COMPARISON HELPER
    # =================================================================
    def _run_statistical_comparison(self, mode: str, trials: int, run_func: Callable, **kwargs) -> Dict[str, Any]:
        """Runs the given scheduler/optimization function N times."""
        
        all_tardiness = []
        all_makespan = []
        all_completed_jobs = []
        best_run_data = None # Stores the details (weights/pheromones) of the best single trial

        print(f"--- Running {mode.upper()} for {trials} trial(s) ---")
        
        for trial_num in range(1, trials + 1):
            
            # Set a unique seed for this trial (variable seeds)
            current_seed = DEFAULT_FIXED_SEED + trial_num - 1
            
            # Execute the function for this trial
            trial_result = run_func(initial_seed=current_seed, **kwargs)
            
            if mode in ["run-aco", "run-pso"]:
                metrics = trial_result["best_metrics"]
            else:
                metrics = trial_result

            tardiness = metrics['total_tardiness']
            makespan = metrics['makespan']
            completed = metrics['completed_jobs']
            
            all_tardiness.append(tardiness)
            all_makespan.append(makespan)
            all_completed_jobs.append(completed)
            
            # Track the best overall run for optimization-specific details (weights/pheromones)
            if best_run_data is None or tardiness < best_run_data.get('best_metrics', {'total_tardiness': float('inf')})['total_tardiness']:
                best_run_data = trial_result
            
            if trials > 1:
                print(f"  [Trial {trial_num:2d}, Seed {current_seed}] Tardiness: {tardiness:.2f}, Jobs: {completed}")


        # --- Calculate Final Statistics ---
        mean_tardiness = statistics.mean(all_tardiness)
        std_tardiness = statistics.stdev(all_tardiness) if trials > 1 else 0.0
        
        mean_makespan = statistics.mean(all_makespan)
        std_makespan = statistics.stdev(all_makespan) if trials > 1 else 0.0
        
        mean_completed_jobs = statistics.mean(all_completed_jobs)
        
        # Consolidate results for printing
        final_results = {
            "mode": mode,
            "trials": trials,
            "mean_tardiness": mean_tardiness,
            "std_tardiness": std_tardiness,
            "mean_makespan": mean_makespan,
            "std_makespan": std_makespan,
            "mean_completed_jobs": mean_completed_jobs,
            "best_metrics": best_run_data.get('best_metrics', metrics) 
        }
        
        # Add optimization-specific data from the single best trial
        if mode == "run-aco":
            final_results["best_episode"] = best_run_data.get("best_episode")
            final_results["pheromones"] = best_run_data.get("pheromones")
        elif mode == "run-pso":
            final_results["best_weights"] = best_run_data.get("best_weights")
            
        return final_results


    # =================================================================
    # UNIFIED RESULTS PRINTING FUNCTION
    # =================================================================
    def _print_unified_results(self, mode: str, results: Dict[str, Any]):
        """Standardized output format for all scheduling methods."""
        
        trials = results['trials']
        
        print("\n" + "="*50)
        print(f"      FINAL RESULTS: {mode.upper()} SCHEDULER      ")
        print("="*50)
        
        print(f"Trials Run         : {trials}")
        print(f"Simulation Time Run: {MAX_SIM_TIME:.2f}")
        print(f"Machine Count      : {NUM_MACHINES}")
        print(f"Job Arrival Rate   : {NEW_JOB_ARRIVAL_RATE}")
        print("-" * 50)
        
        # Core Metrics (Present in all modes)
        print(f"Avg Jobs Completed : {results['mean_completed_jobs']:.1f}/{JOBS_PER_EPISODE}")
        
        # Tardiness (Primary Optimization Goal)
        tardiness_line = f"{results['mean_tardiness']:.2f}"
        if trials > 1:
            tardiness_line += f" +/- {results['std_tardiness']:.2f} (Std Dev)"
        print(f"Avg Total Tardiness: {tardiness_line}")
        
        # Makespan
        makespan_line = f"{results['mean_makespan']:.2f}"
        if trials > 1:
            makespan_line += f" +/- {results['std_makespan']:.2f} (Std Dev)"
        print(f"Avg Final Makespan : {makespan_line}")
        
        # Optimization-Specific Metrics (Conditional Printing)
        if mode == "run-aco":
            print("-" * 50)
            print(f"Best Run Episode   : {results.get('best_episode')}")
            # Format pheromones list for cleaner print
            pheromones = results.get('pheromones', [])
            pheromones_str = "[" + ", ".join([f"{p:.2f}" for p in pheromones]) + "]"
            print(f"Best Pheromones    : {pheromones_str}")
            
        elif mode == "run-pso":
            print("-" * 50)
            weights = results.get('best_weights', [])
            weights_str = "[" + ", ".join([f"{w:.2f}" for w in weights]) + "]"
            print(f"Best Weights (W1, W2, W3): {weights_str}")
            
        print("="*50 + "\n")


    # =================================================================
    # EXECUTION DISPATCHER
    # =================================================================
    def execute_mode(self, mode: str, trials: int):
        mode = mode.lower()
        print(f"\n{'='*40}")
        print(f"   JSSP Project: Executing Mode '{mode.upper()}'")
        print(f"{'='*40}")

        # --- 1. REINFORCEMENT LEARNING ---
        if mode in ["train-rl", "test-rl"]:
            print("Note: RL modes typically use their own internal seeding/evaluation methods.")
            try:
                from Deep_RL_Agent import train_agent, test_agent
                if mode == "train-rl":
                    train_agent()
                else:
                    test_agent()
            except ImportError:
                self._handle_import_error("Deep_RL_Agent (and stable-baselines3)")

        # --- 2. BASELINE HEURISTICS (FIFO, SPT) ---
        elif mode in ["run-fifo", "run-spt"]:
            try:
                from Baseline_Heuristics import run_simulation_with_heuristic
                scheduler_type = mode.split('-')[-1]
                
                # Wrapper function for statistical running
                run_func = lambda initial_seed: run_simulation_with_heuristic(scheduler_type, seed=initial_seed)
                
                results = self._run_statistical_comparison(mode, trials, run_func)
                self._print_unified_results(mode, results)
                
            except ImportError:
                self._handle_import_error("Baseline_Heuristics (and simpy)")

        # --- 3. ACO OPTIMIZATION ---
        elif mode == "run-aco":
            try:
                from ACO_Heuristic import ACOManager
                
                # Wrapper function for statistical running
                run_func = lambda initial_seed: ACOManager(rho=0.1, alpha=1.0, beta=2.0).run_optimization(
                    episodes=20, 
                    run_time=MAX_SIM_TIME, 
                    verbose=False, 
                    seed=initial_seed
                )
                
                results = self._run_statistical_comparison(mode, trials, run_func)
                self._print_unified_results(mode, results)
                
            except ImportError:
                self._handle_import_error("ACO_Heuristic")

        # --- 4. PSO OPTIMIZATION ---
        elif mode == "run-pso":
            try:
                from PSO_Heuristic import PSOManager
                
                # Wrapper function for statistical running
                run_func = lambda initial_seed: PSOManager(n_particles=10, w=0.7, c1=1.5, c2=1.5).run_optimization(
                    iterations=15, 
                    run_time=MAX_SIM_TIME, 
                    verbose=False, 
                    seed=initial_seed
                )
                
                results = self._run_statistical_comparison(mode, trials, run_func)
                self._print_unified_results(mode, results)
                
            except ImportError:
                self._handle_import_error("PSO_Heuristic")

        else:
            print(f"Error: Unknown mode '{mode}'.")
            print("Available modes: train-rl, test-rl, run-fifo, run-spt, run-aco, run-pso")


def main():
    """
    Standard command-line interface entry point.
    To run 10 trials: python main.py --mode run-spt --trials 10
    """
    manager = JSSPManager()
    
    parser = argparse.ArgumentParser(description="JSSP Execution Manager")
    parser.add_argument(
        "--mode",
        type=str,
        required=False, 
        choices=["train-rl", "test-rl", "run-fifo", "run-spt", "run-aco", "run-pso"],
        help="The execution mode."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of stochastic trials to run for statistical comparison (N=1 uses a fixed base seed)."
    )
    args, unknown = parser.parse_known_args()
    
    # --------------------------------------------------------------------------------
    # IDE (SPYDER) COMPATIBLE EXECUTION
    # --------------------------------------------------------------------------------
    if not args.mode:
        # 1. Define the default IDE mode and trials here:
        IDE_RUN_MODE = "run-fifo" 
        IDE_RUN_TRIALS = 1 

        print("--- IDE/No-Argument Execution Detected ---")
        print(f"Executing default IDE mode: {IDE_RUN_MODE} (Trials: {IDE_RUN_TRIALS})")

        manager.execute_mode(IDE_RUN_MODE, IDE_RUN_TRIALS)
        
    else:
        # Standard command-line execution
        manager.execute_mode(args.mode, args.trials)


if __name__ == "__main__":
    main()


