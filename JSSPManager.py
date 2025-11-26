# -*- coding: utf-8 -*-
"""
JSSP Manager - Central Execution Dispatcher
Handles switching between RL training, testing, ACO, PSO, and Heuristic baselines.

@author: ANKITH RAMESH BABU
"""

import argparse
import sys
import os

# --- LAZY IMPORTS ---
# We import these inside the methods or try/except blocks to avoid 
# crashing if you are only trying to run one specific mode.

class JSSPManager:
    def __init__(self):
        pass

    def _handle_import_error(self, name):
        print("-" * 60)
        print(f"Execution Failed: Could not import '{name}'.")
        print(f"Please ensure '{name}.py' is in the same directory.")
        print("-" * 60)
        sys.exit(1)

    def execute_mode(self, mode: str):
        mode = mode.lower()
        print(f"\n{'='*40}")
        print(f"   JSSP Project: Executing Mode '{mode.upper()}'")
        print(f"{'='*40}\n")

        # --- 1. REINFORCEMENT LEARNING ---
        if mode in ["train-rl", "test-rl"]:
            try:
                from Deep_RL_Agent import train_agent, test_agent
                if mode == "train-rl":
                    train_agent()
                else:
                    test_agent()
            except ImportError:
                self._handle_import_error("Deep_RL_Agent")

        # --- 2. BASELINE HEURISTICS (FIFO / SPT) ---
        elif mode in ["run-fifo", "run-spt"]:
            try:
                from Baseline_Heuristics import run_simulation_with_heuristic
                # Remove 'run-' prefix to get 'fifo' or 'spt'
                scheduler = mode.split('-')[1] 
                run_simulation_with_heuristic(scheduler_type=scheduler)
            except ImportError:
                self._handle_import_error("Baseline_Heuristics")

        # --- 3. ANT COLONY OPTIMIZATION ---
        elif mode == "run-aco":
            try:
                from ACO_Heuristic import ACOManager
                from JSSP_Simulation_Environment import MAX_SIM_TIME, NEW_JOB_ARRIVAL_RATE, JOBS_PER_EPISODE
                
                print(f"--- Starting ACO with Arrival Rate {NEW_JOB_ARRIVAL_RATE} and {JOBS_PER_EPISODE} jobs ---")
                aco = ACOManager(rho=0.1, alpha=1.0, beta=2.0)
                result = aco.run_optimization(episodes=20, run_time=MAX_SIM_TIME, verbose=True)
                
                # Print Final Summary
                best = result['best']
                print("\n" + "="*40)
                print("       ACO OPTIMIZATION RESULTS       ")
                print("="*40)
                print(f"Best Episode       : {best['ep']}")
                print(f"Best Tardiness     : {best['results']['total_tardiness']:.2f}")
                print(f"Jobs Completed     : {best['results']['completed_jobs']}")
                print(f"Machine Pheromones : {result['pheromones']}")
                print("="*40 + "\n")
            except ImportError:
                self._handle_import_error("ACO_Heuristic")

        # --- 4. PARTICLE SWARM OPTIMIZATION ---
        elif mode == "run-pso":
            try:
                from PSO_Heuristic import PSOManager
                from JSSP_Simulation_Environment import MAX_SIM_TIME, NEW_JOB_ARRIVAL_RATE, JOBS_PER_EPISODE
                
                print(f"--- Starting PSO with Arrival Rate {NEW_JOB_ARRIVAL_RATE} and {JOBS_PER_EPISODE} jobs ---")
                pso = PSOManager(n_particles=10, w=0.7, c1=1.5, c2=1.5)
                result = pso.run_optimization(iterations=15, run_time=MAX_SIM_TIME, verbose=True)
                
                print("\n" + "="*40)
                print("       PSO OPTIMIZATION RESULTS       ")
                print("="*40)
                print(f"Best Tardiness     : {result['best_score']:.2f}")
                print(f"Best Weights       : {result['best_weights']}")
                print("="*40 + "\n")
            except ImportError:
                self._handle_import_error("PSO_Heuristic")

        else:
            print(f"Error: Unknown mode '{mode}'.")
            print("Available modes: train-rl, test-rl, run-fifo, run-spt, run-aco, run-pso")

def main():
    manager = JSSPManager()
    parser = argparse.ArgumentParser(description="JSSP Execution Manager")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train-rl", "test-rl", "run-fifo", "run-spt", "run-aco", "run-pso"],
        help="The execution mode."
    )
    args = parser.parse_args()
    manager.execute_mode(args.mode)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # IDE Fallback
        IDE_RUN_MODE = "run-aco" 
        print(f"No arguments detected. Using IDE fallback: {IDE_RUN_MODE}")
        JSSPManager().execute_mode(IDE_RUN_MODE)