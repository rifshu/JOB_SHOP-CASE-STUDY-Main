# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 23:39:01 2025

@author: SHAIK RIFSHU
"""

import argparse
import sys
# Functions imported from sibling files
from Deep_RL_Agent import train_agent, test_agent
from Baseline_Heuristics import run_simulation_with_heuristic
class JSSPManager:
    """
    Object-Oriented structure for managing the execution modes of the 
    Job Shop Scheduling Problem (JSSP) project.
    
    This class handles the routing of different execution commands (RL training,
    testing, heuristic baselines) and manages dependency checks.
    """
    
    def __init__(self):
        """ Initializes the JSSP Manager. """
        pass

    def _handle_import_error(self, package_group: str):
        """ Helper to print installation instructions on ImportError. """
        print("-" * 60)
        print(f"Execution Failed: A required library is missing.")
        
        if package_group == 'RL':
            print("Missing Packages: 'stable_baselines3' or 'gymnasium'/'gym'")
            print("ACTION: Install with -> pip install stable-baselines3[extra] gymnasium")
        elif package_group == 'SIM':
            print("Missing Package: 'simpy'")
            print("ACTION: Install with -> pip install simpy")
            
        print("-" * 60)
        sys.exit(1)

    def execute_mode(self, mode: str):
        """
        Runs the specified project mode. This method is the central dispatcher.
        
        Args:
            mode (str): The execution command ('train-rl', 'run-fifo', etc.).
        """
        mode = mode.lower()
        print(f"\n--- JSSP Project: Executing Mode '{mode.upper()}' ---\n")

        if mode == "train-rl":
            try:
                # Calls the training function from rl_agent.py
                train_agent()
            except ImportError:
                self._handle_import_error('RL')
            
        elif mode == "test-rl":
            try:
                # Calls the testing function from rl_agent.py
                test_agent()
            except ImportError:
                self._handle_import_error('RL')
            
        elif mode == "run-fifo":
            try:
                # Calls the heuristic simulation from baseline_heuristics.py
                run_simulation_with_heuristic(scheduler_type="fifo")
            except ImportError:
                self._handle_import_error('SIM')
            
        elif mode == "run-spt":
            try:
                # Calls the heuristic simulation from baseline_heuristics.py
                run_simulation_with_heuristic(scheduler_type="spt")
            except ImportError:
                self._handle_import_error('SIM')
        
        else:
            print(f"Error: Unknown mode '{mode}'. Available modes are: train-rl, test-rl, run-fifo, run-spt.")


def main():
    """
    Standard command-line interface entry point. (For use in bash/terminal)
    """
    manager = JSSPManager()
    
    parser = argparse.ArgumentParser(description="JSSP Simulation and RL")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train-rl", "test-rl", "run-fifo", "run-spt"],
        help="The mode to run the script in."
    )
    
    args = parser.parse_args()
    manager.execute_mode(args.mode)

if __name__ == "__main__":
    # --------------------------------------------------------------------------------
    # IDE (SPYDER) COMPATIBLE EXECUTION
    # --------------------------------------------------------------------------------
    # This block allows you to run the file directly in Spyder (using F5 or Run) 
    # without needing command-line arguments.
    
    # 1. Define the mode you want to run here:
    IDE_RUN_MODE = "run-spt" # <-- CHANGE THIS TO: "train-rl", "test-rl", or "run-spt"

    # 2. Instantiate the class and execute the mode:
    JSSPManager().execute_mode(IDE_RUN_MODE)
    
    # NOTE: To use the traditional command-line interface (bash), you must comment 
    # out the three lines above and uncomment the main() call below:
    # main()
