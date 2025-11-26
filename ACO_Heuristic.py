# ACO_Heuristic.py
# -*- coding: utf-8 -*-
"""
Ant Colony Optimization scheduler for the dynamic Job Shop.
Refined to prioritize Bottlenecks (Global ACO) and Shortest Jobs (Local Rule).
Optimizes for TARDINESS when Makespan is constrained.

@author: ANKITH RAMESH BABU
"""

import simpy
import random
import numpy as np
from copy import deepcopy

# Import constants
from JSSP_Simulation_Environment import (
    Job, NUM_MACHINES, JOBS_PER_EPISODE, NEW_JOB_ARRIVAL_RATE,
    MEAN_TIME_TO_FAILURE, MEAN_TIME_TO_REPAIR, MAX_SIM_TIME
)

# Import necessary helper functions and context from Baseline_Heuristics
try:
    from Baseline_Heuristics import (
        HeuristicSimContext,
        process_operation_heuristic, 
        generate_job_heuristic, 
        dynamic_job_generator_heuristic, 
        breakdown_generator_heuristic,
        run_simulation_with_scheduler # Needed if you decide to call it directly
    )
except ImportError:
    raise ImportError("Please ensure Baseline_Heuristics.py is in the same folder.")


class ACOManager:
    """
    Manages the Ant Colony Optimization process.
    """
    def __init__(self, rho=0.1, alpha=1.0, beta=2.0, Q=1.0):
        # ... (ACO constants setup) ...
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.Q = Q
        
        self.pheromones = np.ones(NUM_MACHINES) * 0.5
        
    def _heuristic_for_machine(self, ctx: HeuristicSimContext, machine_id: int, eps: float = 1e-6):
        """Calculates the heuristic desirability (eta) for a given machine based on jobs in its queue."""
        # ... (logic remains the same) ...
        queue = ctx.machine_queues[machine_id]
        total_desirability = 0.0
        
        for job in queue:
            machine, proc_time = job.get_current_op_details()
            
            if machine is not None:
                pt_factor = 1.0 / (proc_time + eps)
                slack = job.due_date - ctx.env.now
                dd_factor = 10.0 if slack <= 0 else 1.0 / (slack + eps) 
                total_desirability += pt_factor * 0.5 + dd_factor * 0.5
                
        eta = total_desirability
        return eta if eta > 0 else eps

    def _select_machine(self, ctx: HeuristicSimContext, decision_machines: list[int], eps: float = 1e-6) -> int:
        """Ant decision: Select a machine based on Pheromone (tau) and Heuristic (eta)."""
        # ... (logic remains the same) ...
        taus = np.array([self.pheromones[i] for i in decision_machines], dtype=np.float64)
        etas = np.array([self._heuristic_for_machine(ctx, i) for i in decision_machines], dtype=np.float64)
        
        numerators = (taus**self.alpha) * (etas**self.beta)
        
        if np.sum(numerators) == 0:
            probabilities = np.ones(len(decision_machines)) / len(decision_machines)
        else:
            probabilities = numerators / np.sum(numerators)
            
        chosen_index = np.random.choice(len(decision_machines), p=probabilities)
        return decision_machines[chosen_index]

    def _evaporate_and_deposit(self, results: dict, visited_machines: dict):
        """Applies pheromone evaporation and deposition based on the episode's results."""
        # ... (logic remains the same) ...
        self.pheromones *= (1 - self.rho)
        
        tardiness = results["total_tardiness"]
        
        if tardiness < 1e-6:
            deposit_amount = 100.0 * self.Q 
        else:
            deposit_amount = self.Q / tardiness
            
        for machine_id in visited_machines.keys():
            self.pheromones[machine_id] += deposit_amount 
            
        min_pheromone = 0.01
        self.pheromones = np.maximum(self.pheromones, min_pheromone)

    def aco_scheduler(self, ctx: HeuristicSimContext, visited_dict: dict = None):
        """The main SimPy process for the ACO scheduler (the Ant)."""
        # ... (logic remains the same) ...
        while True:
            yield ctx.env.timeout(1) 

            decision_machines = []
            for machine_id in range(NUM_MACHINES):
                queue = ctx.machine_queues[machine_id]
                
                is_available = (
                    ctx.machine_states[machine_id] == 0 and 
                    ctx.machines[machine_id].count == 0 and 
                    len(queue) > 0
                )

                if is_available:
                    decision_machines.append(machine_id)
            
            if len(decision_machines) > 0:
                chosen_machine = self._select_machine(ctx, decision_machines)
                
                if visited_dict is not None:
                    visited_dict[chosen_machine] = visited_dict.get(chosen_machine, 0) + 1

                queue = ctx.machine_queues[chosen_machine]
                
                # Apply SPT: Find the job with the shortest remaining PT
                job_candidates = []
                for job in queue:
                    _, proc_time = job.get_current_op_details()
                    job_candidates.append((proc_time, job))
                    
                selected_job = min(job_candidates, key=lambda x: x[0])[1]
                
                queue.remove(selected_job) 
                ctx.env.process(process_operation_heuristic(selected_job, chosen_machine, ctx))


    def run_episode(self, run_time: int = MAX_SIM_TIME, track_visits: bool = True, random_seed=None):
        """Runs a single simulation episode (Ant traversal) using the current pheromone levels."""
        
        # We set the seed here to control environment and job randomness for this specific episode
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            
        env = simpy.Environment()
        ctx = HeuristicSimContext(env)

        visited = {} if track_visits else None

        # --- Setup Processes (Same as Baseline) ---
        for i in range(NUM_MACHINES):
            env.process(breakdown_generator_heuristic(ctx, i))
            
        for _ in range(JOBS_PER_EPISODE):
            env.process(generate_job_heuristic(ctx, initial_job=True))
        
        env.process(dynamic_job_generator_heuristic(ctx))
        
        env.process(self.aco_scheduler(ctx, visited))

        # Run simulation
        env.run(until=run_time)

        # Compute stats
        makespan = env.now
        total_tardiness = 0
        for job in ctx.completed_jobs:
            tardiness = max(0, job.completion_time - job.due_date)
            total_tardiness += tardiness
        
        return {
            "makespan": makespan,
            "total_tardiness": total_tardiness,
            "completed_jobs": len(ctx.completed_jobs),
            "visits": visited
        }

    def run_optimization(self, episodes: int, run_time: int = MAX_SIM_TIME, verbose: bool = True, seed=None):
        """
        Main optimization loop: runs multiple episodes, updates pheromones.
        FIX: Uses the provided 'seed' to control the random number generator 
        for environment setup across all episodes in this trial.
        """
        history = []
        best = None
        
        print(f"--- Starting ACO Optimization ({episodes} episodes) ---")
        
        for ep in range(1, episodes + 1):
            
            # The seed for the environment setup (Job/Breakdown generation) is determined by 
            # the 'seed' passed from main.py, ensuring that each trial runs against the 
            # same set of random events. We pass the trial seed to run_episode.
            
            results = self.run_episode(
                run_time=run_time, 
                track_visits=True, 
                random_seed=seed # Pass the trial seed to ensure environmental consistency
            )
            
            makespan = results["makespan"]
            tardiness = results["total_tardiness"]
            visits = results["visits"]

            self._evaporate_and_deposit(results, visited_machines=visits)

            history.append(tardiness)
            
            if best is None or tardiness < best["results"]["total_tardiness"]:
                best = {
                    "makespan": makespan, 
                    "results": results, 
                    "pheromones": self.pheromones.copy(), 
                    "ep": ep
                }

            if verbose:
                print(f"[ACO] Ep {ep:3d} | Tardiness: {tardiness:.2f} | Jobs: {results['completed_jobs']}")

        print(f"--- ACO Best Result: Episode {best['ep']} (Tardiness: {best['results']['total_tardiness']:.2f}) ---")

        # --- Standardized Return Structure ---
        return {
            "mode": "run-aco",
            "best_metrics": best['results'],
            "best_episode": best['ep'],
            "pheromones": best["pheromones"].tolist(),
            "num_machines": NUM_MACHINES,
            "arrival_rate": NEW_JOB_ARRIVAL_RATE,
            "jobs_per_episode": JOBS_PER_EPISODE,
            "episodes": episodes
        }