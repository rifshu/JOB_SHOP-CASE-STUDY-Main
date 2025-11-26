# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 01:07:53 2025

@author: SHAIK RIFSHU
"""

import numpy as np
import random
import simpy
from copy import deepcopy

# Import constants
from JSSP_Simulation_Environment import (
    Job, NUM_MACHINES, MAX_SIM_TIME, JOBS_PER_EPISODE, NEW_JOB_ARRIVAL_RATE
)

# Import necessary helper functions and context from Baseline_Heuristics
try:
    from Baseline_Heuristics import (
        HeuristicSimContext,
        process_operation_heuristic, 
        run_simulation_with_scheduler
    )
except ImportError:
    raise ImportError("Please ensure Baseline_Heuristics.py is in the same folder.")

# --- 1. DATA STRUCTURES ---

class Particle:
    """Represents a 'particle' (a set of weights) in the PSO algorithm."""
    def __init__(self, num_weights: int):
        self.position = np.random.uniform(0, 5, num_weights)
        self.velocity = np.random.uniform(-1, 1, num_weights)
        self.pbest_position = self.position.copy()
        self.pbest_score = float('inf') 
        self.pbest_full_results = None 

# --- 2. THE PSO-OPTIMIZED SCHEDULER ---

def pso_scheduler(ctx: HeuristicSimContext, weights: np.ndarray):
    """
    A SimPy process that implements a priority rule with weights determined by PSO.
    """
    W1, W2, W3 = weights
    eps = 1e-6 

    while True:
        yield ctx.env.timeout(1) 

        for machine_id in range(NUM_MACHINES):
            queue = ctx.machine_queues[machine_id]
            
            can_schedule = (
                ctx.machine_states[machine_id] == 0 and 
                ctx.machines[machine_id].count == 0 and 
                len(queue) > 0
            )

            if can_schedule:
                
                best_score = float('-inf')
                selected_job = None
                
                for job in queue:
                    _, proc_time = job.get_current_op_details()
                    
                    spt_factor = 1.0 / (proc_time + eps)
                    slack = job.due_date - ctx.env.now
                    edd_factor = 10.0 if slack <= 0 else 1.0 / (slack + eps) 
                    priority_factor = 1.0 / (job.priority + eps) 
                    
                    priority_score = (W1 * spt_factor) + (W2 * edd_factor) + (W3 * priority_factor)
                    
                    if priority_score > best_score:
                        best_score = priority_score
                        selected_job = job
                
                if selected_job:
                    queue.remove(selected_job) 
                    ctx.env.process(process_operation_heuristic(selected_job, machine_id, ctx))


# --- 3. PSO OPTIMIZATION MANAGER ---

class PSOManager:
    """Manages the Particle Swarm Optimization process."""
    def __init__(self, n_particles: int, w: float, c1: float, c2: float):
        self.n_particles = n_particles
        self.num_weights = 3 
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.swarm = [Particle(self.num_weights) for _ in range(n_particles)]
        self.gbest_position = np.random.uniform(0, 5, self.num_weights)
        self.gbest_score = float('inf')
        self.gbest_full_results = None

    def evaluate(self, weights: np.ndarray, run_time: int, seed: int = None) -> dict:
        """
        Runs the simulation with the given weights and returns the full results dictionary.
        FIX: Added seed for environment reproducibility during evaluation.
        """
        results = run_simulation_with_scheduler(
            scheduler_func=pso_scheduler, 
            run_time=run_time, 
            weights=weights,
            seed=seed # Pass the seed to the underlying simulation environment
        )
        
        return results

    def update_particle(self, particle: Particle):
        """Updates a particle's velocity and position."""
        # ... (logic remains the same) ...
        r1 = np.random.rand(self.num_weights)
        r2 = np.random.rand(self.num_weights)
        
        new_velocity = (
            self.w * particle.velocity +                                  
            self.c1 * r1 * (particle.pbest_position - particle.position) + 
            self.c2 * r2 * (self.gbest_position - particle.position)       
        )
        particle.velocity = new_velocity
        
        new_position = particle.position + particle.velocity
        particle.position = np.clip(new_position, 0, 10) 


    def run_optimization(self, iterations: int, run_time: int = MAX_SIM_TIME, verbose: bool = True, seed=None):
        """
        Main PSO optimization loop.
        FIX: Uses the provided 'seed' to control the random number generator 
        for environment setup across all evaluations in this trial.
        """
        
        # NOTE: We only seed here to ensure particle initialization is consistent if needed,
        # but the main environment seeding happens inside evaluate().
        if seed is not None:
             random.seed(seed)
             np.random.seed(seed)
        
        print(f"--- Starting PSO Optimization ({iterations} iterations) ---")
        
        for iteration in range(1, iterations + 1):
            
            for particle in self.swarm:
                # Pass the seed to the evaluation function to control the environment
                full_results = self.evaluate(particle.position, run_time, seed=seed)
                score = full_results["total_tardiness"]
                
                # ... (Update pbest/gbest logic remains the same) ...
                if score < particle.pbest_score:
                    particle.pbest_score = score
                    particle.pbest_position = particle.position.copy()
                    particle.pbest_full_results = full_results 
                
                if score < self.gbest_score:
                    self.gbest_score = score
                    self.gbest_position = particle.position.copy()
                    self.gbest_full_results = full_results 
            
            for particle in self.swarm:
                self.update_particle(particle)

            if verbose:
                print(f"[PSO] Iteration {iteration:2d} | Best Score (Tardiness): {self.gbest_score:.2f} | Weights: {self.gbest_position.round(2)}")

        print(f"--- PSO Optimization Finished ---")

        # --- Return full structured results ---
        return {
            "mode": "run-pso",
            "best_metrics": self.gbest_full_results,
            "best_weights": self.gbest_position.tolist(),
            "best_score": self.gbest_score,
            "num_machines": NUM_MACHINES,
            "arrival_rate": NEW_JOB_ARRIVAL_RATE,
            "jobs_per_episode": JOBS_PER_EPISODE,
            "iterations": iterations
        }