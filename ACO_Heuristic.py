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

# Import the same constants and classes used by your baseline files
from JSSP_Simulation_Environment import (
    Job, NUM_MACHINES, JOBS_PER_EPISODE, NEW_JOB_ARRIVAL_RATE,
    MEAN_TIME_TO_FAILURE, MEAN_TIME_TO_REPAIR
)

# Import necessary helper functions from Baseline_Heuristics
try:
    from Baseline_Heuristics import (
        process_operation_heuristic, 
        generate_job_heuristic, 
        dynamic_job_generator_heuristic, 
        breakdown_generator_heuristic
    )
except ImportError:
    raise ImportError("Please ensure Baseline_Heuristics.py is in the same folder.")


class ACOContext:
    """
    Context for running an episode under ACO control.
    """
    def __init__(self, env, pheromones=None):
        self.env = env
        self.machines = [simpy.PreemptiveResource(env, capacity=1) for _ in range(NUM_MACHINES)]
        self.machine_queues = [[] for _ in range(NUM_MACHINES)]
        self.machine_states = np.zeros(NUM_MACHINES, dtype=np.float32)  # 0: working, 1: broken
        self.pending_jobs = []
        self.completed_jobs = []
        self.job_counter = 0
        # Pheromones are stored globally in the manager
        self.pheromones = pheromones


class ACOManager:
    """
    Holds pheromones and parameters; runs multiple episodes and updates pheromones.
    """
    def __init__(self,
                 alpha=1.0, beta=2.0, rho=0.1, Q=100.0,
                 initial_tau=1.0,
                 seed=None):
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.initial_tau = initial_tau
        # Pheromone per machine: "How good is it to focus on this machine?"
        self.pheromones = np.full(NUM_MACHINES, initial_tau, dtype=np.float64)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _heuristic_for_machine(self, ctx, machine_id, eps=1e-6):
        """
        FIXED: Heuristic information (eta) for a machine.
        Previous version: 1.0 / total_proc (Prioritized idle machines).
        New version: Returns total_proc. 
        
        Logic: We want the Ant to pay attention to 'Bottlenecks' (machines with 
        lots of work piling up) so we can clear them out.
        """
        queue = ctx.machine_queues[machine_id]
        total_proc = 0.0
        for job in queue:
            op = job.get_next_operation()
            if op:
                total_proc += op[1] # op[1] is duration
        
        # Return total load. Higher load = Higher attractiveness for the scheduler to visit it.
        return total_proc + eps

    def _select_machine(self, ctx, decision_machines):
        """
        Probabilistic machine selection using pheromones and heuristic.
        """
        taus = np.array([self.pheromones[i] for i in decision_machines], dtype=np.float64)
        etas = np.array([self._heuristic_for_machine(ctx, i) for i in decision_machines], dtype=np.float64)

        # Standard ACO probability formula
        numerators = (taus ** self.alpha) * (etas ** self.beta)
        total = numerators.sum()
        
        if total <= 0 or np.isnan(total):
            return random.choice(decision_machines)

        probs = numerators / total
        choice = np.random.choice(len(decision_machines), p=probs)
        return decision_machines[choice]

    def _evaporate_and_deposit(self, results, visited_machines=None):
        """
        Update pheromones based on TARDINESS (Quality) rather than Makespan (Time).
        """
        # 1. Evaporation
        self.pheromones *= (1.0 - self.rho)

        # 2. Extract Metrics
        total_tardiness = results["total_tardiness"]
        completed_jobs = results["completed_jobs"]

        # 3. Calculate Fitness (Higher is better)
        # We want Low Tardiness and High Throughput.
        # Formula: Give a base score, penalize tardiness. 
        # Adding 1.0 to prevent division by zero.
        if completed_jobs == 0:
            fitness = 0.001 # Penalty for doing nothing
        else:
            # Normalize tardiness per job to be fair
            avg_tardiness = total_tardiness / completed_jobs
            fitness = 1000.0 / (avg_tardiness + 1.0)

        # 4. Deposit
        deposit_amount = self.Q * fitness

        if visited_machines is None:
            self.pheromones += deposit_amount / len(self.pheromones)
        else:
            total_visits = sum(visited_machines.values()) if len(visited_machines) > 0 else 1
            for m, cnt in visited_machines.items():
                # Deposit proportional to how much this machine contributed to the success
                self.pheromones[m] += deposit_amount * (cnt / total_visits)

        # 5. Clamp limits
        min_tau = 0.01
        max_tau = 1000.0
        self.pheromones = np.clip(self.pheromones, min_tau, max_tau)

    def aco_scheduler(self, ctx: ACOContext, visited_dict=None):
        """
        FIXED Scheduler:
        1. ACO decides WHICH MACHINE to activate (prioritizing bottlenecks).
        2. Local Rule decides WHICH JOB to pick (SPT - Shortest Processing Time).
        """
        while True:
            # 1. Identify Candidate Machines (Idle, Working, Has Queue)
            decision_machines = []
            for i in range(NUM_MACHINES):
                if (ctx.machine_states[i] == 0 and
                    ctx.machines[i].count == 0 and
                    len(ctx.machine_queues[i]) > 0):
                    decision_machines.append(i)

            if len(decision_machines) > 0:
                # 2. Select Machine (The "Ant" decides where to go)
                chosen = self._select_machine(ctx, decision_machines)
                
                # Track usage for pheromone updates
                if visited_dict is not None:
                    visited_dict[chosen] = visited_dict.get(chosen, 0) + 1

                # 3. Process Job: Apply SPT Rule (Shortest Processing Time)
                # We sort the specific machine's queue so we pop the quickest job.
                # This ensures that even if we pick a bottleneck machine, we flow jobs fast.
                ctx.machine_queues[chosen].sort(key=lambda j: j.get_next_operation()[1])
                
                # Pop the best job (index 0 is now the shortest)
                job_to_process = ctx.machine_queues[chosen].pop(0)
                
                ctx.env.process(process_operation_heuristic(ctx, job_to_process, chosen))

            # Small timeout to allow simulation events (arrivals/completions) to occur
            yield ctx.env.timeout(1)

    def run_episode(self, run_time=1000, track_visits=False, random_seed=None):
        """
        Run a single simulation episode using ACO policy.
        """
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        env = simpy.Environment()
        ctx = ACOContext(env, pheromones=self.pheromones)
        visited = {} 

        # Start breakdown generators
        for i in range(NUM_MACHINES):
            env.process(breakdown_generator_heuristic(ctx, i))

        # Create initial jobs
        for _ in range(JOBS_PER_EPISODE):
            env.process(generate_job_heuristic(ctx, initial_job=True))

        # Start dynamic generator
        env.process(dynamic_job_generator_heuristic(ctx))

        # Start the ACO scheduler
        env.process(self.aco_scheduler(ctx, visited if track_visits else None))

        # Run simulation
        env.run(until=run_time)

        # Compute stats
        makespan = env.now
        total_tardiness = 0.0
        for job in ctx.completed_jobs:
            tardiness = max(0, job.completion_time - job.due_date)
            total_tardiness += tardiness

        results = {
            "makespan": makespan,
            "total_tardiness": total_tardiness,
            "completed_jobs": len(ctx.completed_jobs),
            "visits": visited
        }
        return results

    def run_optimization(self, episodes=50, run_time=1000, verbose=True, seed=None):
        best = None
        history = []
        
        print(f"--- Starting ACO Optimization ({episodes} episodes) ---")
        
        for ep in range(1, episodes + 1):
            # Run episode
            results = self.run_episode(
                run_time=run_time, 
                track_visits=True, 
                random_seed=(None if seed is None else seed + ep)
            )
            
            makespan = results["makespan"]
            tardiness = results["total_tardiness"]
            visits = results["visits"]

            # --- UPDATE CALL CHANGED HERE ---
            # Pass the full 'results' dictionary so we can see Tardiness
            self._evaporate_and_deposit(results, visited_machines=visits)

            history.append(tardiness) # Track tardiness instead of makespan
            
            # Track best based on LOWEST TARDINESS, not lowest makespan
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
        return {"best": best, "history": history, "pheromones": self.pheromones.copy()}