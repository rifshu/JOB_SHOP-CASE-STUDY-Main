# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 23:32:11 2025

@author: SHAIK RIFSHU
"""

import simpy
import random
import numpy as np
from JSSP_Simulation_Environment import (
    Job, NUM_MACHINES, JOBS_PER_EPISODE, NEW_JOB_ARRIVAL_RATE,
    MEAN_TIME_TO_FAILURE, MEAN_TIME_TO_REPAIR, MAX_SIM_TIME
)

# =================================================================
# 1. HEURISTIC SIMULATION CONTEXT
# =================================================================

class HeuristicSimContext:
    """
    Context for running an episode under Heuristic control.
    Holds all shared state for SimPy processes.
    """
    def __init__(self, env):
        self.env = env
        self.machines = [simpy.PreemptiveResource(env, capacity=1) for _ in range(NUM_MACHINES)]
        self.machine_queues = [[] for _ in range(NUM_MACHINES)] # Jobs waiting for machines
        self.machine_states = np.zeros(NUM_MACHINES, dtype=np.float32) # 0=OK, >0=Repairing
        self.pending_jobs = []
        self.completed_jobs = []
        self.job_counter = 0

# =================================================================
# 2. SIMPY HELPER PROCESSES (Used by all Heuristics, ACO, and PSO)
# =================================================================

def generate_job_heuristic(ctx: HeuristicSimContext, initial_job=False):
    """ Generates a single job (used for initial setup). """
    ctx.job_counter += 1
    job_id = ctx.job_counter
    
    # Generate job details (matching JSSP_Simulation_Environment logic)
    op_count = random.randint(3, 7)
    route = [random.randrange(NUM_MACHINES) for _ in range(op_count)]
    proc_times = [random.randint(5, 50) for _ in range(op_count)]
    
    total_proc_time = sum(proc_times)
    slack = random.uniform(1.5, 3.0) 
    due_date = ctx.env.now + total_proc_time * slack
    
    # Job constructor handles the 4 arguments for heuristic mode
    new_job = Job(job_id, route, proc_times, due_date)
    
    # Add job to the queue of its first required machine
    next_machine, _ = new_job.get_current_op_details()
    if next_machine is not None:
        ctx.machine_queues[next_machine].append(new_job)

    if initial_job:
        yield ctx.env.timeout(0) 
        
def dynamic_job_generator_heuristic(ctx: HeuristicSimContext):
    """ SimPy process to generate new jobs dynamically throughout the simulation. """
    while True:
        # Randomly determine arrival time
        yield ctx.env.timeout(random.expovariate(1.0/NEW_JOB_ARRIVAL_RATE))
        ctx.env.process(generate_job_heuristic(ctx))

def breakdown_generator_heuristic(ctx: HeuristicSimContext, machine_id: int):
    """ SimPy process to simulate machine breakdowns. """
    while True:
        # Time until next failure
        yield ctx.env.timeout(random.expovariate(1.0/MEAN_TIME_TO_FAILURE))
        
        # Breakdown occurs
        ctx.machine_states[machine_id] = 1 # Mark as broken (1: in repair process)
        
        # Request the machine (preempts any ongoing job)
        with ctx.machines[machine_id].request(priority=0) as req: # High priority (0)
            yield req
            
            # Repair time
            repair_time = random.expovariate(1.0/MEAN_TIME_TO_REPAIR)
            ctx.machine_states[machine_id] = repair_time 
            yield ctx.env.timeout(repair_time)
            
            # Repair complete
            ctx.machine_states[machine_id] = 0 # Mark as repaired (0)

def process_operation_heuristic(job: Job, machine_id: int, ctx: HeuristicSimContext):
    """ SimPy process to handle a job's operation on a machine. """
    
    _, proc_time = job.get_current_op_details()
    
    # Request the machine for processing (low priority, can be preempted)
    with ctx.machines[machine_id].request(priority=1) as req: 
        
        start_time = ctx.env.now
        try:
            # Wait for the machine and yield the processing time
            yield req & ctx.env.timeout(proc_time)
            
            # Operation completed
            if job.advance():
                # Job finished
                job.completion_time = ctx.env.now
                ctx.completed_jobs.append(job)
            else:
                # Job moves to the next machine's queue
                next_machine, _ = job.get_current_op_details()
                if next_machine is not None:
                    ctx.machine_queues[next_machine].append(job)
                    
        except simpy.Interrupt as i:
            # Preemption occurred (e.g., by a breakdown)
            time_lost = ctx.env.now - start_time
            remaining_time = proc_time - time_lost
            job.processing_times[job.current_op] = remaining_time # Update remaining time
            
            # Put the job back at the front of the queue
            ctx.machine_queues[machine_id].insert(0, job) 

# =================================================================
# 3. BASELINE SCHEDULER PROCESSES
# =================================================================

def fifo_scheduler(ctx: HeuristicSimContext):
    """
    A simpy process that implements the FIFO (First In, First Out) 
    scheduling rule.
    """
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
                # FIFO: Select the job at the head of the queue (first element)
                selected_job = queue.pop(0) 
                
                # Start the processing SimPy process
                ctx.env.process(process_operation_heuristic(selected_job, machine_id, ctx))


def spt_scheduler(ctx: HeuristicSimContext):
    """
    A simpy process that implements the SPT (Shortest Processing Time) 
    scheduling rule.
    """
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
                # SPT: Find the job with the shortest remaining processing time (PT)
                
                # Create a list of (PT, job) tuples
                job_candidates = []
                for job in queue:
                    _, proc_time = job.get_current_op_details()
                    job_candidates.append((proc_time, job))
                    
                # Select the job with the minimum PT
                selected_job = min(job_candidates, key=lambda x: x[0])[1]
                
                # Remove the selected job from the queue
                queue.remove(selected_job) 
                
                # Start the processing SimPy process
                ctx.env.process(process_operation_heuristic(selected_job, machine_id, ctx))


# =================================================================
# 4. GENERALIZED EXECUTION RUNNERS
# =================================================================

def run_simulation_with_scheduler(scheduler_func, run_time, seed=None, **kwargs):
    """
    Utility to run a simulation with an external scheduler function.
    FIX: Added seed functionality for environment reproducibility.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    env = simpy.Environment()
    ctx = HeuristicSimContext(env)

    # Setup Breakdown generators
    for i in range(NUM_MACHINES):
        env.process(breakdown_generator_heuristic(ctx, i))

    # Create initial jobs
    for _ in range(JOBS_PER_EPISODE):
        env.process(generate_job_heuristic(ctx, initial_job=True))
        
    # Start dynamic arrival generator
    env.process(dynamic_job_generator_heuristic(ctx))
    
    # Start the chosen scheduler process
    env.process(scheduler_func(ctx, **kwargs))

    # Run the simulation until the specified time
    env.run(until=run_time)
    
    # Collect and return final stats
    makespan = env.now
    total_tardiness = 0
    for job in ctx.completed_jobs:
        tardiness = max(0, job.completion_time - job.due_date)
        total_tardiness += tardiness
    
    return {
        "makespan": makespan,
        "total_tardiness": total_tardiness,
        "completed_jobs": len(ctx.completed_jobs)
    }


def run_simulation_with_heuristic(scheduler_type: str, seed: int):
    """ 
    Runs the full simulation for a simple heuristic and packages results.
    FIX: Accepts and passes a seed.
    """
    
    if scheduler_type == "fifo":
        scheduler_func = fifo_scheduler
    elif scheduler_type == "spt":
        scheduler_func = spt_scheduler
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
    # Use the generalized runner, passing the seed
    results = run_simulation_with_scheduler(scheduler_func, run_time=MAX_SIM_TIME, seed=seed)
    
    # Package results for unified printing in JSSPManager
    results["scheduler_type"] = scheduler_type.upper()
    
    return results

