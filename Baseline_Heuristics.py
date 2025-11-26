# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 23:32:11 2025

@author: SHAIK RIFSHU
"""

import simpy
import random
from JSSP_Simulation_Environment import (
    Job, JobShopEnv, NUM_MACHINES, JOBS_PER_EPISODE, NEW_JOB_ARRIVAL_RATE,
    MEAN_TIME_TO_FAILURE, MEAN_TIME_TO_REPAIR
)
import numpy as np

"""
This file contains the logic for the baseline schedulers (FIFO, SPT).
It has been updated to run in the dynamic environment with breakdowns
to ensure a fair comparison with the RL agent.
"""

# We need to create a global context for the heuristic simulation
# to share state (like machine_states) between processes.
class HeuristicSimContext:
    def __init__(self, env):
        self.env = env
        self.machines = [simpy.PreemptiveResource(env, capacity=1) for _ in range(NUM_MACHINES)]
        self.machine_queues = [[] for _ in range(NUM_MACHINES)]
        self.machine_states = np.zeros(NUM_MACHINES, dtype=np.float32)
        self.pending_jobs = []
        self.completed_jobs = []
        self.job_counter = 0

def fifo_scheduler(ctx: HeuristicSimContext):
    """
    A simpy process that implements the FIFO scheduling rule in the dynamic env.
    """
    while True:
        for i in range(NUM_MACHINES):
            # Check if machine is WORKING, IDLE, and has a QUEUE
            if (ctx.machine_states[i] == 0 and
                ctx.machines[i].count == 0 and
                len(ctx.machine_queues[i]) > 0):
                
                # Get the first job in the queue (FIFO)
                job_to_process = ctx.machine_queues[i].pop(0)
                ctx.env.process(process_operation_heuristic(ctx, job_to_process, i))

        # Wait for the next event (e.g., job arrival, completion, repair)
        # This is a simple polling mechanism. A better way would use events.
        yield ctx.env.timeout(1)


def spt_scheduler(ctx: HeuristicSimContext):
    """
    A simpy process that implements the Shortest Processing Time (SPT) rule
    in the dynamic env.
    """
    while True:
        for i in range(NUM_MACHINES):
            # Check if machine is WORKING, IDLE, and has a QUEUE
            if (ctx.machine_states[i] == 0 and
                ctx.machines[i].count == 0 and
                len(ctx.machine_queues[i]) > 0):
                
                # Find the job with the shortest processing time
                queue = ctx.machine_queues[i]
                best_job = None
                spt = float('inf')
                best_job_index = -1

                for idx, job in enumerate(queue):
                    op = job.get_next_operation()
                    if op and op[1] < spt:
                        spt = op[1]
                        best_job = job
                        best_job_index = idx
                
                if best_job:
                    job_to_process = ctx.machine_queues[i].pop(best_job_index)
                    ctx.env.process(process_operation_heuristic(ctx, job_to_process, i))

        yield ctx.env.timeout(1)

def breakdown_generator_heuristic(ctx: HeuristicSimContext, machine_id):
    """ Simulates breakdowns for a machine in the heuristic context. """
    while True:
        yield ctx.env.timeout(random.expovariate(1.0 / MEAN_TIME_TO_FAILURE))
        
        print(f"Time {ctx.env.now:.2f}: [Heuristic] !!! Machine {machine_id} BROKE DOWN !!!")
        ctx.machine_states[machine_id] = 1.0
        
        with ctx.machines[machine_id].request(priority=-1) as req:
            yield req
            yield ctx.env.timeout(random.expovariate(1.0 / MEAN_TIME_TO_REPAIR))
        
        print(f"Time {ctx.env.now:.2f}: [Heuristic] +++ Machine {machine_id} REPAIRED +++")
        ctx.machine_states[machine_id] = 0.0

def process_operation_heuristic(ctx: HeuristicSimContext, job, machine_id):
    """
    A standalone simpy process for processing an operation,
    updated to handle preemptions from breakdowns.
    """
    op = job.get_next_operation()
    if not op or op[0] != machine_id:
        return

    proc_time = op[1]
    print(f"Time {ctx.env.now:.2f}: [Heuristic] Machine {machine_id} START Job {job.job_id} (Op {job.current_op}). Duration {proc_time}")
    
    try:
        with ctx.machines[machine_id].request(priority=0) as req:
            yield req
            yield ctx.env.timeout(proc_time)
        
        # --- Operation successful ---
        print(f"Time {ctx.env.now:.2f}: [Heuristic] Machine {machine_id} END Job {job.job_id} (Op {job.current_op}).")
        job_finished = job.complete_operation()
        
        if job_finished:
            job.completion_time = ctx.env.now
            ctx.pending_jobs.remove(job)
            ctx.completed_jobs.append(job)
            print(f"Time {ctx.env.now:.2f}: [Heuristic] Job {job.job_id} COMPLETED.")
        else:
            # Place job in queue for its next operation
            next_op = job.get_next_operation()
            if next_op:
                next_machine_id = next_op[0]
                ctx.machine_queues[next_machine_id].append(job)
                print(f"Time {ctx.env.now:.2f}: [Heuristic] Job {job.job_id} queued for Machine {next_machine_id}.")
                
    except simpy.Interrupt:
        # --- Operation INTERRUPTED ---
        print(f"Time {ctx.env.now:.2f}: [Heuristic] !!! INTERRUPT !!! Job {job.job_id} on Machine {machine_id} stopped.")
        # Re-queue the job at the front
        ctx.machine_queues[machine_id].insert(0, job)

def generate_job_heuristic(ctx: HeuristicSimContext, arrival_delay=0, initial_job=False):
    """ Process for generating a new job in the heuristic context. """
    if not initial_job:
        yield ctx.env.timeout(arrival_delay)
        
    job_id = ctx.job_counter
    ctx.job_counter += 1
    job = Job(job_id, ctx.env.now)
    ctx.pending_jobs.append(job)
    
    print(f"Time {ctx.env.now:.2f}: [Heuristic] Job {job.job_id} arrived (Due {job.due_date:.2f}).")
    
    # Place job in its first queue
    op = job.get_next_operation()
    if op:
        ctx.machine_queues[op[0]].append(job)

def dynamic_job_generator_heuristic(ctx: HeuristicSimContext):
    """ Continuously generates jobs in the heuristic context. """
    while True:
        arrival_delay = random.expovariate(1.0 / NEW_JOB_ARRIVAL_RATE)
        yield ctx.env.process(generate_job_heuristic(ctx, arrival_delay))

def run_simulation_with_heuristic(scheduler_type="fifo"):
    """
    Runs a full simulation using a specified heuristic in the DYNAMIC environment.
    """
    env = simpy.Environment()
    ctx = HeuristicSimContext(env)
    
    # Start breakdown generators
    for i in range(NUM_MACHINES):
        env.process(breakdown_generator_heuristic(ctx, i))

    # Create initial jobs
    for _ in range(JOBS_PER_EPISODE):
        env.process(generate_job_heuristic(ctx, initial_job=True))
        
    # Start dynamic generator
    env.process(dynamic_job_generator_heuristic(ctx))
    
    # Start the chosen scheduler process
    if scheduler_type == "fifo":
        env.process(fifo_scheduler(ctx))
    elif scheduler_type == "spt":
        env.process(spt_scheduler(ctx))
    else:
        raise ValueError("Unknown scheduler type")

    print(f"--- Running simulation with {scheduler_type.upper()} scheduler ---")
    run_time = 1000 # Run for a fixed time to compare
    env.run(until=run_time)
    
    print(f"--- Simulation with {scheduler_type.upper()} finished at time {env.now} ---")
    
    
    print("\n--- RUN CONFIGURATION AND RESULTS ---")
    print(f"Number of Machines Used: {NUM_MACHINES}")
    print(f"Number of Jobs Completed: {len(ctx.completed_jobs)}")
    print("-----------------------------------\n")
    # Collect and print final stats
    makespan = env.now
    total_tardiness = 0
    for job in ctx.completed_jobs:
        tardiness = max(0, job.completion_time - job.due_date)
        total_tardiness += tardiness
        
    print(f"Total Jobs Completed: {len(ctx.completed_jobs)}")
    print(f"Makespan (Sim Time): {makespan:.2f}")
    print(f"Total Tardiness: {total_tardiness:.2f}")
    
    
    