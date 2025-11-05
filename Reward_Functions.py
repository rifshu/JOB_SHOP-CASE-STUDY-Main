# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 23:30:41 2025

@author: SHAIK RIFSHU
"""

import numpy as np

def calculate_reward_simple(env_state, done):
    """
    A simple reward function, updated for the dynamic environment.
    - Penalizes machine idle time (when they could be working).
    - Gives a large final reward based on makespan and tardiness.
    
    Args:
        env_state (JobShopEnv): The environment instance.
        done (bool): Whether the episode has finished.
    """
    
    # 1. Penalize machine idle time (encourages utilization)
    idle_machines = 0
    for i in range(env_state.num_machines):
        # An idle machine is only "bad" if it's NOT broken and has a queue
        if (env_state.machines[i].count == 0 and 
            len(env_state.machine_queues[i]) > 0 and
            env_state.machine_states[i] == 0):
            idle_machines += 1
            
    # Reward is negative, so this is a penalty
    reward = -idle_machines 
    
    if done:
        # 2. Final reward based on makespan (total time)
        # We want to minimize makespan, so we use a large negative reward.
        makespan = env_state.env.now
        reward -= makespan * 10 # Heavily penalize long makespans
        
        # 3. Penalize job tardiness
        total_tardiness = 0
        for job in env_state.completed_jobs:
            # We now have due dates!
            tardiness = max(0, job.completion_time - job.due_date)
            total_tardiness += tardiness
            
        reward -= total_tardiness * 5 # Penalize tardiness
        
    return reward

def calculate_reward_dense(env, prev_obs, action, obs, done):
    """
    A denser reward signal, calculated at each step.
    Reward = - (sum of all job queue times since last step)
    
    We want to minimize the time jobs spend waiting.
    """
    
    # Calculate total time jobs have spent in queues
    current_queue_time = 0
    for queue in env.machine_queues:
        for job in queue:
            # Add processing time of this op (as a proxy for waiting)
            op = job.get_next_operation()
            if op:
                current_queue_time += op[1]

    # A simple dense reward: negative of total time in queues
    # The agent will learn to take actions that reduce this value.
    reward = -current_queue_time
    
    if done:
        # Add a bonus/penalty for the final makespan
        makespan = env.env.now
        reward -= makespan * 2 # Final penalty for overall time
        
    return reward

def calculate_reward_dynamic(env, done):
    """
    A more advanced reward function that uses dynamic job properties.
    This is a "dense" reward given at each step.
    
    - Penalizes waiting time, weighted by job priority.
    - Penalizes jobs that are approaching their due date (urgency).
    - Penalizes invalid actions.
    - Gives a large final penalty based on makespan and tardiness.
    
    Args:
        env (JobShopEnv): The environment instance.
        done (bool): Whether the episode has finished.
    """
    if done:
        # On the final step, calculate makespan and tardiness
        makespan = env.env.now
        total_tardiness = 0
        for job in env.completed_jobs:
            tardiness = max(0, job.completion_time - job.due_date)
            total_tardiness += tardiness
            
        # Final reward is heavily based on minimizing tardiness and makespan
        # The values 5 and 10 are hyperparameters you can tune.
        return - (makespan * 5) - (total_tardiness * 10)

    # --- Dense, per-step reward ---
    reward = 0
    
    # 1. Penalty for invalid action (from env)
    if env.last_action_invalid:
        reward -= 100.0

    # 2. Penalty for machine idle time (if it could be working)
    idle_penalty = 0
    for i in range(env.num_machines):
        if (env.machines[i].count == 0 and 
            len(env.machine_queues[i]) > 0 and
            env.machine_states[i] == 0):
            idle_penalty += 1.0
    reward -= idle_penalty
    
    # 3. Penalty for weighted queue time (Flowtime)
    # This encourages clearing high-priority jobs first.
    weighted_queue_time = 0
    urgency_penalty = 0
    for queue in env.machine_queues:
        for job in queue:
            # Higher priority (lower number) = bigger penalty for waiting
            # (1 / 1) > (1 / 5)
            weighted_queue_time += (1 / job.priority) * 1.0 # 1.0 is a proxy for time_step
            
            # Urgency penalty: how close is the job to its due date?
            time_to_due = job.due_date - env.env.now
            if time_to_due < 0:
                urgency_penalty += 10.0 # Job is already late!
            elif time_to_due < (job.due_date - job.arrival_time) * 0.25:
                # If less than 25% of its slack time is left
                urgency_penalty += 2.0 
                
    reward -= weighted_queue_time
    reward -= urgency_penalty

    # Return the cumulative reward for this step
    return reward

