# ACO_Sscheduler.py
@author : ANKITH RAMESH BABU

import numpy as np
import random
from JSSP_Simulation_Environment import JobShopEnv, NUM_MACHINES, JOBS_PER_EPISODE

class Ant:
    def __init__(self, job_ids):
        self.schedule = list(job_ids)
        random.shuffle(self.schedule)  # Start with random order

def evaluate_schedule(schedule, env_class):
    # Env must start clean for each new ant
    env = env_class()
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    # Each value in schedule means "try to process this job's next operation ASAP"
    # But your env chooses by machine, not job, so we need a simple mapping:
    while not terminated:
        # Find idle machine
        action = -1
        for i in range(NUM_MACHINES):
            if (env.machines[i].count == 0 and 
                len(env.machine_queues[i]) > 0 and
                env.machine_states[i] == 0):
                action = i  # Choose machine by schedule order (basic version)
                break
        if action == -1:
            action = random.randrange(NUM_MACHINES)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    total_tardiness = sum(max(0, job.completion_time - job.due_date) for job in env.completed_jobs)
    makespan = env.env.now
    return total_tardiness, makespan

def ant_colony_optimization(num_ants=10, num_iterations=10):
    print("\n--- Starting ACO Scheduler ---")
    # Assume jobs are 0 to JOBS_PER_EPISODE-1
    job_ids = list(range(JOBS_PER_EPISODE)) 
    pheromone = np.ones((JOBS_PER_EPISODE, JOBS_PER_EPISODE))  # Pheromone for transitions
    results = []
    best_schedule = None
    best_score = float('inf')
    
    for iteration in range(num_iterations):
        ant_schedules = []
        for ant_num in range(num_ants):
            schedule = [random.choice(job_ids)]  # Start with a random job
            while len(schedule) < len(job_ids):
                last_job = schedule[-1]
                candidates = [j for j in job_ids if j not in schedule]
                probs = np.array([pheromone[last_job][j] for j in candidates])
                probs = probs / probs.sum()
                next_job = np.random.choice(candidates, p=probs)
                schedule.append(next_job)
            ant_schedules.append(schedule)
        
        # Evaluate ants
        scores = []
        for sch in ant_schedules:
            tardiness, makespan = evaluate_schedule(sch, JobShopEnv)
            score = tardiness + makespan  # You can adjust weights
            scores.append(score)
            if score < best_score:
                best_score = score
                best_schedule = sch
        results.append(best_score)
        
        # Update pheromones
        for idx, sch in enumerate(ant_schedules):
            score = scores[idx]
            for i in range(len(sch) - 1):
                pheromone[sch[i]][sch[i+1]] += 1.0 / (score + 1)
        pheromone *= 0.9  # Evaporation
        
        print(f"Iteration {iteration+1}: Best Score So Far = {best_score}")

    print("\nACO Finished.")
    print("Best Schedule Found:", best_schedule)
    print("Best Combined Score (tardiness+makespan):", best_score)
    return best_schedule, results

if __name__ == "__main__":
    best_schedule, scores = ant_colony_optimization(num_ants=10, num_iterations=10)
    print("Scores per iteration:", scores)
