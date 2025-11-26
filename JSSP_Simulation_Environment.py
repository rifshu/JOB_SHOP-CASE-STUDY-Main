# -*- coding: utf-8 -*-
"""
JSSP Simulation Environment (OpenAI Gym / Gymnasium Compatible)
Updated for ACO Integration, Solvable Physics, Backward Compatibility, and Detailed Reporting.
@authors: SHAIK RIFSHU, ANKITHN RAMESH BABU
"""

import simpy
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- HYPERPARAMETERS & PHYSICS (TUNED FOR 1000s BASELINE) ---
NEW_JOB_ARRIVAL_RATE = 25.0  # Jobs arrive every ~25s (Slower pace for 1000s window)
MEAN_TIME_TO_FAILURE = 200.0
MEAN_TIME_TO_REPAIR = 10.0
NUM_MACHINES = 3
JOBS_PER_EPISODE = 20        # Goal: Finish 20 jobs
MAX_SIM_TIME = 1000          # Strict Baseline limit

# --- DATA STRUCTURES ---
class Job:
    def __init__(self, job_id, arg2=None, arg3=None, arg4=None):
        """
        Polymorphic __init__ to handle both RL Environment and Baseline Heuristic calls.
        """
        self.job_id = job_id
        self.current_op = 0
        self.completion_time = None
        self.priority = random.randint(1, 5)

        # DETECT SIGNATURE
        if arg3 is None and arg4 is None:
            # --- CASE A: Random Generation (Used by RL/Env) ---
            # arg2 is arrival_time
            self.arrival_time = arg2 if arg2 is not None else 0
            self._generate_random_ops()
        else:
            # --- CASE B: Explicit Definition (Used by Baseline_Heuristics.py) ---
            # arg2=operations, arg3=release_time, arg4=due_date
            self.operations = arg2
            self.arrival_time = arg3
            self.due_date = arg4

    # --- COMPATIBILITY LAYER (Fixes AttributeErrors) ---
    @property
    def current_operation_index(self):
        return self.current_op

    @current_operation_index.setter
    def current_operation_index(self, value):
        self.current_op = value

    @property
    def release_time(self):
        return self.arrival_time

    def advance_operation(self):
        return self.complete_operation()
    # --------------------------------------------------

    def _generate_random_ops(self):
        """Generates random operations for the job."""
        m_order = list(range(NUM_MACHINES))
        random.shuffle(m_order)
        
        self.operations = []
        for m in m_order:
            duration = random.randint(5, 15) # Operations take 5-15 seconds
            self.operations.append((m, duration))
            
        # Calculate simplified due date based on processing time
        total_proc = sum(op[1] for op in self.operations)
        self.due_date = self.arrival_time + (total_proc * 2.5)

    def get_next_operation(self):
        if self.current_op < len(self.operations):
            return self.operations[self.current_op]
        return None

    def complete_operation(self):
        self.current_op += 1
        return self.current_op >= len(self.operations)

class JobShopEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(JobShopEnv, self).__init__()
        
        self.num_machines = NUM_MACHINES
        self.action_space = spaces.Discrete(self.num_machines)
        
        # Observation: [Machine_Status (N), Queue_Lengths (N)]
        low = np.zeros(self.num_machines * 2)
        high = np.full(self.num_machines * 2, np.inf)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        self.env = None
        self.machines = []
        self.machine_queues = []
        self.machine_states = [] # 0=working, 1=broken
        self.pending_jobs = []
        self.completed_jobs = []
        self.job_counter = 0
        
        # SimPy specific
        self.waiting_process = None
        self.last_action_invalid = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.env = simpy.Environment()
        self.machines = [simpy.PreemptiveResource(self.env, capacity=1) for _ in range(self.num_machines)]
        self.machine_queues = [[] for _ in range(self.num_machines)]
        self.machine_states = np.zeros(self.num_machines, dtype=np.float32)
        
        self.pending_jobs = []
        self.completed_jobs = []
        self.job_counter = 0
        
        # Start the generator processes
        self.env.process(self.dynamic_job_generator())
        for i in range(self.num_machines):
            self.env.process(self.breakdown_generator(i))
            
        # Run slightly to populate initial state
        self.env.run(until=1)
        
        return self._get_observation(), self._get_info()

    def _get_observation(self):
        status = self.machine_states
        queue_lens = np.array([len(q) for q in self.machine_queues], dtype=np.float32)
        return np.concatenate([status, queue_lens])

    def _get_info(self):
        return {
            "makespan": self.env.now,
            "completed_jobs": len(self.completed_jobs),
            "pending_jobs": len(self.pending_jobs)
        }

    def dynamic_job_generator(self):
        # Generates up to JOBS_PER_EPISODE
        while self.job_counter < JOBS_PER_EPISODE and self.env.now < MAX_SIM_TIME:
            arrival_delay = random.expovariate(1.0 / NEW_JOB_ARRIVAL_RATE)
            yield self.env.timeout(arrival_delay)
            
            job_id = self.job_counter
            self.job_counter += 1
            # Use Signature A: (id, time)
            job = Job(job_id, self.env.now)
            
            first_op = job.get_next_operation()
            if first_op:
                m_id = first_op[0]
                self.machine_queues[m_id].append(job)
                self.pending_jobs.append(job)

    def breakdown_generator(self, machine_id):
        while True:
            yield self.env.timeout(random.expovariate(1.0 / MEAN_TIME_TO_FAILURE))
            
            # Break machine
            self.machine_states[machine_id] = 1.0
            # Preempt
            if self.machines[machine_id].count > 0:
                pass 
            
            yield self.env.timeout(random.expovariate(1.0 / MEAN_TIME_TO_REPAIR))
            self.machine_states[machine_id] = 0.0

    def step(self, action):
        self.last_action_invalid = False
        m_id = action
        reward = 0
        
        if 0 <= m_id < self.num_machines:
            if len(self.machine_queues[m_id]) > 0 and self.machine_states[m_id] == 0:
                # SPT Rule for local dispatch
                self.machine_queues[m_id].sort(key=lambda j: j.get_next_operation()[1])
                job = self.machine_queues[m_id].pop(0)
                self.env.process(self._process_job(job, m_id))
            else:
                reward -= 1 
        else:
            reward -= 5 

        try:
            self.env.run(until=self.env.now + 5)
        except simpy.core.EmptySchedule:
            pass

        all_jobs_generated = (self.job_counter >= JOBS_PER_EPISODE)
        all_jobs_finished = (len(self.completed_jobs) == self.job_counter)
        # TERMINATION: Either we finish all jobs OR we hit the hard time limit
        terminated = (all_jobs_generated and all_jobs_finished) or (self.env.now >= MAX_SIM_TIME)
        truncated = False
        reward -= (len(self.pending_jobs) * 0.1) 
        
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _process_job(self, job, machine_id):
        op = job.get_next_operation()
        duration = op[1]
        
        try:
            with self.machines[machine_id].request(priority=0) as req:
                yield req
                yield self.env.timeout(duration)
                
            is_complete = job.complete_operation()
            if is_complete:
                job.completion_time = self.env.now
                self.completed_jobs.append(job)
                if job in self.pending_jobs:
                    self.pending_jobs.remove(job)
            else:
                next_op = job.get_next_operation()
                if next_op:
                    self.machine_queues[next_op[0]].append(job)
        except simpy.Interrupt:
            self.machine_queues[machine_id].insert(0, job)

    def render(self, mode='human'):
        print(f"Time: {self.env.now:.2f} | Completed: {len(self.completed_jobs)}/{JOBS_PER_EPISODE}")


# --- ENTRY POINT ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="none", help="run-aco | run-env")
    args = parser.parse_args()

    if args.mode == "run-aco":
        # Fix: Import inside main to prevent circular dependency
        from ACO_Heuristic import ACOManager
        
        print(f"--- Starting ACO with Arrival Rate {NEW_JOB_ARRIVAL_RATE} and {JOBS_PER_EPISODE} jobs ---")
        aco = ACOManager(rho=0.1, alpha=1.0, beta=2.0) 
        
        result = aco.run_optimization(episodes=20, run_time=MAX_SIM_TIME, verbose=True)
        
        # --- DETAILED METRICS OUTPUT ---
        best_metrics = result['best']['results']
        
        print("\n" + "="*40)
        print("       ACO OPTIMIZATION RESULTS       ")
        print("="*40)
        print(f"Best Episode Found : {result['best']['ep']}")
        print(f"Best Makespan      : {result['best']['makespan']:.2f}")
        print(f"Total Tardiness    : {best_metrics['total_tardiness']:.2f}")
        print(f"Jobs Completed     : {best_metrics['completed_jobs']}/{JOBS_PER_EPISODE}")
        print(f"Machines Used      : {NUM_MACHINES}")
        print(f"Machine Pheromones : {result['pheromones']}")
        print("="*40 + "\n")

    elif args.mode == "run-env":
        print("Sanity Check: Running Random Agent")
        env = JobShopEnv()
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            action = env.action_space.sample()
            obs, reward, terminated, _, info = env.step(action)
        print(f"Random Walk Finished. Makespan: {info['makespan']:.2f}, Jobs Done: {info['completed_jobs']}")