# JSSP_Simulation_Environment.py
# -*- coding: utf-8 -*-
"""
JSSP Simulation Environment (OpenAI Gym / Gymnasium Compatible)
FIXED: Job constructor now correctly sets 'op_count' for Heuristic calls.

@authors: SHAIK RIFSHU, ANKITHN RAMESH BABU (Fixed by AI)
"""

import simpy
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# --- HYPERPARAMETERS & PHYSICS (TUNED FOR 1000s BASELINE) ---
NEW_JOB_ARRIVAL_RATE = 10.0  # Jobs arrive every ~25s (Slower pace for 1000s window)
MEAN_TIME_TO_FAILURE = 200.0
MEAN_TIME_TO_REPAIR = 10.0
NUM_MACHINES = 20
JOBS_PER_EPISODE = 150        # Goal: Finish 20 jobs
MAX_SIM_TIME = 10000          # Strict Baseline limit

# --- DATA STRUCTURES ---
class Job:
    def __init__(self, job_id, arg2=None, arg3=None, arg4=None):
        """
        Polymorphic __init__ to handle both RL Environment and Baseline Heuristic calls.
        
        Signatures:
        1. RL Env: Job(job_id, due_date)  (arg3 and arg4 are None)
        2. Heuristic: Job(job_id, route, processing_times, due_date) (4 arguments)
        """
        self.job_id = job_id
        self.current_op = 0
        self.completion_time = None
        self.priority = random.randint(1, 5)

        # --- HEURISTIC/SIMPY CALL (4 arguments provided) ---
        if arg4 is not None:
            # Assumes signature: Job(job_id, route, processing_times, due_date)
            self.route = arg2           # arg2 is route (list)
            self.processing_times = arg3 # arg3 is proc_times (list)
            self.due_date = arg4        # arg4 is due_date (float)
            self.op_count = len(self.route) # <--- CRUCIAL FIX: op_count is set here

        # --- RL ENV CALL (2 or 3 arguments provided) ---
        else: 
            # Assumes signature: Job(job_id, due_date) or Job(job_id, due_date, op_count)
            # Use arg2 as due_date
            self.due_date = arg2
            
            # Use arg3 as op_count, or generate random if not provided
            # Using isinstance(arg3, int) is more robust than arg3 is not None
            self.op_count = arg3 if isinstance(arg3, int) else random.randint(3, 7)
            
            # Generate route and processing times randomly based on op_count
            self.route = [random.randrange(NUM_MACHINES) for _ in range(self.op_count)]
            self.processing_times = [random.randint(5, 50) for _ in range(self.op_count)]


    def get_current_op_details(self):
        """ Returns (machine_id, processing_time) for the current operation. """
        if self.current_op >= self.op_count:
            return None, None
        return self.route[self.current_op], self.processing_times[self.current_op]

    def advance(self):
        """ Increments the operation counter. """
        self.current_op += 1
        return self.current_op >= self.op_count # Returns True if job is finished

    def __lt__(self, other):
        """ Comparison method for priority queue. Lower priority number is higher priority. """
        return self.priority < other.priority

# Helper for debugging
def print_sim_state(env, machines, machine_queues):
    # This is a placeholder/template for a more complex state observation
    # and is not strictly required for the Gym environment.
    print(f"Time: {env.now}")
    for i in range(NUM_MACHINES):
        queue_lengths = len(machine_queues[i])
        print(f"  Machine {i}: Queue Length={queue_lengths}, State={machines[i].count}")

# --- REWARD FUNCTION IMPORT ---
# Note: You need a valid reward_functions.py in your directory
from reward_functions import calculate_reward_simple, calculate_reward_dense 

# --- GYMNASIUM ENVIRONMENT ---
class JobShopEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.num_machines = NUM_MACHINES
        self.max_sim_time = MAX_SIM_TIME
        
        # --- Observation Space ---
        # Space size: NUM_MACHINES (machine status) + NUM_MACHINES * 4 (queue features)
        self.observation_space = spaces.Box(low=0, high=1000, shape=(NUM_MACHINES * 5,), dtype=np.float32)
        
        # --- Action Space ---
        self.action_space = spaces.Discrete(self.num_machines) 
        
        self.env = simpy.Environment()
        self.machines = [simpy.PreemptiveResource(self.env, capacity=1) for _ in range(self.num_machines)]
        self.machine_queues = [[] for _ in range(self.num_machines)]
        self.machine_states = np.zeros(self.num_machines, dtype=np.float32) # 0: Working, >0: Broken/Repairing
        self.pending_jobs = [] 
        self.completed_jobs = []
        self.job_counter = 0
        
        self.reward_function = calculate_reward_simple # Or calculate_reward_dense
        self.render_mode = render_mode
        self.last_action_invalid = False
        
        self.job_generator = self.dynamic_job_generator()
        self.breakdown_generators = [self.breakdown_generator(i) for i in range(self.num_machines)]

    def _get_obs(self):
        """ Generates the current state observation for the RL agent. """
        
        obs = []
        for i in range(self.num_machines):
            # 1. Machine Busy/Idle (0 or 1)
            is_busy = 1.0 if self.machines[i].count > 0 and self.machine_states[i] == 0 else 0.0 
            obs.append(is_busy) 
            
            # 2. Queue Length
            obs.append(len(self.machine_queues[i]))
            
            # 3. Next Job's Remaining Processing Time (0 if queue is empty)
            if self.machine_queues[i]:
                next_job = self.machine_queues[i][0]
                _, pt = next_job.get_current_op_details()
                obs.append(pt if pt is not None else 0.0)
            else:
                obs.append(0.0)
                
            # 4. Next Job's Due Date (relative to now)
            if self.machine_queues[i]:
                next_job = self.machine_queues[i][0]
                obs.append(max(0, next_job.due_date - self.env.now))
            else:
                obs.append(0.0)
                
            # 5. Machine Broken Status (0=OK, >0=Time remaining)
            obs.append(self.machine_states[i])
            
        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        """ Returns auxiliary information. """
        return {
            "time": self.env.now,
            "jobs_completed": len(self.completed_jobs),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # --- Reset Simulation State ---
        self.env = simpy.Environment()
        self.machines = [simpy.PreemptiveResource(self.env, capacity=1) for _ in range(self.num_machines)]
        self.machine_queues = [[] for _ in range(self.num_machines)]
        self.machine_states = np.zeros(self.num_machines, dtype=np.float32)
        self.pending_jobs = [] 
        self.completed_jobs = []
        self.job_counter = 0
        self.last_action_invalid = False

        # --- Setup Processes ---
        # The generator processes are now SimPy processes
        self.job_generator_proc = self.env.process(self.dynamic_job_generator())
        self.breakdown_procs = [self.env.process(self.breakdown_generator(i)) for i in range(self.num_machines)]

        # Run until the first decision point (when first jobs arrive)
        self.env.run(until=1) 
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        """ Executes the chosen action (scheduling decision). """
        
        machine_id = action
        self.last_action_invalid = False
        
        if self.machine_states[machine_id] > 0:
            # Cannot schedule on a broken machine
            self.last_action_invalid = True
            # Advance time slightly to check for breakdowns/arrivals/repairs
            self.env.run(self.env.now + 1)

        elif not self.machine_queues[machine_id]:
            # Cannot schedule if queue is empty
            self.last_action_invalid = True
            # Advance time slightly
            self.env.run(self.env.now + 1)
        
        else:
            # --- VALID ACTION: Schedule the best job from the chosen machine's queue (using FIFO here) ---
            
            job_to_schedule = self.machine_queues[machine_id].pop(0) # FIFO selection
            
            self.env.process(self.process_operation(job_to_schedule, machine_id))
            
            # Advance time by 1 step to allow SimPy processes to run
            self.env.run(self.env.now + 1) 
            
            
        # --- Check for Termination ---
        terminated = len(self.completed_jobs) >= JOBS_PER_EPISODE
        truncated = self.env.now >= self.max_sim_time
        done = terminated or truncated

        # --- Calculate Reward ---
        reward = self.reward_function(self, done) 
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def dynamic_job_generator(self):
        """ SimPy process to generate new jobs dynamically. """
        while True:
            # Randomly determine arrival time
            yield self.env.timeout(random.expovariate(1.0/NEW_JOB_ARRIVAL_RATE))
            
            # Generate new job
            self.job_counter += 1
            op_count = random.randint(3, 7)
            route = [random.randrange(self.num_machines) for _ in range(op_count)]
            proc_times = [random.randint(5, 50) for _ in range(op_count)]
            
            # Due date is set relative to arrival time 
            total_proc_time = sum(proc_times)
            slack = random.uniform(1.5, 3.0) 
            due_date = self.env.now + total_proc_time * slack
            
            # Uses the 4-argument Heuristic signature for consistency
            new_job = Job(self.job_counter, route, proc_times, due_date)
            
            # Add job to the queue of its first required machine
            next_machine, _ = new_job.get_current_op_details()
            if next_machine is not None:
                self.machine_queues[next_machine].append(new_job)

    def breakdown_generator(self, machine_id):
        """ SimPy process to simulate machine breakdowns. """
        while True:
            # Time until next failure
            yield self.env.timeout(random.expovariate(1.0/MEAN_TIME_TO_FAILURE))
            
            # Breakdown occurs
            # print(f"[{self.env.now:.2f}] Machine {machine_id} broke down.")
            self.machine_states[machine_id] = 1 # Mark as broken
            
            # Request the machine (preempts any ongoing job)
            with self.machines[machine_id].request(priority=0) as req: # High priority
                yield req
                
                # Repair time
                repair_time = random.expovariate(1.0/MEAN_TIME_TO_REPAIR)
                self.machine_states[machine_id] = repair_time # Mark repair time remaining
                yield self.env.timeout(repair_time)
                
                # Repair complete
                # print(f"[{self.env.now:.2f}] Machine {machine_id} repaired.")
                self.machine_states[machine_id] = 0 # Mark as repaired

    def process_operation(self, job, machine_id):
        """ SimPy process to handle a job's operation on a machine. """
        
        _, proc_time = job.get_current_op_details()
        
        # Request the machine for processing (low priority, can be preempted)
        with self.machines[machine_id].request(priority=1) as req: 
            
            start_time = self.env.now
            try:
                # Wait for the machine and yield the processing time
                yield req & self.env.timeout(proc_time)
                
                # Operation completed
                if job.advance():
                    # Job finished
                    job.completion_time = self.env.now
                    self.completed_jobs.append(job)
                else:
                    # Job moves to the next machine's queue
                    next_machine, _ = job.get_current_op_details()
                    if next_machine is not None:
                        self.machine_queues[next_machine].append(job)
                        
            except simpy.Interrupt as i:
                # Preemption occurred (e.g., by a breakdown)
                time_lost = self.env.now - start_time
                remaining_time = proc_time - time_lost
                job.processing_times[job.current_op] = remaining_time # Update remaining time
                
                # Put the job back at the front of the queue
                self.machine_queues[machine_id].insert(0, job) 
                # print(f"[{self.env.now:.2f}] Job {job.job_id} preempted on Machine {machine_id}. Remaining: {remaining_time:.2f}")

    def render(self):
        """ Prints current state for human readability. """
        if self.render_mode == "human":
            print(f"Time: {self.env.now:.2f} | Completed Jobs: {len(self.completed_jobs)}")
            for i in range(self.num_machines):
                state = "BROKEN" if self.machine_states[i] > 0 else "OK"
                queue_len = len(self.machine_queues[i])
                print(f"  M{i}: Status={state}, Queue={queue_len}")
        elif self.render_mode == "ansi":
            return f"Time: {self.env.now:.2f}, Jobs: {len(self.completed_jobs)}"
        else:
            pass # No rendering