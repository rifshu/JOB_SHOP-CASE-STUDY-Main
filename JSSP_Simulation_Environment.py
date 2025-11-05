# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 23:29:49 2025

@author: SHAIK RIFSHU
"""

import simpy
import numpy as np
import random
try:
    # Use gymnasium if available (newer standard)
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    # Fallback to gym
    import gym
    from gym import spaces

# --- CONFIGURATION ---
NUM_MACHINES = 3
JOBS_PER_EPISODE = 10 # Initial jobs to start with
MAX_OPERATIONS_PER_JOB = 3
MAX_PROC_TIME = 15 # Max processing time for a single operation
NEW_JOB_ARRIVAL_RATE = 5 # Average time (in sim units) for a new job to arrive
RANDOM_SEED = 42

# --- DYNAMIC EVENT CONFIGURATION ---
# Average time until a machine breaks
MEAN_TIME_TO_FAILURE = 100 
# Average time it takes to repair a machine
MEAN_TIME_TO_REPAIR = 10 
# Factor for calculating job due dates (total_proc_time * factor)
DUE_DATE_SLACK_FACTOR = 2.0 

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class Job:
    """ A job consists of a sequence of operations, priority, and due date. """
    def __init__(self, job_id, arrival_time):
        self.job_id = job_id
        self.arrival_time = arrival_time
        self.operations = [] # List of (machine_id, proc_time) tuples
        self.current_op = 0
        self.completion_time = -1
        self.priority = random.randint(1, 5) # Add priority (1=high, 5=low)
        
        # Generate random operations
        num_ops = random.randint(1, MAX_OPERATIONS_PER_JOB)
        machines = random.sample(range(NUM_MACHINES), num_ops)
        total_proc_time = 0
        for i in range(num_ops):
            proc_time = random.randint(1, MAX_PROC_TIME)
            self.operations.append((machines[i], proc_time))
            total_proc_time += proc_time
            
        # Add a due date
        self.due_date = arrival_time + total_proc_time * DUE_DATE_SLACK_FACTOR

    def get_next_operation(self):
        """ Returns the next operation (machine_id, proc_time) if any. """
        if self.current_op < len(self.operations):
            return self.operations[self.current_op]
        return None

    def complete_operation(self):
        """ Mark the current operation as complete. """
        self.current_op += 1
        if self.current_op == len(self.operations):
            return True # Job is finished
        return False # More operations remaining

class JobShopEnv(gym.Env):
    """
    A dynamic Job Shop Simulation Environment with machine breakdowns.
    """
    def __init__(self):
        super(JobShopEnv, self).__init__()
        
        self.env = simpy.Environment()
        # Use PreemptiveResource to allow breakdowns to interrupt jobs
        self.machines = [simpy.PreemptiveResource(self.env, capacity=1) for _ in range(NUM_MACHINES)]
        # Track machine state: 0 = working, 1 = broken
        self.machine_states = np.zeros(NUM_MACHINES, dtype=np.float32)
        
        # --- OBSERVATION SPACE ---
        # We need to tell the agent the state of each machine.
        # Obs: [q_time_m0, state_m0, q_time_m1, state_m1, ...]
        # state is 0.0 (working) or 1.0 (broken)
        obs_shape = NUM_MACHINES * 2
        low_obs = np.zeros(obs_shape, dtype=np.float32)
        high_obs = np.full(obs_shape, np.inf, dtype=np.float32)
        for i in range(NUM_MACHINES):
            high_obs[i*2 + 1] = 1.0 # Machine state is either 0 or 1
            
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, dtype=np.float32
        )
        
        # --- ACTION SPACE ---
        # Action: "Choose machine `i` to process its next job"
        self.action_space = spaces.Discrete(NUM_MACHINES)

        # We need these attributes for the reward function
        self.num_machines = NUM_MACHINES 
        self.last_action_invalid = False
        
        self.reset()

    def _get_observation(self):
        """
        Get the current state of the environment.
        State = [queue_time_m0, state_m0, queue_time_m1, state_m1, ...]
        """
        obs = np.zeros(NUM_MACHINES * 2, dtype=np.float32)
        for i in range(NUM_MACHINES):
            # Get total processing time in the queue
            queue = self.machine_queues[i]
            total_proc_time = 0
            for job in queue:
                op = job.get_next_operation()
                if op:
                    total_proc_time += op[1] # op[1] is proc_time
            
            obs[i*2] = total_proc_time
            obs[i*2 + 1] = self.machine_states[i] # Add machine state
        return obs

    def reset(self, seed=None, options=None):
        """ Reset the environment to an initial state. """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.env = simpy.Environment()
        self.machines = [simpy.PreemptiveResource(self.env, capacity=1) for _ in range(NUM_MACHINES)]
        self.machine_states = np.zeros(NUM_MACHINES, dtype=np.float32)
        
        # Logical queues for jobs waiting for each machine
        self.machine_queues = [[] for _ in range(NUM_MACHINES)]
        self.pending_jobs = [] # All jobs in the system
        self.completed_jobs = []
        self.job_counter = 0
        self.last_event_time = 0
        self.last_action_invalid = False # Track invalid actions for reward
        
        # Create initial set of jobs
        for _ in range(JOBS_PER_EPISODE):
            self.env.process(self.generate_job(initial_job=True))
            
        # Start the dynamic job generator
        self.env.process(self.dynamic_job_generator())
        
        # Start breakdown generators for each machine
        for i in range(NUM_MACHINES):
            self.env.process(self.breakdown_generator(i))
            
        # Run simulation until the first decision point
        self.env.run(until=0) 
        
        self.decision_point_process = self.env.process(self.run_to_next_decision())
        self.waiting_process = None
        
        # Run until the first event blocks
        self.env.run()

        return self._get_observation(), self._get_info()

    def generate_job(self, arrival_delay=0, initial_job=False):
        """ Process for generating a new job. """
        if not initial_job:
            yield self.env.timeout(arrival_delay)
            
        job_id = self.job_counter
        self.job_counter += 1
        job = Job(job_id, self.env.now)
        self.pending_jobs.append(job)
        
        print(f"Time {self.env.now:.2f}: Job {job.job_id} arrived (Priority {job.priority}, Due {job.due_date:.2f}).")
        self.place_job_in_queue(job)

    def dynamic_job_generator(self):
        """ Continuously generates jobs at dynamic intervals. """
        while True:
            arrival_delay = random.expovariate(1.0 / NEW_JOB_ARRIVAL_RATE)
            yield self.env.process(self.generate_job(arrival_delay))

    def place_job_in_queue(self, job):
        """ Places a job in the appropriate machine's queue. """
        op = job.get_next_operation()
        if op:
            machine_id = op[0]
            self.machine_queues[machine_id].append(job)
            print(f"Time {self.env.now:.2f}: Job {job.job_id} queued for Machine {machine_id}.")
            
            # If the machine is idle AND working, it's a decision point
            if self.machines[machine_id].count == 0 and self.machine_states[machine_id] == 0:
                if self.decision_point_process and self.decision_point_process.is_alive:
                    self.decision_point_process.interrupt()

    def breakdown_generator(self, machine_id):
        """ A simpy process that simulates machine breakdowns and repairs. """
        while True:
            # Wait for Time-To-Failure (TTF)
            yield self.env.timeout(random.expovariate(1.0 / MEAN_TIME_TO_FAILURE))
            
            # --- Machine BROKEN ---
            print(f"Time {self.env.now:.2f}: !!! Machine {machine_id} BROKE DOWN !!!")
            self.machine_states[machine_id] = 1.0 # Mark as broken
            
            # Request the machine with high priority (-1) to interrupt any running job
            # This is a 'seize' operation for repair.
            with self.machines[machine_id].request(priority=-1) as req:
                yield req
                
                # Machine is now seized (and any running job is interrupted)
                # Wait for Time-To-Repair (TTR)
                yield self.env.timeout(random.expovariate(1.0 / MEAN_TIME_TO_REPAIR))
            
            # --- Machine REPAIRED ---
            print(f"Time {self.env.now:.2f}: +++ Machine {machine_id} REPAIRED +++")
            self.machine_states[machine_id] = 0.0 # Mark as working
            
            # Trigger a new decision, as this machine might be available now
            if self.decision_point_process and self.decision_point_process.is_alive:
                self.decision_point_process.interrupt()


    def run_to_next_decision(self):
        """
        A simpy process that runs the simulation until an action is needed.
        Action needed = machine is idle, working, AND has a queue.
        """
        self.waiting_process = None
        try:
            while True:
                # Find idle, working machines with waiting jobs
                decision_machines = []
                for i in range(NUM_MACHINES):
                    if (self.machines[i].count == 0 and 
                        len(self.machine_queues[i]) > 0 and
                        self.machine_states[i] == 0):
                        decision_machines.append(i)
                
                if not decision_machines:
                    # No decisions to make, wait for an event
                    yield self.env.timeout(100) # Will be interrupted
                else:
                    # A decision is needed!
                    self.waiting_process = self.env.event()
                    yield self.waiting_process # Block until step() provides an action
                    
                    action = self.waiting_process.value # Action provided by step()
                    
                    if action in decision_machines:
                        # Valid action: start processing the job
                        # TODO: Implement better queue logic (e.g., SPT, EDD)
                        # For now, we use simple FIFO
                        job = self.machine_queues[action].pop(0) 
                        self.env.process(self.process_operation(job, action))
                    else:
                        # Invalid action (machine busy, broken, or empty queue)
                        self.last_action_invalid = True
                        
        except simpy.Interrupt:
            # Interrupted by a job arrival, repair, or completion
            pass # The loop will restart and find the new decision point

    def process_operation(self, job, machine_id):
        """ 
        A simpy process for a machine processing a single job operation.
        This can be interrupted by a breakdown.
        """
        op = job.get_next_operation()
        if not op or op[0] != machine_id:
            print(f"Error: Job {job.job_id} mismatch on Machine {machine_id}")
            return

        proc_time = op[1]
        print(f"Time {self.env.now:.2f}: Machine {machine_id} START Job {job.job_id} (Op {job.current_op}). Duration {proc_time}")
        
        try:
            # Request the machine with default priority (0)
            with self.machines[machine_id].request(priority=0) as req:
                yield req 
                yield self.env.timeout(proc_time) # Process the job
            
            # --- Operation successful (not interrupted) ---
            print(f"Time {self.env.now:.2f}: Machine {machine_id} END Job {job.job_id} (Op {job.current_op}).")
            job_finished = job.complete_operation()
            
            if job_finished:
                job.completion_time = self.env.now
                self.pending_jobs.remove(job)
                self.completed_jobs.append(job)
                print(f"Time {self.env.now:.2f}: Job {job.job_id} COMPLETED.")
            else:
                # Place job in queue for its next operation
                self.place_job_in_queue(job)
                
        except simpy.Interrupt:
            # --- Operation INTERRUPTED (by breakdown) ---
            print(f"Time {self.env.now:.2f}: !!! INTERRUPT !!! Job {job.job_id} on Machine {machine_id} stopped by breakdown.")
            # Do NOT complete operation. Re-queue the job for the same machine.
            self.machine_queues[machine_id].insert(0, job) # Re-queue at the front
            
        # After completion OR interrupt, this machine's state might have changed
        # or it might be free. Interrupt the decision loop to re-evaluate.
        if self.decision_point_process and self.decision_point_process.is_alive:
            self.decision_point_process.interrupt()

    def step(self, action):
        """
        Apply an action to the environment.
        """
        self.last_action_invalid = False # Reset flag
        
        if self.waiting_process and not self.waiting_process.triggered:
            self.waiting_process.succeed(value=action)
            self.waiting_process = None # Clear the event
        else:
            # Agent took a step when no decision was needed
            self.last_action_invalid = True
            
        # Run the simulation until the next decision point or termination
        try:
            self.env.run()
        except simpy.core.EmptySchedule:
            # Simulation finished
            pass
            
        # Calculate state, reward, done, info
        obs = self._get_observation()
        
        # Check for termination
        terminated = len(self.completed_jobs) >= JOBS_PER_EPISODE and len(self.pending_jobs) == 0
        
        if self.env.now > 1000: # Time limit
            terminated = True
            
        truncated = False 
        
        # --- Reward Calculation ---
        # Import a function from reward_functions.py to keep this clean
        # For now, a simple example:
        from reward_functions import calculate_reward_dynamic
        reward = calculate_reward_dynamic(self, terminated)

        return obs, reward, terminated, truncated, self._get_info()
        
    def _get_info(self):
        """ Return auxiliary info. """
        return {
            "completed_jobs": len(self.completed_jobs),
            "pending_jobs": len(self.pending_jobs),
            "time": self.env.now,
            "machine_states": self.machine_states.copy()
        }

    def render(self, mode='human'):
        """ Print a simple representation of the environment state. """
        print("-" * 50)
        print(f"Current Time: {self.env.now:.2f}")
        for i in range(NUM_MACHINES):
            queue_str = ", ".join([f"J{job.job_id}({job.priority})" for job in self.machine_queues[i]])
            
            machine_status = "BUSY"
            if self.machine_states[i] == 1.0:
                machine_status = "BROKEN"
            elif self.machines[i].count == 0:
                machine_status = "IDLE"
                
            print(f"Machine {i} [{machine_status}]: Queue = [{queue_str}]")
        print(f"Completed: {len(self.completed_jobs)}, Pending: {len(self.pending_jobs)}")
        print("-" * 50)

# Example of how to run the environment standalone
if __name__ == "__main__":
    env = JobShopEnv()
    obs, info = env.reset()
    env.render()
    
    terminated = False
    total_reward = 0
    
    # Simple "policy": always choose the first available machine
    while not (terminated):
        # Find first idle, working machine with a queue
        action = -1
        for i in range(NUM_MACHINES):
            if (env.machines[i].count == 0 and 
                len(env.machine_queues[i]) > 0 and
                env.machine_states[i] == 0):
                action = i
                break
                
        if action == -1:
            # No valid action, take a random "no-op"
            action = env.action_space.sample() 
            print("No valid move, taking random action.")
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
            
    print(f"Simulation finished. Total Reward: {total_reward}")
    print(f"Final Time (Makespan): {env.env.now}")
    env.render()

