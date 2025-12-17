import os
import json
import optuna
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from JSSP_Simulation_Environment import JobShopEnv

# Configuration
LOG_DIR = "./logs/"
MODEL_PATH = "ppo_jssp_model"
PARAMS_PATH = "best_hyperparams.json"
OPTUNA_TRIALS = 50  # Number of hyperparameter sets to try

def ensure_directories():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

def objective(trial):
    """
    Optuna objective function modified to optimize for:
    1. Maximizing Jobs Completed
    2. Minimizing Total Tardiness
    3. Minimizing Final Makespan
    """
    env = JobShopEnv()
    env = Monitor(env)

    # Define the search space for PPO
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.9, 0.9999),
        "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048]),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "ent_coef": trial.suggest_float("ent_coef", 0.00001, 0.01, log=True),
    }

    # Initialize and train for a short duration to test parameter viability
    model = PPO("MlpPolicy", env, verbose=0, **params)
    model.learn(total_timesteps=10000) 

    # --- EVALUATION FOR SPECIFIC METRICS ---
    # We run 5 evaluation episodes to get a stable average of your target KPIs
    tardiness_results = []
    makespan_results = []
    completed_results = []

    for _ in range(5):
        obs, info = env.reset()
        done = False
        truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
        
        # Access the raw environment to calculate metrics
        raw_env = env.unwrapped
        
        # 1. Total Tardiness (Sum of Max(0, Completion - Due Date))
        ep_tardiness = sum(max(0, job.completion_time - job.due_date) for job in raw_env.completed_jobs)
        
        # 2. Makespan (Current simulation time at end)
        ep_makespan = raw_env.env.now
        
        # 3. Jobs Completed
        ep_completed = len(raw_env.completed_jobs)

        tardiness_results.append(ep_tardiness)
        makespan_results.append(ep_makespan)
        completed_results.append(ep_completed)

    avg_tardiness = np.mean(tardiness_results)
    avg_makespan = np.mean(makespan_results)
    avg_completed = np.mean(completed_results)

    # --- WEIGHTED SCORE CALCULATION ---
    # Optuna is set to MAXIMIZE this value.
    # We add jobs completed and subtract (penalize) tardiness and makespan.
    # Weights can be adjusted based on priority.
    score = (avg_completed * 100) - (avg_tardiness * 1.0) - (avg_makespan * 0.1)
    
    return score

def train_with_optuna():
    """Runs Optuna optimization and saves the best parameters."""
    ensure_directories()
    print("--- Starting Hyperparameter Optimization (Targeting Tardiness/Makespan) ---")
    
    # We maximize the score defined in the objective
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=OPTUNA_TRIALS)

    print(f"Best Trial Score: {study.best_value}")
    print(f"Best Params Found: {study.best_params}")

    # Save best parameters to JSON
    with open(PARAMS_PATH, "w") as f:
        json.dump(study.best_params, f)

    # Train the FINAL model using the best found hyperparameters
    print("--- Training final model with best parameters ---")
    env = JobShopEnv()
    env = Monitor(env)
    model = PPO("MlpPolicy", env, verbose=1, **study.best_params)
    model.learn(total_timesteps=50000)
    model.save(MODEL_PATH)

def test_agent():
    """Tests the agent using the best saved hyperparameters."""
    env = JobShopEnv()
    
    if os.path.exists(PARAMS_PATH):
        with open(PARAMS_PATH, "r") as f:
            best_params = json.load(f)
        print(f"Loaded best hyperparameters: {best_params}")

    if os.path.exists(MODEL_PATH + ".zip"):
        model = PPO.load(MODEL_PATH, env=env)
        
        # Run 10 test episodes to see performance
        total_t = []
        total_m = []
        total_c = []
        
        for i in range(10):
            obs, info = env.reset()
            done = truncated = False
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, truncated, info = env.step(action)
            
            raw = env.unwrapped
            tardiness = sum(max(0, j.completion_time - j.due_date) for j in raw.completed_jobs)
            total_t.append(tardiness)
            total_m.append(raw.env.now)
            total_c.append(len(raw.completed_jobs))
            
        print(f"\nTest Results (10 Episodes):")
        print(f"Avg Jobs Completed : {np.mean(total_c):.2f}")
        print(f"Avg Total Tardiness: {np.mean(total_t):.2f}")
        print(f"Avg Final Makespan : {np.mean(total_m):.2f}")
    else:
        print("Model file not found. Please train first.")