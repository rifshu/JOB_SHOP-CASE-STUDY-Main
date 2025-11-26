# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 23:32:49 2025

@author: SHAIK RIFSHU
"""

import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

from JSSP_Simulation_Environment import JobShopEnv # Import your custom environment

# --- CONFIGURATION ---
MODEL_DIR = "models/dqn"
LOG_DIR = "logs/dqn"
MODEL_NAME = "jssp_dqn"
TIMESTEPS_TO_TRAIN = 20000

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def create_env():
    """ Utility function to create and wrap the environment. """
    env = JobShopEnv()
    # Note: Environments returned by create_env are automatically wrapped
    # by stable-baselines3 with a TimeLimit wrapper.
    return env

def train_agent():
    """
    Train a DQN agent on the JobShopEnv.
    """
    print("--- Starting Agent Training ---")
    
    # Create a vectorized environment
    env = DummyVecEnv([lambda: create_env()])
    
    # Define the DQN model
    # 'MlpPolicy' is a standard feed-forward neural network
    # Stable-Baselines3 will automatically detect the new observation space
    model = DQN(
        'MlpPolicy', 
        env, 
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=1e-4,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=250,
        exploration_fraction=0.1,
        exploration_final_eps=0.02
    )
    
    # Setup a callback to save the model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=max(TIMESTEPS_TO_TRAIN // 10, 1000),
        save_path=MODEL_DIR,
        name_prefix=MODEL_NAME
    )
    
    # Train the agent
    model.learn(
        total_timesteps=TIMESTEPS_TO_TRAIN,
        callback=checkpoint_callback,
        tb_log_name=MODEL_NAME
    )
    
    # Save the final model
    final_model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_final")
    model.save(final_model_path)
    
    print(f"--- Training Complete. Model saved to {final_model_path} ---")
    
    # Evaluate the trained agent
    print("--- Evaluating Trained Agent ---")
    # We create a new, separate environment for evaluation
    eval_env = create_env()
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Evaluation: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")

def test_agent():
    """
    Test a trained DQN agent.
    """
    print("--- Testing Trained Agent ---")
    
    model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_final.zip")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please train the agent first by running: python main.py --mode train-rl")
        return

    # Load the trained model
    model = DQN.load(model_path)
    
    # Create a single environment to test on
    env = create_env()
    
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0
    
    while not (terminated or truncated):
        env.render()
        
        # Get the agent's action
        # We use deterministic=True for testing to get the best-known action
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        
        if terminated or truncated:
            print("--- Episode Finished ---")
            break
            
    env.render()
    print(f"Test finished. Total Reward: {total_reward:.2f}")
    print(f"Final Makespan: {env.env.now:.2f}")
    print(f"Completed Jobs: {info.get('completed_jobs', 0)}")

if __name__ == "__main__":
    # You can run this file directly to train
    train_agent()
    # Or to test
    # test_agent()