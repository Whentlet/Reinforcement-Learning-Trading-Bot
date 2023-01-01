import gym
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import argparse
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dummy Trading Environment (Simplified for demonstration)
class TradingEnv(gym.Env):
    """
    A custom OpenAI Gym environment for simulating stock trading.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, df, window_size=10):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.current_step = self.window_size
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.episode_history = []

        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = gym.spaces.Discrete(3)

        # Observation space: window_size * (OHLCV) + balance + shares_held
        # For simplicity, let's assume 5 features per step (Open, High, Low, Close, Volume)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                                shape=(self.window_size * 5 + 2,), dtype=np.float32)

    def _get_observation(self):
        obs = self.df.loc[self.current_step - self.window_size : self.current_step - 1, ['Open', 'High', 'Low', 'Close', 'Volume']].values.flatten()
        return np.append(obs, [self.balance, self.shares_held])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.episode_history = []
        logging.info("Environment reset.")
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.df):
            terminated = True
            reward = self.net_worth - self.initial_balance # Final reward
            logging.info(f"Episode terminated. Final Net Worth: {self.net_worth}")
            return self._get_observation(), reward, terminated, False, {}

        current_price = self.df.loc[self.current_step - 1, 'Close']

        if action == 1: # Buy
            if self.balance > current_price:
                self.shares_held += 1
                self.balance -= current_price
                logging.debug(f"Bought 1 share at {current_price}")
        elif action == 2: # Sell
            if self.shares_held > 0:
                self.shares_held -= 1
                self.balance += current_price
                logging.debug(f"Sold 1 share at {current_price}")

        self.net_worth = self.balance + self.shares_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # Reward calculation (simple: change in net worth)
        reward = self.net_worth - self.episode_history[-1]['net_worth'] if self.episode_history else 0

        self.episode_history.append({
            'step': self.current_step,
            'balance': self.balance,
            'shares': self.shares_held,
            'net_worth': self.net_worth,
            'action': action,
            'price': current_price
        })

        observation = self._get_observation()
        terminated = False
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

def generate_dummy_stock_data(num_days=1000):
    dates = pd.date_range(start="2020-01-01", periods=num_days)
    data = {
        'Open': np.random.uniform(100, 200, num_days),
        'High': np.random.uniform(200, 300, num_days),
        'Low': np.random.uniform(50, 100, num_days),
        'Close': np.random.uniform(100, 200, num_days),
        'Volume': np.random.randint(100000, 1000000, num_days)
    }
    df = pd.DataFrame(data, index=dates)
    df['High'] = df[['Open', 'Close']].max(axis=1) + np.random.uniform(0, 10, num_days)
    df['Low'] = df[['Open', 'Close']].min(axis=1) - np.random.uniform(0, 10, num_days)
    df['Close'] = df['Open'] + np.random.uniform(-10, 10, num_days)
    df = df.apply(lambda x: x.abs())
    return df

def main():
    parser = argparse.ArgumentParser(description="Train RL Trading Bot")
    parser.add_argument("--algo", type=str, default="ppo", help="RL Algorithm (ppo, dqn)")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total timesteps for training")
    args = parser.parse_args()

    logging.info(f"Starting training with {args.algo.upper()} for {args.timesteps} timesteps...")

    # Generate dummy historical data
    df = generate_dummy_stock_data(num_days=2000)

    # Create environment
    env = DummyVecEnv([lambda: TradingEnv(df)])

    # Initialize model
    if args.algo.lower() == "ppo":
        model = PPO("MlpPolicy", env, verbose=1)
    else:
        raise ValueError("Unsupported algorithm. Use 'ppo'.")

    # Train model
    model.learn(total_timesteps=args.timesteps)

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save(f"models/{args.algo}_trading_model")
    logging.info("Training completed and model saved.")

if __name__ == "__main__":
    main()
