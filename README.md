# Reinforcement Learning Trading Bot

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This repository contains an autonomous trading bot that utilizes Deep Reinforcement Learning (DRL) to discover and execute optimal trading strategies in simulated financial markets.

## 🚀 Features
- **Custom Gym Environment**: A tailored OpenAI Gym environment for financial trading.
- **DRL Algorithms**: Implementation of Proximal Policy Optimization (PPO) and Deep Q-Network (DQN).
- **Backtesting Engine**: Robust backtesting framework to evaluate strategy performance on historical data.
- **Risk Management**: Built-in risk management protocols to limit drawdowns.

## 🛠️ Technologies
- Python
- OpenAI Gym
- Stable Baselines3
- Pandas, NumPy
- Matplotlib (for visualization)

## ⚙️ Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Whentlet/Reinforcement-Learning-Trading-Bot.git
   cd Reinforcement-Learning-Trading-Bot
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run a training session:
   ```bash
   python train.py --algo ppo --timesteps 100000
   ```

## 📈 Usage
After training, use the `evaluate.py` script to test the model's performance on unseen historical data and visualize the trading actions.

## 📄 License
This project is licensed under the MIT License.
