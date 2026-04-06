"""
ER-Sentinel DQN Training Script
================================
Trains the DQN agent on the Hospital Triage environment automatically.
Reports average reward every 500 episodes.

Usage:
    python train.py                     # defaults: 5000 episodes, hard difficulty
    python train.py --epochs 10000      # custom episode count
    python train.py --difficulty easy   # custom difficulty
"""

import argparse
import time
import os
import sys
import numpy as np
from tqdm import tqdm

from env.triage_env import HospitalTriageEnv
from training.dqn_agent import DQNAgent


def train(num_epochs: int, difficulty: str, report_interval: int = 500):
    env = HospitalTriageEnv()
    agent = DQNAgent(
        state_dim=HospitalTriageEnv.STATE_DIM,
        action_dim=HospitalTriageEnv.ACTION_DIM,
    )

    print("=" * 60)
    print(f"  🏥 ER-Sentinel — DQN Training")
    print(f"  Episodes : {num_epochs}")
    print(f"  Difficulty: {difficulty}")
    print(f"  Report every: {report_interval} episodes")
    print("=" * 60)

    all_rewards = []
    best_avg = -float("inf")
    start_time = time.time()

    pbar = tqdm(range(1, num_epochs + 1), desc="Training", unit="ep",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")

    for ep in pbar:
        obs = env.reset(task_id=difficulty)
        state = env.obs_to_vector(obs)
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            obs, reward, done = env.step(action)
            next_state = env.obs_to_vector(obs)

            # Store transition & learn
            agent.memory.append((state, action, reward, next_state, float(done)))
            agent.train_step()

            state = next_state
            episode_reward += reward

        all_rewards.append(episode_reward)

        # ── Periodic Report ──────────────────────────────────
        if ep % report_interval == 0:
            window = all_rewards[-report_interval:]
            avg = np.mean(window)
            mn, mx = np.min(window), np.max(window)
            elapsed = time.time() - start_time

            tag = ""
            if avg > best_avg:
                best_avg = avg
                tag = "  ⭐ new best"

            tqdm.write(
                f"  EP {ep:>6}/{num_epochs}  |  "
                f"avg {avg:>7.2f}  min {mn:>7.2f}  max {mx:>7.2f}  |  "
                f"ε {agent.epsilon:.4f}  |  "
                f"{elapsed:>6.1f}s{tag}"
            )

    pbar.close()

    # ── Final Summary ────────────────────────────────────────
    total_time = time.time() - start_time
    final_avg = np.mean(all_rewards[-report_interval:])
    print("\n" + "=" * 60)
    print(f"  ✅ Training Complete")
    print(f"  Total time  : {total_time:.1f}s")
    print(f"  Final avg   : {final_avg:.2f}")
    print(f"  Best avg    : {best_avg:.2f}")
    print(f"  Final ε     : {agent.epsilon:.4f}")
    print("=" * 60)

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ER-Sentinel DQN Trainer")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training episodes")
    parser.add_argument("--difficulty", type=str, default="hard", choices=["easy", "medium", "hard"])
    parser.add_argument("--report", type=int, default=500, help="Report interval (episodes)")
    args = parser.parse_args()

    train(args.epochs, args.difficulty, args.report)
