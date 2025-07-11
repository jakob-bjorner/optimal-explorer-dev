import sys
from typing import List, Dict, Tuple
from pathlib import Path
import json
from datetime import datetime
import itertools
import random
import argparse
import copy
import math

# Add parent directory to path to import from mdps
sys.path.append(str(Path(__file__).parent.parent.parent))
from mdps.bandits import MultiArmedBandits


class UCBAgent:
    def __init__(self, arm_names=['blue', 'green', 'red', 'yellow', 'purple'], max_steps=20, alpha=1.0):
        self.arm_names = arm_names
        self.max_steps = max_steps
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.counts = {arm: 0 for arm in self.arm_names}
        self.sums = {arm: 0.0 for arm in self.arm_names}
        self.t = 0

    def _update_belief(self, arm: str, reward: float) -> None:
        self.counts[arm] += 1
        self.sums[arm] += reward
        self.t += 1

    def select_action(self, history: List[Tuple[str, float]]) -> str:
        # Update beliefs from history if not already up to date
        if self.t < len(history):
            self.reset()
            for a, r in history:
                self._update_belief(a, r)
        # UCB selection
        ucb_values = {}
        for arm in self.arm_names:
            n = self.counts[arm]
            if n == 0:
                ucb_values[arm] = float('inf')  # Ensure each arm is pulled at least once
            else:
                mean = self.sums[arm] / n
                bonus = self.alpha * math.sqrt(math.log(max(1, self.t + 1)) / n)
                ucb_values[arm] = mean + bonus
        # Select arm with highest UCB value
        best_arm = max(self.arm_names, key=lambda a: ucb_values[a])
        return best_arm


class RandomAgent:
    def __init__(self, arm_names=['blue', 'green', 'red', 'yellow', 'purple'], max_steps=20):
        self.arm_names = arm_names
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        pass  # No state to reset for random agent

    def _update_belief(self, arm: str, reward: float) -> None:
        pass  # Random agent does not update beliefs

    def select_action(self, history: List[Tuple[str, float]]) -> str:
        return random.choice(self.arm_names)


def save_game_log(game_id: int, history: List[Tuple[str, float]], total_reward: float, regret_per_attempt: list, model: str, env_config: dict):
    """Save game log to a JSONL file (log to {model}.jsonl)."""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_entry = {
        "game_id": game_id,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "env_config": env_config,
        "total_reward": total_reward,
        "num_attempts": len(history),
        "regret_per_attempt": regret_per_attempt,
        "cumulative_regret": float(sum(regret_per_attempt)),
        "history": [
            {
                "attempt": i + 1,
                "action": a,
                "reward": reward,
                "regret": regret_per_attempt[i] if i < len(regret_per_attempt) else None
            }
            for i, (a, reward) in enumerate(history)
        ]
    }
    log_file = log_dir / f"{model}_{env_config.get('reward_type', 'bernoulli')}.jsonl"
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


def play_single_game(agent, game_id: int, model: str, env_config: dict) -> Tuple[float, int, List[float]]:
    """Play a single game with the given agent. Returns (total_reward, num_attempts, regret_per_attempt)."""
    mdp = MultiArmedBandits(
        arm_names=env_config.get('arm_names'),
        num_arms=env_config.get('num_arms', 5),
        max_steps=env_config.get('max_steps', 50),
        reward_type=env_config.get('reward_type', 'gaussian'),
        noise_level=env_config.get('noise_level', 'medium'),
        seed=game_id
    )
    mdp.reset(seed=game_id)
    agent.reset()
    history = []
    regret_per_attempt = []
    total_reward = 0.0
    for _ in range(mdp.max_steps):
        a = agent.select_action(history)
        obs, reward, done, info = mdp.step(a)
        history.append((a, reward))
        agent._update_belief(a, reward)
        action_index = mdp.arm_names.index(a)
        # Regret: difference between optimal mean and chosen arm mean
        regret = float(max(mdp.arm_means) - mdp.arm_means[action_index])
        regret_per_attempt.append(regret)
        total_reward += reward
        if done:
            break
    save_game_log(game_id, history, total_reward, regret_per_attempt, model, env_config)
    return total_reward, len(history), regret_per_attempt


def run_agent(agent_class, model_name, num_games, arm_names, env_config, alpha=None, max_steps=50):
    results = []
    total_regret = 0.0
    total_cumulative_regret = 0.0
    total_reward = 0.0
    all_regret_per_attempt = []

    # clear the log file
    log_file = Path(__file__).parent / "logs" / f"{model_name}_{env_config.get('reward_type', 'bernoulli')}.jsonl"
    if log_file.exists():
        log_file.unlink()
    else:
        log_file.touch()
    print(f"Cleared log file: {log_file}")

    for game_id in range(num_games):
        if model_name == "Random":
            agent = agent_class(max_steps=max_steps, arm_names=arm_names)
        else:
            agent = agent_class(max_steps=max_steps, arm_names=arm_names, alpha=alpha)
        game_total_reward, attempts, regret_per_attempt = play_single_game(agent, game_id, model_name, env_config)
        results.append((game_total_reward, attempts, regret_per_attempt))
        total_reward += game_total_reward
        total_regret += sum(regret_per_attempt)
        total_cumulative_regret += sum(regret_per_attempt)
        all_regret_per_attempt.extend(regret_per_attempt)

    total_attempts = sum(attempts for _, attempts, _ in results)
    avg_attempts = total_attempts / len(results)
    avg_reward = total_reward / num_games
    avg_regret = total_regret / num_games
    avg_cumulative_regret = total_cumulative_regret / num_games

    print(f"\n{'='*50}")
    print(f"{model_name.upper()} AGENT RESULTS ({num_games} games)")
    print(f"Arms: {env_config['arm_names']}, Reward type: {env_config['reward_type']}, Noise: {env_config['noise_level']}")
    print(f"{'='*50}")
    print(f"Average total reward per game: {avg_reward:.3f}")
    print(f"Average attempts per game: {avg_attempts:.2f}")
    print(f"Total regret: {total_regret:.3f}")
    print(f"Average regret per game: {avg_regret:.3f}")
    print(f"Average cumulative regret per game: {avg_cumulative_regret:.3f}")
    print(f"Cumulative regret per attempt (mean over all attempts): {sum(all_regret_per_attempt)/len(all_regret_per_attempt):.4f}")
    print(f"Total games: {num_games}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=50, help="Arm pull budget")
    parser.add_argument("--num_arms", type=int, default=10, help="Number of arms")
    parser.add_argument("--arm_names", type=str, default=None, help="Arm names (comma-separated, default: 0,1,...)")
    parser.add_argument("--reward_type", type=str, default="bernoulli", help="Reward type: gaussian or bernoulli")
    parser.add_argument("--noise_level", type=str, default="medium", help="Noise level: low, medium, high")
    parser.add_argument("--num_games", type=int, default=50, help="Number of games to play")
    parser.add_argument("--model", type=str, default="ucb_mab", help="Model name (for logging)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Exploration-exploitation tradeoff parameter")
    args = parser.parse_args()

    num_games = args.num_games

    if args.arm_names is not None:
        arm_names = args.arm_names.split(",")
    else:
        arm_names = [str(i+1) for i in range(args.num_arms)]

    env_config = {
        'arm_names': arm_names,
        'num_arms': len(arm_names),
        'max_steps': args.max_steps,
        'reward_type': args.reward_type,
        'noise_level': args.noise_level
    }

    # Run UCB agent
    run_agent(
        agent_class=UCBAgent,
        model_name=args.model,
        num_games=num_games,
        arm_names=arm_names,
        env_config=env_config,
        alpha=args.alpha,
        max_steps=args.max_steps
    )

    # Run Random agent
    run_agent(
        agent_class=RandomAgent,
        model_name="Random",
        num_games=num_games,
        arm_names=arm_names,
        env_config=env_config,
        max_steps=args.max_steps
    )


if __name__ == "__main__":
    main()