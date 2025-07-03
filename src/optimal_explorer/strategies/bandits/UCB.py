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
from mdps.bandits import BanditBestArmSelection

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


def save_game_log(game_id: int, history: List[Tuple[str, float]], success: bool, 
                  best_arm: str, final_answer: str, model: str):
    """Save game log to a JSONL file (log to bayes_optimal.jsonl)."""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_entry = {
        "game_id": game_id,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "best_arm": best_arm,
        "final_answer": final_answer,
        "success": success,
        "num_attempts": len(history),
        "history": [
            {
                "attempt": i + 1,
                "action": a,
                "reward": reward,
            }
            for i, (a, reward) in enumerate(history)
        ]
    }
    log_file = log_dir / f"{model}.jsonl"
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


def play_single_game(agent: UCBAgent, game_id: int, model: str) -> Tuple[bool, int, List[float]]:
    """Play a single game with UCB agent. Returns (success, num_attempts, regret_per_attempt)."""
    mdp = BanditBestArmSelection(arm_names=agent.arm_names, max_steps=agent.max_steps)
    mdp.reset(seed=game_id)
    agent.reset()
    history = []
    regret_per_attempt = []
    # Exploration phase
    for _ in range(mdp.max_steps):
        a = agent.select_action(history)
        obs, reward, done, info = mdp.step(a)
        reward = info.get('reward')
        history.append((a, reward))
        agent._update_belief(a, reward)
        action_index = mdp.arm_names.index(a)
        regret_per_attempt.append(mdp.arm_means[mdp.best_arm] - mdp.arm_means[action_index])
    # Final answer phase
    # Agent selects the arm with the highest estimated mean
    estimated_means = {arm: (agent.sums[arm] / agent.counts[arm]) if agent.counts[arm] > 0 else float('-inf') for arm in agent.arm_names}
    final_answer = max(agent.arm_names, key=lambda a: estimated_means[a])
    obs, reward, done, info = mdp.step(final_answer)
    success = info.get('correct', False)
    save_game_log(game_id, history, success, mdp.best_arm_name, final_answer, model)
    return success, len(history), regret_per_attempt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=20, help="Arm pull budget")
    parser.add_argument("--arm_names", type=str, default="blue,green,red,yellow,purple", help="Arm names")
    parser.add_argument("--num-games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--model", type=str, default="UCB", help="Model name (for logging)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Exploration-exploitation tradeoff parameter")
    args = parser.parse_args()

    num_games = args.num_games
    model = args.model

    results = []
    total_regret = 0.0
    for game_id in range(num_games):
        agent = UCBAgent(max_steps=args.max_steps, arm_names=args.arm_names.split(","), alpha=args.alpha)
        success, attempts, regret_per_attempt = play_single_game(agent, game_id, model)
        results.append((success, attempts, regret_per_attempt))
        total_regret += sum(regret_per_attempt)
    wins = sum(1 for success, _, _ in results if success)
    total_attempts = sum(attempts for _, attempts, _ in results)
    avg_attempts = total_attempts / len(results)
    print(f"UCB AGENT RESULTS ({num_games} games)")
    print(f"Win rate: {wins}/{num_games} ({100.0 * wins / num_games:.1f}%)")
    print(f"Average attempts per game: {avg_attempts:.2f}")
    print(f"Total regret: {total_regret:.1f}")
    print(f"Average regret per game: {total_regret/num_games:.3f}")


if __name__ == "__main__":
    main()