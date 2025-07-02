import sys
from typing import List, Dict, Tuple, Set
from pathlib import Path
import json
from datetime import datetime
import itertools
import random
import argparse
import copy

# Add parent directory to path to import from mdps
sys.path.append(str(Path(__file__).parent.parent.parent))
from mdps.bandits import BanditBestArmSelection

class UCBAgent:
    def __init__(self, arm_names=['blue', 'green', 'red', 'yellow', 'purple'], max_steps=20):
        self.arm_names = arm_names
        self.max_steps = max_steps
        self.beliefs = None # TODO: implement this
        
    
    def _update_belief(self, arm: str, reward: int) -> None:
        """Update belief based on new feedback."""
        self.beliefs = None # TODO: implement this
    
    def select_action(self, history: List[Tuple[str, int]]) -> str:
        """Select the best action given history following the UCB algorithm."""
         # TODO: implement this
        return None


def save_game_log(game_id: int, history: List[Tuple[str, List[int]]], success: bool, 
                  target: str, model: str):
    """Save game log to a JSONL file (no prompt_style, log to bayes_optimal.jsonl)."""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_entry = {
        "game_id": game_id,
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "target_combination": target,
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
    """Play a single game with Bayes-optimal agent. Returns (success, num_attempts, regret_per_attempt)."""
    mdp = BanditBestArmSelection(arm_names=agent.arm_names, max_steps=agent.max_steps)
    mdp.reset(seed=game_id)
    history = []
    regret_per_attempt = []
    while len(history) < mdp.max_attempts:
        a = agent.select_action(history)
        obs, reward, done, info = mdp.step(a)
        reward = info.get('reward')
        history.append((a, reward))
        agent._update_belief(a, reward)
        regret_per_attempt.append(mdp.arm_means[mdp.best_arm] - mdp.arm_means[a])
        if done and reward == 1.0:
            save_game_log(game_id, history, True, mdp.target_combination, model)
            return True, len(history), regret_per_attempt
    save_game_log(game_id, history, False, mdp.target_combination, model)
    return False, len(history), regret_per_attempt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=20, help="Arm pull budget")
    parser.add_argument("--arm_names", type=str, default="blue,green,red,yellow,purple", help="Arm names")
    parser.add_argument("--num-games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--model", type=str, default="UCB", help="Model name (for logging)")
    args = parser.parse_args()

    num_games = args.num_games
    model = args.model

    results = []
    total_regret = 0.0
    print(f"\nRunning UCB agent on {num_games} games...")
    print(f"Pull budget: {args.max_steps}, Arms: {args.arm_names}")
    for game_id in range(num_games):
        # Reset agent state for each game
        agent = UCBAgent(max_steps=args.max_steps, arm_names=args.arm_names.split(","))
        success, attempts, regret_per_attempt = play_single_game(agent, game_id, model)
        results.append((success, attempts, regret_per_attempt))
        total_regret += sum(regret_per_attempt)
        if (game_id + 1) % 10 == 0:
            print(f"Completed {game_id + 1} games...")
    wins = sum(1 for success, _, _ in results if success)
    total_attempts = sum(attempts for _, attempts, _ in results)
    avg_attempts = total_attempts / len(results)
    print(f"\n{'='*50}")
    print(f"UCB AGENT RESULTS ({num_games} games)")
    print(f"Pull budget: {args.max_steps}, Arms: {args.arm_names}")
    print(f"{'='*50}")
    print(f"Win rate: {wins}%")
    print(f"Average attempts per game: {avg_attempts:.2f}")
    print(f"Total regret: {total_regret:.1f}")
    print(f"Average regret per game: {total_regret/num_games:.3f}")
    attempt_dist = {}
    for _, attempts, _ in results:
        attempt_dist[attempts] = attempt_dist.get(attempts, 0) + 1
    print("\nAttempt distribution:")
    for attempts in sorted(attempt_dist.keys()):
        print(f"{attempts} attempts: {attempt_dist[attempts]} games")
    failures = [(i, attempts) for i, (success, attempts, _) in enumerate(results) if not success]
    if failures:
        print(f"\nFailed games: {len(failures)}")
        print("Game IDs of failures:", [game_id for game_id, _ in failures])
    else:
        print("\nNo failures! Perfect performance.")


if __name__ == "__main__":
    main()