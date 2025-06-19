import sys
import numpy as np
from typing import List, Dict, Tuple, Set
from pathlib import Path
import json
from datetime import datetime
import itertools
import random

# Add parent directory to path to import from mdps
sys.path.append(str(Path(__file__).parent.parent.parent))
from mdps.combination_lock import CombinationLock

class BayesOptimalAgent:
    def __init__(self, combination_length: int = 3, max_attempts: int = 8, vocab: str = '0123456789'):
        self.combination_length = combination_length
        self.max_attempts = max_attempts
        self.vocab = vocab
        self.all_combinations = self._generate_all_combinations()
        self.untested_chars = set(vocab)  # Characters not yet tested
        self.found_chars = set()  # Characters we know are in the combination
        self.known_positions = [None] * combination_length  # Known positions of characters
        
    def _generate_all_combinations(self) -> List[str]:
        """Generate all possible combinations with distinct characters."""
        chars = list(self.vocab)
        combinations = []
        for combo in itertools.permutations(chars, self.combination_length):
            combinations.append(''.join(combo))
        return combinations
    
    def _is_feedback_consistent(self, guess: str, target: str, observed_feedback: List[int]) -> bool:
        """Check if the observed feedback is consistent with guess and target."""
        expected_feedback = self._compute_feedback(guess, target)
        return expected_feedback == observed_feedback
    
    def _compute_feedback(self, guess: str, target: str) -> List[int]:
        """Compute the feedback for a guess given the target."""
        feedback = []
        target_chars = list(target)
        guess_chars = list(guess)
        
        # First pass: mark correct positions
        for i in range(len(guess_chars)):
            if guess_chars[i] == target_chars[i]:
                feedback.append(2)  # Correct position
                target_chars[i] = None
            else:
                feedback.append(0)
        
        # Second pass: mark correct digits in wrong positions
        for i in range(len(guess_chars)):
            if feedback[i] != 2:
                if guess_chars[i] in target_chars:
                    feedback[i] = 1  # Wrong position
                    target_chars[target_chars.index(guess_chars[i])] = None
        
        return feedback
    
    def _update_belief(self, guess: str, feedback: List[int]) -> None:
        """Update belief based on new feedback."""
        guess_chars = list(guess)
        
        # Update untested and found characters
        for i, char in enumerate(guess_chars):
            if feedback[i] > 0:  # Character is in the combination
                self.found_chars.add(char)
            self.untested_chars.discard(char)
            
            # Update known positions
            if feedback[i] == 2:  # Correct position
                self.known_positions[i] = char
    
    def select_action(self, history: List[Tuple[str, List[int]]]) -> str:
        """Select the Bayes-optimal action given history."""
        if not history:
            # First guess: sample uniformly from untested characters
            chars = list(self.untested_chars)
            random.shuffle(chars)
            return ''.join(chars[:self.combination_length])
        
        # If we know all positions, try the correct combination
        if all(pos is not None for pos in self.known_positions):
            return ''.join(self.known_positions)
        
        # If we have found all characters but not their positions
        if len(self.found_chars) == self.combination_length:
            # Try a permutation of the found characters
            chars = list(self.found_chars)
            random.shuffle(chars)
            return ''.join(chars)
        
        # Otherwise, sample uniformly from remaining untested characters
        chars = list(self.untested_chars)
        random.shuffle(chars)
        return ''.join(chars[:self.combination_length])


def save_game_log(game_id: int, history: List[Tuple[str, List[int]]], success: bool, 
                  target: str, belief_sizes: List[int]):
    """Save game log to a JSONL file."""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_entry = {
        "game_id": game_id,
        "timestamp": datetime.now().isoformat(),
        "target_combination": target,
        "success": success,
        "num_attempts": len(history),
        "belief_sizes": belief_sizes,
        "history": [
            {
                "attempt": i + 1,
                "guess": guess,
                "feedback": feedback,
                "feedback_str": ''.join(['⬜' if f == 0 else '🟨' if f == 1 else '🟩' for f in feedback])
            }
            for i, (guess, feedback) in enumerate(history)
        ]
    }
    
    log_file = log_dir / "bayes_optimal.jsonl"
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


def play_single_game(agent: BayesOptimalAgent, game_id: int) -> Tuple[bool, int, List[float]]:
    """Play a single game with Bayes-optimal agent. Returns (success, num_attempts, regret_per_attempt)."""
    mdp = CombinationLock(vocab=agent.vocab)
    mdp.reset(seed=game_id)
    
    history = []
    belief_sizes = []
    regret_per_attempt = []
    
    while len(history) < mdp.max_attempts:
        # Select action based on current belief
        guess = agent.select_action(history)
        
        # Take action
        obs, reward, done, info = mdp.step(guess)
        feedback = info['feedback']
        history.append((guess, feedback))
        
        # Update belief
        agent._update_belief(guess, feedback)
        belief_sizes.append(len(agent.untested_chars) + len(agent.found_chars))
        
        # Calculate regret as shortfall against optimal value function
        # V*(s) = 1, V^π(s) = 1 if solved, 0 if not solved
        regret_per_attempt.append(1.0 - (1.0 if done and reward == 1.0 else 0.0))
        
        if done and reward == 1.0:
            save_game_log(game_id, history, True, mdp.target_combination, belief_sizes)
            return True, len(history), regret_per_attempt
    
    # Failed to find combination
    save_game_log(game_id, history, False, mdp.target_combination, belief_sizes)
    return False, len(history), regret_per_attempt


def main():
    # Test with different vocabularies and combination lengths
    test_configs = [
        (3, '0123456789'),  # Original 3-digit case
        (3, '!@#$%^&*pqrs5678'),  # Same as prompting.py
        (4, '0123456789'),  # 4-digit case
        (3, 'abcdefghijklmnopqrstuvwxyz'),  # Letters only
    ]
    
    for combination_length, vocab in test_configs:
        agent = BayesOptimalAgent(combination_length=combination_length, vocab=vocab)
        num_games = 100
        
        # Play games
        results = []
        total_regret = 0.0
        
        print(f"\nRunning Bayes-optimal agent on {num_games} games...")
        print(f"Combination length: {combination_length}, Vocabulary: {vocab}")
        for game_id in range(num_games):
            success, attempts, regret_per_attempt = play_single_game(agent, game_id)
            results.append((success, attempts, regret_per_attempt))
            total_regret += sum(regret_per_attempt)
            
            if (game_id + 1) % 10 == 0:
                print(f"Completed {game_id + 1} games...")
        
        # Calculate statistics
        wins = sum(1 for success, _, _ in results if success)
        total_attempts = sum(attempts for _, attempts, _ in results)
        avg_attempts = total_attempts / len(results)
        
        print(f"\n{'='*50}")
        print(f"BAYES-OPTIMAL AGENT RESULTS ({num_games} games)")
        print(f"Combination length: {combination_length}, Vocabulary: {vocab}")
        print(f"{'='*50}")
        print(f"Win rate: {wins}%")
        print(f"Average attempts per game: {avg_attempts:.2f}")
        print(f"Total regret: {total_regret:.1f}")
        print(f"Average regret per game: {total_regret/num_games:.3f}")
        
        # Print distribution of attempts
        attempt_dist = {}
        for _, attempts, _ in results:
            attempt_dist[attempts] = attempt_dist.get(attempts, 0) + 1
        
        print("\nAttempt distribution:")
        for attempts in sorted(attempt_dist.keys()):
            print(f"{attempts} attempts: {attempt_dist[attempts]} games")
        
        # Calculate cumulative regret by attempt number
        regret_by_attempts = {}
        for _, _, regret_per_attempt in results:
            for attempt_num, regret in enumerate(regret_per_attempt, 1):
                regret_by_attempts[attempt_num] = regret_by_attempts.get(attempt_num, 0) + regret
        
        print("\nAverage cumulative regret by attempt number:")
        cumulative_regret = 0
        for attempt_num in sorted(regret_by_attempts.keys()):
            avg_regret = regret_by_attempts[attempt_num] / num_games
            cumulative_regret += avg_regret
            print(f"After {attempt_num} attempts: {cumulative_regret:.3f} regret")
        
        # Store results in JSON file
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "combination_length": combination_length,
            "vocab": vocab,
            "num_games": num_games,
            "win_rate": wins,
            "avg_attempts": avg_attempts,
            "total_regret": total_regret,
            "avg_regret_per_game": total_regret / num_games,
            "attempt_distribution": attempt_dist,
            "cumulative_regret": {
                str(attempt_num): cumulative_regret
                for attempt_num, cumulative_regret in enumerate(
                    [sum(regret_by_attempts.get(i, 0) / num_games for i in range(1, j + 1))
                     for j in range(1, max(regret_by_attempts.keys()) + 1)],
                    1
                )
            }
        }
        
        results_file = results_dir / f"bayes_optimal_results_l{combination_length}_v{len(vocab)}.json"
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Analyze failure cases
        failures = [(i, attempts) for i, (success, attempts, _) in enumerate(results) if not success]
        if failures:
            print(f"\nFailed games: {len(failures)}")
            print("Game IDs of failures:", [game_id for game_id, _ in failures])
        else:
            print("\nNo failures! Perfect performance.")


if __name__ == "__main__":
    main()