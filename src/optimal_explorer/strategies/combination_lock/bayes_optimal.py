import sys
import numpy as np
from typing import List, Dict, Tuple, Set
from pathlib import Path
import json
from datetime import datetime
import itertools

# Add parent directory to path to import from mdps
sys.path.append(str(Path(__file__).parent.parent.parent))
from mdps.combination_lock import CombinationLock

class BayesOptimalAgent:
    def __init__(self, combination_length: int = 3, max_attempts: int = 8):
        self.combination_length = combination_length
        self.max_attempts = max_attempts
        self.all_combinations = self._generate_all_combinations()
        
    def _generate_all_combinations(self) -> List[str]:
        """Generate all possible combinations with distinct digits."""
        digits = list(range(10))
        combinations = []
        for combo in itertools.permutations(digits, self.combination_length):
            combinations.append(''.join(map(str, combo)))
        return combinations
    
    def _is_feedback_consistent(self, guess: str, target: str, observed_feedback: List[int]) -> bool:
        """Check if the observed feedback is consistent with guess and target."""
        expected_feedback = self._compute_feedback(guess, target)
        return expected_feedback == observed_feedback
    
    def _compute_feedback(self, guess: str, target: str) -> List[int]:
        """Compute the feedback for a guess given the target."""
        feedback = []
        target_digits = list(target)
        guess_digits = list(guess)
        
        # First pass: mark correct positions
        for i in range(len(guess_digits)):
            if guess_digits[i] == target_digits[i]:
                feedback.append(2)  # Correct position
                target_digits[i] = None
            else:
                feedback.append(0)
        
        # Second pass: mark correct digits in wrong positions
        for i in range(len(guess_digits)):
            if feedback[i] != 2:
                if guess_digits[i] in target_digits:
                    feedback[i] = 1  # Wrong position
                    target_digits[target_digits.index(guess_digits[i])] = None
        
        return feedback
    
    def _update_belief(self, consistent_combinations: Set[str], guess: str, feedback: List[int]) -> Set[str]:
        """Update belief based on new feedback."""
        new_consistent = set()
        for combo in consistent_combinations:
            if self._is_feedback_consistent(guess, combo, feedback):
                new_consistent.add(combo)
        return new_consistent
    
    def _compute_information_gain(self, guess: str, consistent_combinations: Set[str]) -> float:
        """Compute expected information gain for a guess."""
        if guess in consistent_combinations:
            # If this guess is a possible combination, it could win immediately
            return float('inf')
        
        # Count how many combinations would give each possible feedback
        feedback_counts = {}
        for combo in consistent_combinations:
            feedback = tuple(self._compute_feedback(guess, combo))
            feedback_counts[feedback] = feedback_counts.get(feedback, 0) + 1
        
        # Compute entropy reduction
        total = len(consistent_combinations)
        entropy = 0
        for count in feedback_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        return entropy
    
    def select_action(self, history: List[Tuple[str, List[int]]], consistent_combinations: Set[str]) -> str:
        """Select the Bayes-optimal action given history and current belief."""
        remaining_attempts = self.max_attempts - len(history)
        
        if remaining_attempts == 1:
            # Last attempt: guess the most likely combination
            return list(consistent_combinations)[0]
        
        # If we have narrowed down to few possibilities, try them directly
        if len(consistent_combinations) <= remaining_attempts:
            return list(consistent_combinations)[0]
        
        # Otherwise, choose the guess that maximizes information gain
        best_guess = None
        best_score = -float('inf')
        
        for guess in self.all_combinations:
            score = self._compute_information_gain(guess, consistent_combinations)
            if score > best_score:
                best_score = score
                best_guess = guess
        
        return best_guess


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
    mdp = CombinationLock()
    mdp.reset(seed=game_id)
    
    history = []
    belief_sizes = []
    consistent_combinations = set(agent.all_combinations)
    belief_sizes.append(len(consistent_combinations))
    regret_per_attempt = []
    
    while len(history) < mdp.max_attempts:
        # Select action based on current belief
        guess = agent.select_action(history, consistent_combinations)
        
        # Take action
        obs, reward, done, info = mdp.step(guess)
        feedback = info['feedback']
        history.append((guess, feedback))
        
        # Update belief
        consistent_combinations = agent._update_belief(consistent_combinations, guess, feedback)
        belief_sizes.append(len(consistent_combinations))
        
        # Calculate regret as shortfall against optimal value function
        # V*(s) = 1, V^π(s) = 1 if solved, 0 if not solved
        regret_per_attempt.append(1.0 - (1.0 if done and reward == 1.0 else 0.0))
        
        if done and reward == 1.0:
            save_game_log(game_id, history, True, mdp.target_combination, belief_sizes)
            return True, len(history), regret_per_attempt
        
        if len(consistent_combinations) == 0:
            # This shouldn't happen with correct implementation
            print(f"Warning: No consistent combinations left in game {game_id}")
            break
    
    # Failed to find combination
    save_game_log(game_id, history, False, mdp.target_combination, belief_sizes)
    return False, len(history), regret_per_attempt


def main():
    agent = BayesOptimalAgent()
    num_games = 100
    
    # Play games
    results = []
    total_regret = 0.0
    
    print(f"Running Bayes-optimal agent on {num_games} games...")
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
    
    results_file = results_dir / "bayes_optimal_results.json"
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