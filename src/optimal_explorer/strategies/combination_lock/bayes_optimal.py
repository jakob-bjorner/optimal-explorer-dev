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
    
    def _update_belief(self, guess: str, feedback: List[int], posterior: List[List[str]]) -> None:
        """Update belief based on new feedback."""
        guess_chars = list(guess)
        
        # Update untested and found characters
        for i, char in enumerate(guess_chars):
            self.untested_chars.discard(char)

            if feedback[i] > 0:  # Character is in the combination
                self.found_chars.add(char)
            
            # Update known positions
            if feedback[i] == 2:  # Correct position
                self.known_positions[i] = char

        for pos,pos_chars in enumerate(posterior):
            if len(pos_chars) == 1:
                self.known_positions[pos] = pos_chars[0]
                self.found_chars.add(pos_chars[0])
                self.untested_chars.discard(pos_chars[0])
    
    def select_action(self, history: List[Tuple[str, List[int]]], posterior: List[List[str]]) -> str:
        """Select the best action given history using a more intelligent heuristic."""
        tried_guesses = {h[0] for h in history}

        # 1. Certainty: If we know the full combination, guess it.
        if None not in self.known_positions:
            return "".join(self.known_positions)

        all_posterior_chars = set([char for position_chars in posterior for char in position_chars])
         # 2. Refinement Phase: We know all characters, just need their positions.
        guess = self.known_positions
        if len(all_posterior_chars) == self.combination_length:
            working_posterior = copy.deepcopy(posterior)
            for pos, pos_chars in enumerate(working_posterior):
                if guess[pos] is not None:
                    continue
                next_char_choices = set(pos_chars)-set(guess)
                if len(next_char_choices) == 0:
                    import pdb; pdb.set_trace()
                guess[pos] = random.choice(list(next_char_choices))
                new_guesses = True
                while new_guesses and None in guess:
                    new_guesses = False
                    for posterior_i in range(pos+1, self.combination_length): # remove the guessed chars from posteriors in other positions
                        for guess_i in range(pos, self.combination_length):
                            if guess_i != posterior_i:
                                working_posterior[posterior_i] = [c for c in working_posterior[posterior_i] if c != guess[guess_i]]
                        if len(working_posterior[posterior_i]) == 1: # if this forces a later position to only have one choice, then we can fill it in
                            guess[posterior_i] = working_posterior[posterior_i][0]
                            new_guesses = True
                
            assert len(guess) == self.combination_length and "".join(guess) not in tried_guesses
            return "".join(guess)
            
        # 3. Exploration Phase: We still need to find new characters. Do not guess known symbols in their correct positions, that produces no new information!
        untested_chars = list(self.untested_chars)
        random.shuffle(untested_chars)

        # if the set of untested symbols >= 3, uniformly sample a guess from the set of untested symbols.
        if len(untested_chars) >= self.combination_length:
            return "".join(untested_chars[:self.combination_length])
        
        known_chars_not_in_place = [c for c in self.found_chars if c not in self.known_positions]
        # elif we have tested symbols that we know are in the combination but don't know their position - fill remaining slots by trying them in a new position
        if len(known_chars_not_in_place) > 0:
            if len(untested_chars) != 2:
                import pdb; pdb.set_trace()
            assert self.combination_length == 3 and len(untested_chars) == 2 # this strategy is only optimal for length 3 combinations
            extra_char_to_use = random.choice(known_chars_not_in_place)
            possible_positions_for_extra_char = [i for i, possible_chars in enumerate(posterior) if extra_char_to_use in possible_chars]
            position_for_extra_char = random.choice(possible_positions_for_extra_char)
            guess = "".join(untested_chars)
            guess = guess[:position_for_extra_char] + extra_char_to_use + guess[position_for_extra_char:]
            return guess

        # elif no such symbols exist, ie we know the position of all tested symbols, then we can't gain any info from the remaining slots so use arbitrary symbols. 
        guess = "".join(untested_chars) + list(set(self.vocab)-self.untested_chars)[:self.combination_length - len(untested_chars)]
        return guess

        # OLD:
        # Gather characters to fill the empty slots. Prioritize untested characters.
        # Use existing found_chars (that aren't locked in place) if needed to form a valid guess.
        chars_for_filling = list(self.untested_chars)
        random.shuffle(chars_for_filling)
        
        known_chars_in_place = {c for c in template if c is not None}
        other_usable_chars = list(self.found_chars - known_chars_in_place)
        random.shuffle(other_usable_chars)
        
        # Combine lists to ensure we have enough unique characters for a valid guess
        potential_fillers = chars_for_filling + other_usable_chars
        
        fill_idx = 0
        for i in range(self.combination_length):
            if template[i] is None:
                # Find the next unique character to insert
                while fill_idx < len(potential_fillers) and potential_fillers[fill_idx] in template:
                    fill_idx += 1
                if fill_idx < len(potential_fillers):
                    template[i] = potential_fillers[fill_idx]
                    fill_idx += 1

        # If we successfully built a full, valid guess, return it
        if None not in template and len(set(template)) == self.combination_length:
            guess = "".join(template)
            if guess not in tried_guesses:
                return guess
        
        # 4. Fallback: If above strategies fail, find any valid, untried combination.
        # This is a safety net for edge cases.
        all_possible_guesses = self.all_combinations
        random.shuffle(all_possible_guesses) # Randomize to avoid getting stuck
        for g in all_possible_guesses:
            if g not in tried_guesses:
                return g

        # Ultimate fallback: Should realistically never be reached if max_attempts is reasonable.
        return random.choice(self.all_combinations)


def save_game_log(game_id: int, history: List[Tuple[str, List[int]]], success: bool, 
                  target: str, belief_sizes: List[int], model: str):
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
    log_file = log_dir / f"{model}.jsonl"
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


def play_single_game(agent: BayesOptimalAgent, game_id: int, model: str) -> Tuple[bool, int, List[float]]:
    """Play a single game with Bayes-optimal agent. Returns (success, num_attempts, regret_per_attempt)."""
    mdp = CombinationLock(vocab=agent.vocab, max_attempts=agent.max_attempts, combination_length=agent.combination_length)
    mdp.reset(seed=game_id)
    history = []
    belief_sizes = []
    regret_per_attempt = []
    while len(history) < mdp.max_attempts:
        guess = agent.select_action(history, mdp.posterior)
        obs, reward, done, info = mdp.step(guess)
        feedback = info.get('feedback')
        if feedback is None:
            print(f"Invalid guess encountered: {guess}. Info: {info}. Replacing with a random valid guess.")
            # Generate a random valid guess
            valid_guesses = [g for g in agent.all_combinations if g not in [h[0] for h in history]]
            if valid_guesses:
                guess = random.choice(valid_guesses)
            else:
                guess = agent.select_action([], mdp.posterior)  # fallback to agent's default
            obs, reward, done, info = mdp.step(guess)
            feedback = info.get('feedback')
            if feedback is None:
                print(f"Still invalid after random guess: {guess}. Info: {info}. Ending game.")
                break
        history.append((guess, feedback))
        agent._update_belief(guess, feedback, mdp.posterior)
        belief_sizes.append(len(agent.untested_chars) + len(agent.found_chars))
        regret_per_attempt.append(1.0 - (1.0 if done and reward == 1.0 else 0.0))
        if done and reward == 1.0:
            save_game_log(game_id, history, True, mdp.target_combination, belief_sizes, model)
            return True, len(history), regret_per_attempt
    save_game_log(game_id, history, False, mdp.target_combination, belief_sizes, model)
    return False, len(history), regret_per_attempt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combination-length", type=int, default=3, help="Length of the combination")
    parser.add_argument("--vocab", type=str, default="!@#$%^&*pqrs5678", help="Vocabulary to use")
    parser.add_argument("--max-attempts", type=int, default=12, help="Maximum number of attempts")
    parser.add_argument("--num-games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--model", type=str, default="bayes_optimal_2", help="Model name (for logging)")
    args = parser.parse_args()

    agent = BayesOptimalAgent(combination_length=args.combination_length, max_attempts=args.max_attempts, vocab=args.vocab)
    num_games = args.num_games
    model = args.model

    results = []
    total_regret = 0.0
    print(f"\nRunning Bayes-optimal agent on {num_games} games...")
    print(f"Combination length: {args.combination_length}, Vocabulary: {args.vocab}, Max attempts: {args.max_attempts}")
    for game_id in range(num_games):
        # Reset agent state for each game
        agent = BayesOptimalAgent(combination_length=args.combination_length, max_attempts=args.max_attempts, vocab=args.vocab)
        success, attempts, regret_per_attempt = play_single_game(agent, game_id, model)
        results.append((success, attempts, regret_per_attempt))
        total_regret += sum(regret_per_attempt)
        if (game_id + 1) % 10 == 0:
            print(f"Completed {game_id + 1} games...")
    wins = sum(1 for success, _, _ in results if success)
    total_attempts = sum(attempts for _, attempts, _ in results)
    avg_attempts = total_attempts / len(results)
    print(f"\n{'='*50}")
    print(f"BAYES-OPTIMAL AGENT RESULTS ({num_games} games)")
    print(f"Combination length: {args.combination_length}, Vocabulary: {args.vocab}")
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