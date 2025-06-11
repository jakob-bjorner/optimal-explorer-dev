import sys
import asyncio
from typing import List, Dict, Tuple
import random
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path to import from mdps
sys.path.append(str(Path(__file__).parent.parent.parent))
from mdps.combination_lock import CombinationLock
from llm_utils import llm_call

def save_game_log(game_id: int, history: List[Tuple[str, List[int]]], success: bool, target: str):
    """Save game log to a single JSONL file in the logs directory."""
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create log entry
    log_entry = {
        "game_id": game_id,
        "timestamp": datetime.now().isoformat(),
        "target_combination": target,
        "success": success,
        "num_attempts": len(history),
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
    
    # Append to JSONL file
    log_file = log_dir / "prompting.jsonl"
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

async def play_single_game(game_id: int) -> Tuple[bool, int, List[float]]:
    """
    Play a single game of combination lock with the LLM.
    Returns (success, num_attempts, regret_per_attempt)
    """
    mdp = CombinationLock()
    mdp.reset(seed=game_id)  # Use game_id as seed for reproducibility
    
    # Initial system prompt explaining the game
    system_prompt = """You are playing a 3-digit combination lock game. The rules are:
1. You need to guess a 3-digit combination where all digits are different
2. You have 8 attempts to find the correct combination
3. After each guess, you'll get feedback:
   - ⬜ means the digit is not in the combination
   - 🟨 means the digit is in the combination but in the wrong position
   - 🟩 means the digit is in the correct position
4. Make your next guess based on the feedback
5. Respond with ONLY your 3-digit guess, nothing else"""

    # Track game history
    history = []
    regret_per_attempt = []
    
    while True:
        # Create user prompt with game history
        user_prompt = "Game History:\n"
        for i, (guess, feedback) in enumerate(history):
            feedback_str = ''.join(['⬜' if f == 0 else '🟨' if f == 1 else '🟩' for f in feedback])
            user_prompt += f"Attempt {i+1}: {guess} -> {feedback_str}\n"
        
        if not history:
            user_prompt += "Make your first guess (3 digits, all different):"
        else:
            user_prompt += "Based on the feedback, make your next guess:"
        
        # Get LLM's guess
        llm_response = await llm_call(
            system=system_prompt,
            user=user_prompt,
            temperature=0.1  # Low temperature for more deterministic responses
        )
        
        # Clean up response to get just the guess
        guess = ''.join(c for c in llm_response if c.isdigit())
        if len(guess) != 3 or len(set(guess)) != 3:
            # If LLM gives invalid response, make a random valid guess
            digits = list(range(10))
            random.shuffle(digits)
            guess = ''.join(map(str, digits[:3]))
        
        # Make the guess
        obs, reward, done, info = mdp.step(guess)
        feedback = info['feedback']
        history.append((guess, feedback))
        
        # Calculate regret as shortfall against optimal value function
        # V*(s) = 1, V^π(s) = 1 if solved, 0 if not solved
        regret_per_attempt.append(1.0 - (1.0 if done and reward == 1.0 else 0.0))
        
        if done:
            # Save game log before returning
            save_game_log(game_id, history, reward == 1.0, mdp.target_combination)
            return reward == 1.0, len(history), regret_per_attempt

async def main():
    # Play 100 games asynchronously
    tasks = [play_single_game(i) for i in range(100)]
    results = await asyncio.gather(*tasks)
    
    # Calculate statistics
    wins = sum(1 for success, _, _ in results if success)
    total_attempts = sum(attempts for _, attempts, _ in results)
    avg_attempts = total_attempts / len(results)
    
    print(f"\nResults after 100 games:")
    print(f"Win rate: {wins}%")
    print(f"Average attempts per game: {avg_attempts:.2f}")
    
    # Print distribution of attempts
    attempt_dist = {}
    for _, attempts, _ in results:
        attempt_dist[attempts] = attempt_dist.get(attempts, 0) + 1
    
    print("\nAttempt distribution:")
    for attempts in sorted(attempt_dist.keys()):
        print(f"{attempts} attempts: {attempt_dist[attempts]} games")
    
    # Calculate cumulative regret by attempt number
    regret_by_attempts = {}
    total_regret = 0.0
    for _, _, regret_per_attempt in results:
        for attempt_num, regret in enumerate(regret_per_attempt, 1):
            regret_by_attempts[attempt_num] = regret_by_attempts.get(attempt_num, 0) + regret
        total_regret += sum(regret_per_attempt)
    
    print("\nAverage cumulative regret by attempt number:")
    cumulative_regret = 0
    for attempt_num in sorted(regret_by_attempts.keys()):
        avg_regret = regret_by_attempts[attempt_num] / len(results)
        cumulative_regret += avg_regret
        print(f"After {attempt_num} attempts: {cumulative_regret:.3f} regret")
    
    # Store results in JSON file
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "num_games": len(results),
        "win_rate": wins,
        "avg_attempts": avg_attempts,
        "total_regret": total_regret,
        "avg_regret_per_game": total_regret / len(results),
        "attempt_distribution": attempt_dist,
        "cumulative_regret": {
            str(attempt_num): cumulative_regret
            for attempt_num, cumulative_regret in enumerate(
                [sum(regret_by_attempts.get(i, 0) / len(results) for i in range(1, j + 1))
                 for j in range(1, max(regret_by_attempts.keys()) + 1)],
                1
            )
        }
    }
    
    results_file = results_dir / "prompting_results.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main()) 