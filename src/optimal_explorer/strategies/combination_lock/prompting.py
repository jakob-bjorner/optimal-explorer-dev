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

def get_system_prompt(prompt_style: int, vocab: str) -> str:
    """Get the system prompt based on the prompt style."""
    if prompt_style == 1:
        return f"""You are playing a combination lock game. The rules are:
1. Objective - Guess the secret {len(vocab)}-character combination within 8 attempts.
2. Valid characters - You can only use these characters: {list(vocab)}
3. Each character in your guess must be unique (no repeats)
4. Color feedback after each guess:
   - Green (🟩) - the character is in the combination and in the correct position.
   - Yellow (🟨) - the character is in the combination but in a different position.
   - Gray (⬜) - the character does not appear in the combination at all.
5. Respond with ONLY your guess as a string of {len(vocab)} characters, nothing else."""
    else:  # prompt_style == 2
        return f"""You are playing a combination lock game with the goal of optimal exploration. The rules are:
1. Objective - Guess the secret {len(vocab)}-character combination within 8 attempts.
2. Valid characters - You can only use these characters: {list(vocab)}
3. Each character in your guess must be unique (no repeats)
4. Color feedback after each guess:
   - Green (🟩) - the character is in the combination and in the correct position.
   - Yellow (🟨) - the character is in the combination but in a different position.
   - Gray (⬜) - the character does not appear in the combination at all.
5. Important strategy guidelines:
   - Use your first few guesses to explore different characters and positions
   - Pay attention to character frequency and position patterns
   - Use the feedback to systematically narrow down possibilities
6. Respond with ONLY your guess as a string of {len(vocab)} characters, nothing else."""

async def play_single_game(
        game_id: int, 
        prompt_style: int,
        model: str,
        max_attempts: int,
        vocab: str,
        ) -> Tuple[bool, int, List[float]]:
    """
    Play a single game of combination lock with the LLM.
    Returns (success, num_attempts, regret_per_attempt)
    """
    mdp = CombinationLock(max_attempts=max_attempts, vocab=vocab)
    mdp.reset(seed=game_id)  # Use game_id as seed for reproducibility
    
    # Get system prompt based on style
    system_prompt = get_system_prompt(prompt_style, mdp.vocab)
    
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
            user_prompt += f"Make your first guess ({mdp.combination_length} characters, all different):"
        else:
            user_prompt += "Based on the feedback, make your next guess:"
        
        # Get LLM's guess
        data = await llm_call(
            model=model,
            system=system_prompt,
            user=user_prompt,
            temperature=0.1,
            get_everything=True,
        )
        data['game_id'] = game_id
        log_dir = Path(__file__).parent / "logs/llm_calls"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"api_logs_style{prompt_style}_{model.split('/')[-1]}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
        llm_response = data["choices"][0]["message"]["content"]

        print(f'.', end='', flush=True)  # Print a dot for each game to indicate progress
        # Clean up response to get just the guess
        guess = ''.join(c for c in llm_response if c in mdp.vocab)
        if len(guess) != mdp.combination_length or len(set(guess)) != mdp.combination_length:
            # If LLM gives invalid response, make a random valid guess
            chars = list(mdp.vocab)
            random.shuffle(chars)
            guess = ''.join(chars[:mdp.combination_length])
        
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

async def main(
        prompt_style: int = 1, 
        model: str = 'google/gemini-2.5-pro-preview',
        num_games = 10,
        ):
    tasks = [play_single_game(
            game_id=i, 
            prompt_style=prompt_style,
            model=model,
            max_attempts=12,
            vocab='!@#$%^&*pqrs5678',
            ) for i in range(num_games)]
    results = await asyncio.gather(*tasks)
    
    # Calculate statistics
    wins = sum(1 for success, _, _ in results if success)
    total_attempts = sum(attempts for _, attempts, _ in results)
    avg_attempts = total_attempts / len(results)
    
    print(f"\nResults after {num_games} games (Prompt Style {prompt_style}):")
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
        "prompt_style": prompt_style,
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
    model_name = model.split('/')[-1]
    results_file = results_dir / f"prompting_results_style{prompt_style}_{model_name}.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    models = [
        'google/gemini-2.5-pro-preview',
        'openai/o3',
        'anthropic/claude-3.5-sonnet',
        'anthropic/claude-opus-4',
        'deepseek/deepseek-r1-0528',
    ]
    parser.add_argument("--prompt-style", type=int, choices=[1, 2], default=1,
                      help="Prompt style: 1 for basic rules, 2 for optimal exploration strategy")
    parser.add_argument("--num-games", type=int, default=10,
                      help="Number of games to play")
    args = parser.parse_args()
    
    async def run_all_models():
        tasks = [main(args.prompt_style, model=model, num_games=args.num_games) for model in models]
        await asyncio.gather(*tasks)
    
    asyncio.run(run_all_models())