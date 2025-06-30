import sys
import asyncio
from typing import List, Dict, Tuple
import random
from pathlib import Path
import json
from datetime import datetime
from copy import deepcopy

sys.path.append(str(Path(__file__).parent.parent.parent))
from mdps.combination_lock import CombinationLock
from llm_utils import llm_call

DEBUG = True

def save_game_log(
        game_id: int, 
        history: List[Tuple[str, List[int]]], 
        success: bool, 
        target: str,
        prompt_style: int,
        model: str,
        reasoning_effort: str,
        ):
    """Save game log to a single JSONL file in the logs directory."""
    log_dir = Path(__file__).parent / "logs"
    if DEBUG:
        log_dir = Path(__file__).parent / "logs/debug"
    log_dir.mkdir(exist_ok=True)
    
    # Create log entry
    log_entry = {
        "game_id": game_id,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "prompt_style": prompt_style,
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
    reasoning_effort_str = reasoning_effort if reasoning_effort else "default"
    log_file = log_dir / f"game_results/style{prompt_style}_{reasoning_effort_str}.jsonl"
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def get_messages(prompt_style: int, vocab: str, combination_length: int, max_attempts: int, history, mdp, messages, assistant_message_history, error_handled) -> List[Dict[str, str]]:
    """Get the system prompt based on the prompt style."""
    if prompt_style == 11:

        messages = [{"role": "system", 
                 "content": f"""You are playing a combination lock game. The rules are:
1. Objective - Guess the secret {combination_length}-character combination within {max_attempts} attempts.
2. Valid characters - You can only use these characters: {list(vocab)}
3. Each character in your guess must be unique (no repeats)
4. Color feedback after each guess:
   - Green (🟩) - the character is in the combination and in the correct position.
   - Yellow (🟨) - the character is in the combination but in a different position.
   - Gray (⬜) - the character does not appear in the combination at all.
5. Respond with ONLY your guess as a string of {combination_length} characters, nothing else."""}]
        user_prompt = "Game History:\n"
        for i, (guess, feedback) in enumerate(history):
            feedback_str = ''.join(['⬜' if f == 0 else '🟨' if f == 1 else '🟩' for f in feedback])
            user_prompt += f"Attempt {i+1}: {' '.join(guess)} -> {feedback_str}\n"
        
        if not history:
            user_prompt += f"Make your first guess ({mdp.combination_length} characters, all different):"
        else:
            user_prompt += "Based on the feedback, make your next guess:"
        messages += [{'role': 'user', 'content': user_prompt}]
    elif prompt_style == 12:  # prompt_style == 2
        messages = [{'role': 'system', 
                 'content':f"""You are playing a combination lock game with the goal of optimal exploration. The rules are:
1. Objective - Guess the secret {combination_length}-character combination within {max_attempts} attempts.
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
6. Respond with ONLY your guess as a string of {combination_length} characters, nothing else."""}]
        user_prompt = "Game History:\n"
        for i, (guess, feedback) in enumerate(history):
            feedback_str = ''.join(['⬜' if f == 0 else '🟨' if f == 1 else '🟩' for f in feedback])
            user_prompt += f"Attempt {i+1}: {' '.join(guess)} -> {feedback_str}\n"
        
        if not history:
            user_prompt += f"Make your first guess ({mdp.combination_length} characters, all different):"
        else:
            user_prompt += "Based on the feedback, make your next guess:"
        messages += [{'role': 'user', 'content': user_prompt}]
    elif prompt_style == 3: # multi turn environment interaction where you just document the guess from the model
        if messages is None:
            messages = [{"role": "system", 
                 "content": f"""You are playing a combination lock game with the goal of optimal exploration. The rules are:
1. Objective - Guess the secret {combination_length}-character combination within {max_attempts} attempts.
2. Valid characters - You can only use these characters: {list(vocab)}
3. Each character in your guess must be unique (no repeats)
4. Color feedback after each guess:
   - Green (🟩) - the character is in the combination and in the correct position.
   - Yellow (🟨) - the character is in the combination but in a different position.
   - Gray (⬜) - the character does not appear in the combination at all.
5. Respond with ONLY your guess as a string of {combination_length} characters, nothing else."""}]
            messages += [{'role': 'user', 'content': f"Make your first guess ({mdp.combination_length} characters, all different):"}]
        else:
            messages = deepcopy(messages)
            # add the last feedback and model message 
            guess, feedback = history[-1]
            messages += [{'role':"assistant", 'content': guess}]
            feedback_str = ''.join(['⬜' if f == 0 else '🟨' if f == 1 else '🟩' for f in feedback])
            messages += [{'role':"user", 'content': f"{guess} -> {feedback_str}"}]
    elif prompt_style == 4: # multi turn environment interaction where you document some thinking and the guess from the model
        instruction_suffix = f"Please format your response as: <Think>Any step-by-step, short and concise thinking to determine what the next guess should be</Think><Answer> a {mdp.combination_length} length character code, all different</Answer>. Do not say anything after the <Answer> tags. Do not use markdown. The answer tag should only contain {mdp.combination_length} characters."
        if messages is None:
            messages = [{"role": "system", 
                         "content": f"""You are playing a combination lock game with the goal of optimal exploration. The rules are:
1. Objective - Guess the secret {combination_length}-character combination within {max_attempts} attempts.
2. Valid characters - You can only use these characters: {list(vocab)}
3. Each character in your guess must be unique (no repeats)
4. Color feedback after each guess:
   - Green (🟩) - the character is in the combination and in the correct position.
   - Yellow (🟨) - the character is in the combination but in a different position.
   - Gray (⬜) - the character does not appear in the combination at all."""}]
            messages += [{'role': 'user', 'content': f"Make your first guess. " + instruction_suffix}]
        else:
            messages = deepcopy(messages)
            messages += [{"role": "assistant", "content": assistant_message_history[-1]['content']}]
            guess, feedback = history[-1]
            feedback_str = ''.join(['⬜' if f == 0 else '🟨' if f == 1 else '🟩' for f in feedback])
            if error_handled:
                messages += [{'role':"user", 'content': f"Your previous guess was unable to be parsed. Here is the feedback for a random other guess. {guess} -> {feedback_str}. Now make your next guess. " + instruction_suffix}]
            else:
                messages += [{'role':"user", 'content': f"{guess} -> {feedback_str}. Now make your next guess. " + instruction_suffix}]
    elif prompt_style == 5: # multi turn environment interaction where you document the belief state and the guess from the model.
        instruction_suffix = f"""Now update your beliefs based on the feedback, and use your new beliefs to make your next guess about the secret code. Knowledge in your beliefs must only be updated but can never be discarded, forgotten, or removed. Do not say anything about which information is new and updated or old and remains the same. 
Please format your response as: <Beliefs>Your beliefs on what the answer can be given what you know so far</Beliefs><Action> a {mdp.combination_length} length character code, all different</Action>. Do not say anything after the <Action> tags. Do not use markdown. The action tag should only contain {mdp.combination_length} characters."""
        if messages is None:
            messages = [{"role": "system", 
                         "content": f"""You are playing a combination lock game with the goal of optimal exploration. The rules are:
1. Objective - Guess the secret {combination_length}-character combination within {max_attempts} attempts.
2. Valid characters - You can only use these characters: {list(vocab)}
3. Each character in your guess must be unique (no repeats)
4. Color feedback after each guess:
   - Green (🟩) - the character is in the combination and in the correct position.
   - Yellow (🟨) - the character is in the combination but in a different position.
   - Gray (⬜) - the character does not appear in the combination at all."""}]
            messages += [{'role': 'user', 'content': f"Make your first guess. Please format your response as: <Beliefs>Your beliefs on what the answer can be given what you know so far</Beliefs><Action> a {mdp.combination_length} length character code, all different</Action>. Do not say anything after the <Action> tags. Do not use markdown. The action tag should only contain {mdp.combination_length} characters."}]
        else:
            messages = deepcopy(messages)
            messages += [{"role": "assistant", "content": assistant_message_history[-1]['content']}]
            guess, feedback = history[-1]
            feedback_str = ''.join(['⬜' if f == 0 else '🟨' if f == 1 else '🟩' for f in feedback])
            if error_handled:
                messages += [{'role':"user", 'content': f"Your previous guess was unable to be parsed. Here is the feedback for a random other guess. {guess} -> {feedback_str}. " + instruction_suffix}]
            else:
                messages += [{'role':"user", 'content': f"{guess} -> {feedback_str}. " + instruction_suffix}]
    elif prompt_style == 6: # multi turn environment interaction where you document the belief state and the guess from the model and you prompt the assistant in seperate steps to get these reasoning tokens.
                            # we will want to aggregate the token counts so that the ploting utility code reflects the total token price. which includes the prompt and the 
        belief_instruction_suffix = """Now update your beliefs about the secret code with the latest feedback. Knowledge in your beliefs must only be updated but can never be discarded, forgotten, or removed. Do not say anything about which information is new and updated or old and remains the same. 
Please format your response as: <Beliefs>Your new beliefs</Beliefs>"""
        if messages is None:
            messages = [{"role": "system", 
                         "content": f"""You are playing a combination lock game with the goal of optimal exploration. The rules are:
1. Objective - Guess the secret {combination_length}-character combination within {max_attempts} attempts.
2. Valid characters - You can only use these characters: {list(vocab)}
3. Each character in your guess must be unique (no repeats)
4. Color feedback after each guess:
   - Green (🟩) - the character is in the combination and in the correct position.
   - Yellow (🟨) - the character is in the combination but in a different position.
   - Gray (⬜) - the character does not appear in the combination at all."""}]
            messages += [{'role': 'user', 'content': f"Construct a belief state from which you will be able to make a first guess. But do not make the guess yet. Please format your response as: <Beliefs>Your beliefs on what the answer can be given what you know so far</Beliefs."}]
        else:
            messages = deepcopy(messages)
            messages += [{"role": "assistant", "content": assistant_message_history[-1]['content']}]
            guess, feedback = history[-1]
            feedback_str = ''.join(['⬜' if f == 0 else '🟨' if f == 1 else '🟩' for f in feedback])
            if error_handled:
                messages += [{'role':"user", 'content': f"Your previous guess was unable to be parsed. Here is the feedback for a random other guess. {guess} -> {feedback_str}. " + belief_instruction_suffix}]
            else:
                messages += [{'role':"user", 'content': f"{guess} -> {feedback_str}. " + belief_instruction_suffix}]
    else:
        raise Exception(f"invalid {prompt_style = }")
    return messages

async def update_belief(style, messages, model, belief_model_call_store: dict, mdp, ref):
    if style == 6:
        messages = deepcopy(messages)
        data: Dict = await llm_call( # type: ignore
            model=model,
            messages=messages,
            # system=system_prompt,
            # user=user_prompt,
            temperature=0.1,
            get_everything=True,
            reasoning_effort=ref,
        )
        print("^", end='', flush=True)
        belief_model_call_store.update(data)
        messages += [{'role': 'assistant', "content": data["choices"][0]["message"]["content"]}]
        messages += [{'role':"user", 'content': f"Now make your next guess about the secret code based on your current beliefs. Please format your response as: <Action> a {mdp.combination_length} length character code, all different</Action>. Do not say anything after the <Action> tags. Do not use markdown. The action tag should only contain {mdp.combination_length} characters."}]

    return messages

def process_guess_msg(msg_str, vocab, combination_length):
    remove_list = ["**", "</Answer>", "</answer>", "<Answer>", "<answer>", "</Ans>", "</ans>", "<Ans>", "<ans>","<Action>","</Action>","<action>","</action>"]
    def rem_list_from_str(s: str):
        if s.endswith("**"):
            s = s[:-2]
        for rm_str in remove_list:
            s = s.replace(rm_str, "")
        return s
    guess = ''.join(c for c in rem_list_from_str(msg_str) if c in vocab)[-combination_length:].lower()
    return guess

async def play_single_game(
        game_id: int, 
        prompt_style: int,
        model: str,
        max_attempts: int,
        vocab: str,
        reasoning_effort: str = None,  # 'low', 'medium', 'high'
        ) -> Tuple[bool, int, List[float]]:
    """
    Play a single game of combination lock with the LLM.
    Returns (success, num_attempts, regret_per_attempt)
    """
    mdp = CombinationLock(max_attempts=max_attempts, vocab=vocab)
    mdp.reset(seed=game_id)  # Use game_id as seed for reproducibility
    
    # Get system prompt based on style
    
    # Track game history
    history = []
    regret_per_attempt = []
    messages = None
    attempt = 0
    # llm_response = None
    assistant_message_history = []
    error_handled = False

    while True:
        
        messages = get_messages(prompt_style, mdp.vocab, mdp.combination_length, max_attempts, history, mdp, messages, assistant_message_history, error_handled)
        belief_model_call_store = dict()
        messages = await update_belief(prompt_style, messages, model, belief_model_call_store, mdp, reasoning_effort)
        # if I want to call the model multiple times for one enviornment interaction, then I need to record this somehow within one json object. 
        # This will need to be slightly different than the typical json object, but I hope I can reuse some of the plotting code...
        attempt += 1
        # Get LLM's guess
        data: Dict = await llm_call( # type: ignore
            model=model,
            messages=messages,
            # system=system_prompt,
            # user=user_prompt,
            temperature=0.1,
            get_everything=True,
            reasoning_effort=reasoning_effort,
        )
        data['game_id'] = game_id
        data['prompt_style'] = prompt_style
        data['messages'] = messages
        data['model'] = model
        data['vocab'] = mdp.vocab
        data['combination_length'] = mdp.combination_length
        data['max_attempts'] = max_attempts
        data['reasoning_effort'] = reasoning_effort
        data['timestamp'] = datetime.now().isoformat()
        data['attempt'] = attempt
        data['target_combination'] = mdp.target_combination
        if len(belief_model_call_store) != 0:
            data['other_lm_calls'] = [belief_model_call_store]
        error_handled = False

        llm_response = data["choices"][0]["message"]["content"]
        assistant_message_history += [data["choices"][0]["message"]]
        guess = process_guess_msg(llm_response, mdp.vocab, mdp.combination_length) # just ensure it is 3 characters. also replace ** because sometimes it responds in markdown.
        if len(guess) != mdp.combination_length or len(set(guess)) != mdp.combination_length:
            # If LLM gives invalid response, make a random valid guess
            print(f"({guess})")
            chars = list(mdp.vocab)
            random.shuffle(chars)
            guess = ''.join(chars[:mdp.combination_length])
            error_handled = True
        data['error_handled'] = error_handled

        log_dir = Path(__file__).parent / "logs/llm_calls"
        if DEBUG:
            log_dir = Path(__file__).parent / "logs/debug/llm_calls"
            
        log_dir.mkdir(exist_ok=True)
        reasoning_effort_str = reasoning_effort if reasoning_effort else "default"
        log_file = log_dir / f"style{prompt_style}_{model.split('/')[-1]}_{reasoning_effort_str}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')

        print(f'.', end='', flush=True)  # Print a dot for each episode to indicate progress
        # Clean up response to get just the guess
        
        
        # Make the guess
        obs, reward, done, info = mdp.step(guess)
        feedback = info['feedback']
        history.append((guess, feedback))
        
        # Calculate regret as shortfall against optimal value function
        # V*(s) = 1, V^π(s) = 1 if solved, 0 if not solved
        regret_per_attempt.append(1.0 - (1.0 if done and reward == 1.0 else 0.0))
        
        if done:
            # Save game log before returning
            save_game_log(game_id, history, reward == 1.0, mdp.target_combination, prompt_style, model, reasoning_effort=reasoning_effort)
            return reward == 1.0, len(history), regret_per_attempt

async def main(
        prompt_style: int = 1, 
        model: str = 'google/gemini-2.5-pro-preview',
        num_games = 10,
        reasoning_effort: str = None,  # 'low', 'medium', 'high'
        ):
    
    log_dir = Path(__file__).parent / "logs"
    if DEBUG:
        log_dir = Path(__file__).parent / "logs/debug"

    if not reasoning_effort:
        for file in log_dir.rglob(f"style{prompt_style}_*.jsonl"):
            if file.is_file():
                file.unlink()
        print(f"Deleted existing log files for style {prompt_style} in {log_dir}")
        print(f"Starting {num_games} games with prompt style {prompt_style} using model {model}")

    tasks = [play_single_game(
            game_id=i, 
            prompt_style=prompt_style,
            model=model,
            max_attempts=12,
            vocab='!@#$%^&*pqrs5678',
            reasoning_effort=reasoning_effort,
            ) for i in range(num_games)]
    results = await asyncio.gather(*tasks)

async def run_all_games(
        prompt_styles: List[int], 
        models: List[str], 
        num_games: int,
        reasoning_efforts: bool,
        ):
    if not reasoning_efforts:
        for prompt_style in prompt_styles:
            tasks = [main(prompt_style, model=model, num_games=num_games, reasoning_effort=None) for model in models]
            await asyncio.gather(*tasks)
    else:
        for reasoning_effort in ["low", "medium", "high"]:
            for prompt_style in prompt_styles:
                tasks = [main(
                        prompt_style, 
                        model=model, 
                        num_games=num_games,
                        reasoning_effort=reasoning_effort,
                        ) for model in models]
                await asyncio.gather(*tasks)
            print(f"Completed all games with reasoning effort: {reasoning_effort}")

if __name__ == "__main__":
    models = [
        # 'google/gemini-2.5-pro-preview',
        # 'openai/o3',
        # 'anthropic/claude-3.5-sonnet',
        # 'anthropic/claude-opus-4',
        'deepseek/deepseek-r1-0528',
    ]
    reasoning_efforts: bool = False
    prompt_styles = [3]
    num_games = 1
    
    asyncio.run(run_all_games(
        prompt_styles=prompt_styles, 
        models=models, 
        num_games=num_games,
        reasoning_efforts=reasoning_efforts,
    ))
    print("\nAll games completed.")