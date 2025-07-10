import sys
import asyncio
from typing import List, Dict, Tuple, Any
import random
from pathlib import Path
import json
from datetime import datetime
from copy import deepcopy

sys.path.append(str(Path(__file__).parent.parent.parent))
from mdps.bandits import MultiArmedBandits
from llm_utils import llm_call

DEBUG = True
run_name = '_dontreconstruct'

def save_game_log(
        game_id: int,
        history: List[Tuple[str, float]],
        total_reward: float,
        regret_per_attempt: List[float],
        prompt_style: int,
        model: str,
        env_config: dict,
        reasoning_effort: str = None,
    ):
    """Save game log to a single JSONL file in the logs directory."""
    log_dir = Path(__file__).parent / "logs"
    if DEBUG:
        log_dir = Path(__file__).parent / "logs/debug"
    log_dir.mkdir(exist_ok=True)
    log_entry = {
        "game_id": game_id,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "prompt_style": prompt_style,
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
    log_file = log_dir / f"game_results/style{prompt_style}_{model.split('/')[-1]}_{reasoning_effort if reasoning_effort else 'default'}{run_name}.jsonl"
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

def process_action_msg(msg_str, arm_names):
    # Extract the arm name from the LLM response robustly
    msg_str = msg_str.strip().lower()
    for arm in arm_names:
        if arm.lower() in msg_str:
            return arm
    # fallback: first word
    tokens = msg_str.split()
    for token in tokens:
        if token in arm_names:
            return token
    # fallback: random
    return random.choice(arm_names)

async def get_messages(
        prompt_style: int,
        arm_names: List[str],
        max_steps: int,
        history: List[Tuple[str, float]],
        messages: List[Dict[str, str]],
        assistant_message_history: List[Dict[str, str]],
        error_handled: bool,
        reasoning_history: List[str],
        llm_call_params: dict,
        env_config: dict,
    ) -> List[Dict[str, str]]:
    BASIC_SYSTEM_PROMPT = f"""You are playing a multi-armed bandit game. The rules are:\n1. There are {len(arm_names)} arms: {', '.join(arm_names)}.\n2. Each arm gives a reward drawn from a fixed but unknown distribution.\n3. You have {max_steps} steps. On each step, pick one arm to pull and observe the reward.\n4. Your goal is to maximize the total reward over all steps."""
    if prompt_style == 1:
        messages = [
            {"role": "system", "content": BASIC_SYSTEM_PROMPT + "\nRespond with ONLY the name of the arm you want to pull next, nothing else."}
        ]
        user_prompt = "Game History:\n"
        for i, (action, reward) in enumerate(history):
            user_prompt += f"Step {i+1}: {action} -> reward {reward:.3f}\n"
        if not history:
            user_prompt += f"Pick your first arm ({', '.join(arm_names)}):"
        else:
            user_prompt += "Based on the history, pick the next arm to pull:"
        messages += [{"role": "user", "content": user_prompt}]
    elif prompt_style == 2:
        messages = [
            {"role": "system", "content": BASIC_SYSTEM_PROMPT + "\n5. Important: Use clever exploration strategies to maximize your cumulative reward. Try to balance exploring new arms and exploiting good ones.\nRespond with ONLY the name of the arm you want to pull next, nothing else."}
        ]
        user_prompt = "Game History:\n"
        for i, (action, reward) in enumerate(history):
            user_prompt += f"Step {i+1}: {action} -> reward {reward:.3f}\n"
        if not history:
            user_prompt += f"Pick your first arm ({', '.join(arm_names)}):"
        else:
            user_prompt += "Based on the history, pick the next arm to pull:"
        messages += [{"role": "user", "content": user_prompt}]
    else:
        raise Exception(f"invalid prompt_style = {prompt_style}")
    return messages

async def play_single_game(
        game_id: int,
        prompt_style: int,
        model: str,
        max_steps: int,
        arm_names: List[str],
        env_config: dict,
        reasoning_effort: str = None,
        output_info: str = None,
        url: str = None,
    ):
    mdp = MultiArmedBandits(
        arm_names=arm_names,
        num_arms=len(arm_names),
        max_steps=max_steps,
        reward_type=env_config.get('reward_type', 'gaussian'),
        noise_level=env_config.get('noise_level', 'medium'),
        seed=game_id
    )
    mdp.reset(seed=game_id)
    history = []
    regret_per_attempt = []
    messages = None
    assistant_message_history = []
    reasoning_history = []
    error_handled = False
    temperature = 0.1 if "qwen3" not in model else 0.7
    llm_call_params = {
        "model": model,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
        "get_everything": True,
        "url": url
    }
    total_reward = 0.0
    for step in range(max_steps):
        messages = await get_messages(
            prompt_style,
            arm_names,
            max_steps,
            history,
            messages,
            assistant_message_history,
            error_handled,
            reasoning_history,
            llm_call_params,
            env_config
        )
        data: Dict = await llm_call(
            **llm_call_params,
            messages=messages,
        )
        data['game_id'] = game_id
        data['prompt_style'] = prompt_style
        data['messages'] = messages
        data['model'] = model
        data['arm_names'] = arm_names
        data['max_steps'] = max_steps
        data['reasoning_effort'] = reasoning_effort
        data['timestamp'] = datetime.now().isoformat()
        data['attempt'] = step + 1
        if 'choices' in data and data['choices']:
            llm_response = data['choices'][0]['message']['content'] if 'message' in data['choices'][0] else data['choices'][0]['text']
        else:
            llm_response = ''
        action = process_action_msg(llm_response, arm_names)
        error_handled = False
        data['error_handled'] = error_handled
        log_dir = Path(__file__).parent / "logs/llm_calls"
        if DEBUG:
            log_dir = Path(__file__).parent / "logs/debug/llm_calls"
        log_dir.mkdir(exist_ok=True)
        reasoning_effort_str = reasoning_effort if reasoning_effort else "default"
        log_file = log_dir / f"style{prompt_style}_{model.split('/')[-1]}_{reasoning_effort_str}{run_name}.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
        obs, reward, done, info = mdp.step(action)
        history.append((action, reward))
        total_reward += reward
        # Regret: difference between optimal mean and chosen arm mean
        action_index = mdp.arm_names.index(action)
        regret = float(max(mdp.arm_means) - mdp.arm_means[action_index])
        regret_per_attempt.append(regret)
        print('.', end='', flush=True)
        if done:
            break
    save_game_log(game_id, history, total_reward, regret_per_attempt, prompt_style, model, env_config, reasoning_effort=reasoning_effort)
    if output_info == "history":
        return total_reward, len(history), regret_per_attempt, messages, history
    return total_reward, len(history), regret_per_attempt

async def main(
        prompt_style: int = 1,
        model: str = 'google/gemini-2.5-pro-preview',
        num_games: int = 10,
        env_config: dict = None,
        reasoning_effort: str = None,
    ):
    log_dir = Path(__file__).parent / "logs"
    if DEBUG:
        log_dir = Path(__file__).parent / "logs/debug"
    if not reasoning_effort:
        for file in log_dir.rglob(f"style{prompt_style}_*{run_name}.jsonl"):
            if file.is_file():
                file.unlink()
        print(f"Deleted existing log files for style {prompt_style} in {log_dir}")
        print(f"Starting {num_games} games with prompt style {prompt_style} using model {model}")
    arm_names = env_config.get('arm_names', [str(i) for i in range(env_config.get('num_arms', 5))])
    max_steps = env_config.get('max_steps', 50)
    tasks = [play_single_game(
            game_id=i,
            prompt_style=prompt_style,
            model=model,
            max_steps=max_steps,
            arm_names=arm_names,
            env_config=env_config,
            reasoning_effort=reasoning_effort,
        ) for i in range(num_games)]
    results = await asyncio.gather(*tasks)

async def run_all_games(
        prompt_styles: List[int],
        models: List[str],
        num_games: int,
        env_config: dict,
        reasoning_efforts: bool,
    ):
    if not reasoning_efforts:
        for prompt_style in prompt_styles:
            tasks = [main(prompt_style, model=model, num_games=num_games, env_config=env_config, reasoning_effort=None) for model in models]
            await asyncio.gather(*tasks)
    else:
        for reasoning_effort in ["low", "medium", "high"]:
            for prompt_style in prompt_styles:
                tasks = [main(
                        prompt_style,
                        model=model,
                        num_games=num_games,
                        env_config=env_config,
                        reasoning_effort=reasoning_effort,
                    ) for model in models]
                await asyncio.gather(*tasks)
            print(f"Completed all games with reasoning effort: {reasoning_effort}")

if __name__ == "__main__":
    models = [
        'deepseek/deepseek-r1-0528',
    ]
    reasoning_efforts: bool = False
    prompt_styles = [1, 2]
    num_games = 1
    env_config = {
        'arm_names': ['blue', 'green', 'red', 'yellow', 'purple'],
        'num_arms': 5,
        'max_steps': 20,
        'reward_type': 'gaussian',
        'noise_level': 'medium',
    }
    asyncio.run(run_all_games(
        prompt_styles=prompt_styles,
        models=models,
        num_games=num_games,
        env_config=env_config,
        reasoning_efforts=reasoning_efforts,
    ))
    print("\nAll games completed.")
