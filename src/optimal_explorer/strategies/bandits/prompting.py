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

DEBUG = False
run_name = ''

def save_game_log(
        game_id: int,
        history: List[Tuple[str, float]],
        total_reward: float,
        regret_per_attempt: List[float],
        prompt_style: int,
        model: str,
        env_config: dict,
        reasoning_effort: str = None,
        beliefs_per_step: list = None,
    ):
    """Save game log to a single JSONL file in the logs directory."""
    log_dir = Path(__file__).parent / "logs"
    if DEBUG:
        log_dir = Path(__file__).parent / "logs/debug"
    log_dir.mkdir(exist_ok=True)
    # ---
    if beliefs_per_step is not None:
        history_with_beliefs = [
            {
                "attempt": i + 1,
                "action": a,
                "reward": reward,
                "regret": regret_per_attempt[i] if i < len(regret_per_attempt) else None,
                "belief": beliefs_per_step[i] if i < len(beliefs_per_step) else None
            }
            for i, (a, reward) in enumerate(history)
        ]
    else:
        history_with_beliefs = [
            {
                "attempt": i + 1,
                "action": a,
                "reward": reward,
                "regret": regret_per_attempt[i] if i < len(regret_per_attempt) else None
            }
            for i, (a, reward) in enumerate(history)
        ]
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
        "history": history_with_beliefs
    }
    log_file = log_dir / f"game_results/style{prompt_style}_{model.split('/')[-1]}_{reasoning_effort if reasoning_effort else 'default'}_{env_config.get('reward_type', 'bernoulli')}{run_name}.jsonl"
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

async def update_belief(style, messages, llm_call_params, belief_model_call_store: dict, arm_names, history):
    if style in (6, 7):
        messages = deepcopy(messages)
        data: Dict = await llm_call(
            **llm_call_params,
            messages=messages,
        )
        print("^", end='', flush=True)
        belief_model_call_store.update(data)
        messages += [{'role': 'assistant', "content": data["choices"][0]["message"]["content"]}]
        messages += [{'role':"user", 'content': f"Now choose which arm to pull next based only on your current beliefs. Do not reconstruct or reason about previous actions or rewards. Please format your response as: <Action>arm_name</Action>. Do not say anything after the <Action> tags. Do not use markdown. The action tag should only contain the arm name."}]
    return messages

def process_belief_msg(msg_str, arm_names):
    # Extracts a belief dict from the LLM response (expects a JSON-like dict in the response)
    import re
    import ast
    # Try to find a dict in the string
    match = re.search(r'\{.*\}', msg_str, re.DOTALL)
    if match:
        try:
            belief = ast.literal_eval(match.group(0))
            # Only keep arms in arm_names
            return {str(k): v for k, v in belief.items() if str(k) in arm_names}
        except Exception:
            pass
    return {}

async def get_messages(
        prompt_style: int,
        arm_names: List[str],
        max_steps: int,
        history: List[Tuple[str, float]],
        messages,
        assistant_message_history: List[Dict[str, str]],
        error_handled: bool,
        reasoning_history: List[str],
        llm_call_params: dict,
        env_config: dict,
        belief_model_call_store=None,
    ) -> List[Dict[str, str]]:
    BASIC_SYSTEM_PROMPT = f"""You are playing a multi-armed bandit game. The rules are:\n1. There are {len(arm_names)} arms: {', '.join(arm_names)}.\n2. Each arm gives a reward drawn from a fixed but unknown distribution.\n3. You have {max_steps} steps. On each step, pick one arm to pull and observe the reward.\n4. Your goal is to maximize the total reward over all steps."""
    if prompt_style == 3:
        # Multi-turn, document the action and full reasoning history
        if messages is None:
            messages = [{"role": "system", "content": BASIC_SYSTEM_PROMPT + f"\nRespond with ONLY the name of the arm you want to pull next. Your reasoning history so far: {reasoning_history}"}]
            messages += [{"role": "user", "content": f"Pick your first arm ({', '.join(arm_names)}):"}]
        else:
            messages = deepcopy(messages)
            action, reward = history[-1]
            messages += [{"role": "assistant", "content": action}]
            messages += [{"role": "user", "content": f"{action} -> reward {reward:.3f}"}]
    elif prompt_style == 4:
        # Multi-turn, document some thinking and the action
        instruction_suffix = f"Please format your response as: <Think>Any step-by-step, short and concise thinking to determine which arm to pull next</Think><Action>arm_name</Action>. Do not say anything after the <Action> tags. Do not use markdown. The action tag should only contain the arm name."
        if messages is None:
            messages = [{"role": "system", "content": BASIC_SYSTEM_PROMPT}]
            messages += [{"role": "user", "content": f"Pick your first arm. " + instruction_suffix}]
        else:
            messages = deepcopy(messages)
            messages += [{"role": "assistant", "content": assistant_message_history[-1]['content']}]
            action, reward = history[-1]
            if error_handled:
                messages += [{"role": "user", "content": f"Your previous action was unable to be parsed. Now choose arm {len(history)+1}. " + instruction_suffix}]
            else:
                messages += [{"role": "user", "content": f"{action} -> reward {reward:.3f}. Now choose arm {len(history)+1}. " + instruction_suffix}]
    elif prompt_style == 5:
        # Multi-turn, document the belief state and the action
        instruction_suffix = f"""Now update your beliefs based on the observed rewards, and use your new beliefs to choose arm {len(history)+1}. Knowledge in your beliefs must only be updated but can never be discarded, forgotten, or removed. Do not say anything about which information is new and updated or old and remains the same.\nPlease format your response as: <Beliefs>Your beliefs on the reward distribution for each arm so far, in the format: {{'arm_name': {{'count': int, 'mean_reward': float}}, ...}}</Beliefs><Action>arm_name</Action>. Do not say anything after the <Action> tags. Do not use markdown. The action tag should only contain the arm name. """
        if messages is None:
            messages = [{"role": "system", "content": BASIC_SYSTEM_PROMPT}]
            messages += [{"role": "user", "content": f"Pick your first arm. Please format your response as: <Beliefs>Your beliefs on the reward distribution for each arm so far, in the format: {{'arm_name': {{'count': int, 'mean_reward': float}}, ...}}</Beliefs><Action>arm_name</Action>. Do not say anything after the <Action> tags. Do not use markdown. The action tag should only contain the arm name."}]
        else:
            messages = deepcopy(messages)
            messages += [{"role": "assistant", "content": assistant_message_history[-1]['content']}]
            action, reward = history[-1]
            if error_handled:
                messages += [{"role": "user", "content": f"Your previous action was unable to be parsed. " + instruction_suffix}]
            else:
                messages += [{"role": "user", "content": f"{action} -> reward {reward:.3f}. " + instruction_suffix}]
    elif prompt_style in (6, 7):
        # Multi-turn, belief and action in separate steps
        belief_instruction_suffix = """Now update your beliefs about the reward distribution for each arm with the latest reward. Knowledge in your beliefs must only be updated but can never be discarded, forgotten, or removed. Do not say anything about which information is new and updated or old and remains the same.\nPlease format your response as: <Beliefs>Your new beliefs in the format: {'arm_name': {'count': int, 'mean_reward': float}, ...}</Beliefs>"""
        if messages is None:
            messages = [{"role": "system", "content": BASIC_SYSTEM_PROMPT}]
            messages += [{"role": "user", "content": f"Construct a belief state from which you will be able to choose your first arm. But do not choose yet. Please format your response as: <Beliefs>Your beliefs on the reward distribution for each arm so far, in the format: {{'arm_name': {{'count': int, 'mean_reward': float}}, ...}}</Beliefs>."}]
        else:
            messages = deepcopy(messages)
            messages += [{"role": "assistant", "content": assistant_message_history[-1]['content']}]
            action, reward = history[-1]
            if error_handled:
                messages += [{"role": "user", "content": f"Your previous action was unable to be parsed. " + belief_instruction_suffix}]
            else:
                messages += [{"role": "user", "content": f"{action} -> reward {reward:.3f}. " + belief_instruction_suffix}]
        messages = await update_belief(
            prompt_style,
            messages,
            llm_call_params,
            belief_model_call_store,
            arm_names,
            history)
    else:
        raise Exception(f"invalid prompt_style = {prompt_style}")
    return messages

def process_action_msg(msg_str, arm_names):
    # Extract the arm name from the LLM response robustly, including <Action> tags
    import re
    msg_str = msg_str.strip().lower()
    # Try to extract from <Action> tags
    match = re.search(r'<action>(.*?)</action>', msg_str, re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        for arm in arm_names:
            if arm.lower() == candidate:
                return arm
    # fallback: look for arm name in string
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
    belief_model_call_store = {}
    beliefs_per_step = []  # Collect beliefs for main game log
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
            env_config,
            belief_model_call_store
        )
        # For prompt_style 7, only send system, last belief, and last instruction
        if prompt_style == 7 and len(messages) >= 3:
            llm_call_messages = messages[0:1] + messages[-2:]
        else:
            llm_call_messages = messages
        data: Dict = await llm_call(
            **llm_call_params,
            messages=llm_call_messages,
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
        if len(belief_model_call_store) != 0:
            data['other_lm_calls'] = [belief_model_call_store]
        error_handled = False
        if 'choices' in data and data['choices']:
            llm_response = data['choices'][0]['message']['content'] if 'message' in data['choices'][0] else data['choices'][0]['text']
        else:
            llm_response = ''
        reasoning_str = data['choices'][0]['message']['reasoning'] if 'reasoning' in data['choices'][0].get("message", dict()) else ''
        reasoning_history.append(reasoning_str)
        assistant_message_history += [data["choices"][0]["message"] if 'message' in data['choices'][0] else {'role': 'assistant', 'content': data['choices'][0]['text']}]
        # --- Extract and log belief ---
        belief = None
        if prompt_style in (5, 6, 7):
            # Try to extract <Beliefs>...</Beliefs> from the LLM response
            import re
            match = re.search(r'<beliefs>(.*?)</beliefs>', llm_response, re.IGNORECASE | re.DOTALL)
            if match:
                belief_str = match.group(1)
                belief = process_belief_msg(belief_str, arm_names)
            else:
                # fallback: try to extract dict from whole response
                belief = process_belief_msg(llm_response, arm_names)
        data['belief'] = belief
        beliefs_per_step.append(belief)
        # ---
        action = process_action_msg(llm_response, arm_names)
        if action not in arm_names:
            error_handled = True
        data['error_handled'] = error_handled
        log_dir = Path(__file__).parent / "logs/llm_calls"
        if DEBUG:
            log_dir = Path(__file__).parent / "logs/debug/llm_calls"
        log_dir.mkdir(exist_ok=True)
        reasoning_effort_str = reasoning_effort if reasoning_effort else "default"
        log_file = log_dir / f"style{prompt_style}_{model.split('/')[-1]}_{reasoning_effort_str}_{env_config.get('reward_type', 'bernoulli')}{run_name}.jsonl"
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
    # --- Save beliefs in main game log ---
    def add_beliefs_to_history(history, beliefs_per_step):
        new_history = []
        for i, (item, belief) in enumerate(zip(history, beliefs_per_step)):
            a, reward = item
            new_item = {
                'attempt': i + 1,
                'action': a,
                'reward': reward,
                'belief': belief
            }
            new_history.append(new_item)
        return new_history
    # Patch save_game_log call to include beliefs
    save_game_log(game_id, history, total_reward, regret_per_attempt, prompt_style, model, env_config, reasoning_effort=reasoning_effort, beliefs_per_step=beliefs_per_step)
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
        for file in log_dir.rglob(f"style{prompt_style}_{model.split('/')[-1]}*{run_name}.jsonl"):
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
        'google/gemini-2.5-pro',
        'deepseek/deepseek-r1-0528',
        # 'openai/gpt-4.1-nano'
    ]
    reasoning_efforts: bool = False
    prompt_styles = [3, 5, 6]
    num_games = 50
    env_config = {
        'arm_names': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'],
        'num_arms': 10,
        'max_steps': 20,
        'reward_type': 'bernoulli',
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
