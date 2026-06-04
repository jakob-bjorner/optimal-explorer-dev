import sys
import pandas as pd
import json
import re
import re
sys.path.append('../../paprika/') # Remove this when verl and paprika are installed in the same env

from llm_exploration.paprika_config_helper import PaprikaConfigHelper
from verl.interactions.paprika_interaction import PaprikaInteraction

from pprint import pprint as pp

paprika_games = ['twenty_questions', 'guess_my_city', 'murder_mystery', 'customer_service', 'wordle', 'cellular_automata', \
    'mastermind'] # 'battleship', 'minesweeper', 'bandit_bai_fixed_budget' 

import dotenv
dotenv.load_dotenv('../../.env')

import asyncio

sys.path.append('../../src/optimal_explorer')
from llm_utils import llm_call
from pprint import pprint as pp

PORT = 37044

async def update_belief(
        global_info: str,
        curr_belief: str,
        action: str,
        response: str,
        model_name: str,
        word_limit,
    ):

    user_content = f'''\
This is the game information:
{global_info}
Look at the current belief and the agent's action and environment response on that belief. You have to update the current belief based on the action and response, while maintaining important information about the game state needed to take optimal future actions.
Current belief: {curr_belief}
Agent's action: {action}
Environment's response: {response}
Output the updated belief state inside <BELIEF> and </BELIEF> tags.\
Understand that only the generated belief is fed to the agent to pick the next action, not the history, so be sure to include all necessary information.'''

    if word_limit:
        user_content += f'\nMake sure that your belief inside the tags is no longer than {word_limit} words. If needed, compress or condense it.'

    messages = [
        {"role": "system", "content": 'You are a helpful assistant.'},
        {"role": "user", "content": user_content},
    ]

    if 'qwen' in model_name.lower():
        url = f"http://localhost:{PORT}/v1/chat/completions"
    else:
        url = None

    out = await llm_call(
        model=model_name,
        get_everything=True,
        reasoning_effort='high',
        messages=messages,
        url=url
    )

    content = out['choices'][0]['message']['content']
    match = re.search(r"<BELIEF>(.*?)</BELIEF>", content, re.DOTALL | re.IGNORECASE)
    if match:
        belief = match.group(1).strip()
    else:
        # fallback: return the whole content if tags not found
        belief = content.strip()
    
    if 'reasoning' in out['choices'][0]['message'].keys():
        reasoning = out['choices'][0]['message']['reasoning']
    elif 'reasoning_details' in out['choices'][0]['message'].keys() and len(out['choices'][0]['message']['reasoning_details']) > 0:
        reasoning = out['choices'][0]['message']['reasoning_details'][0]['text']
    else:
        reasoning = ''

    return belief, reasoning

async def take_action_both(
        global_info: str,
        belief: str,
        history_str: str,
        model_name: str,
        curr_attempt,
        total_attempts,
    ):
    if 'instruct' in model_name.lower():
        think_format = '<Think> Any step-by-step, short and concise thinking to determine what the next guess should be </Think>\n'
    else:
        think_format = ''
    user_content = f'''\
This is the game information:
{global_info}
You are currently taking your attempt {curr_attempt + 1}, and you have a total of {total_attempts} attempts.
Look at the current belief state and history and give an answer based on it.\
Give an answer that leads to optimal exploration and do not be greedy unless it is the last attempt. Try to maximize the amount of information you have so that you can solve the task correctly.\
History so far:
{history_str}
Belief: {belief}
Please format your response as: {think_format}<Answer> your answer in the correct format mentioned </Answer>'''

    messages = [
        {"role": "system", "content": 'You are a helpful assistant.'},
        {"role": "user", "content": user_content},
    ]

    if 'qwen' in model_name.lower():
        url = f"http://localhost:{PORT}/v1/chat/completions"
    else:
        url = None

    out = await llm_call(
        model=model_name,
        get_everything=True,
        reasoning_effort='high',
        messages=messages,
        url=url
    )

    content = out['choices'][0]['message']['content']
    match = re.search(r"<Answer>(.*?)</Answer>", content, re.DOTALL)
    action = match.group(1).strip() if match else content.strip()
    
    if 'reasoning' in out['choices'][0]['message'].keys():
        reasoning = out['choices'][0]['message']['reasoning']
    elif 'reasoning_details' in out['choices'][0]['message'].keys() and len(out['choices'][0]['message']['reasoning_details']) > 0:
        reasoning = out['choices'][0]['message']['reasoning_details'][0]['text']
    else:
        reasoning = ''

    return action, reasoning, content

async def take_action_history(
        global_info: str,
        history: str,
        model_name: str,
        curr_attempt,
        total_attempts,
    ):
    if 'instruct' in model_name.lower():
        think_format = '<Think> Any step-by-step, short and concise thinking to determine what the next guess should be </Think>\n'
    else:
        think_format = ''
    user_content = f'''\
This is the game information:
{global_info}
You are currently taking your attempt {curr_attempt + 1}, and you have a total of {total_attempts} attempts.
Look at the current history and give an answer based on it.\
Give an answer that leads to optimal exploration and do not be greedy unless it is the last attempt. Try to maximize the amount of information you have so that you can solve the task correctly.\
History so far:
{history}
Please format your response as: {think_format}<Answer> your answer in the correct format mentioned </Answer>'''

    messages = [
        {"role": "system", "content": 'You are a helpful assistant.'},
        {"role": "user", "content": user_content},
    ]

    if 'qwen' in model_name.lower():
        url = f"http://localhost:{PORT}/v1/chat/completions"
    else:
        url = None

    out = await llm_call(
        model=model_name,
        get_everything=True,
        reasoning_effort='high',
        messages=messages,
        url=url
    )

    content = out['choices'][0]['message']['content']

    match = re.search(r"<Answer>(.*?)</Answer>", content, re.DOTALL)
    action = match.group(1).strip() if match else content.strip()
    
    if 'reasoning' in out['choices'][0]['message'].keys():
        reasoning = out['choices'][0]['message']['reasoning']
    elif 'reasoning_details' in out['choices'][0]['message'].keys() and len(out['choices'][0]['message']['reasoning_details']) > 0:
        reasoning = out['choices'][0]['message']['reasoning_details'][0]['text']
    else:
        reasoning = ''

    return action, reasoning, content

async def take_action_belief(
        global_info: str,
        belief: str,
        model_name: str,
        curr_attempt,
        total_attempts,
    ):
    if 'instruct' in model_name.lower():
        think_format = '<Think> Any step-by-step, short and concise thinking to determine what the next guess should be </Think>\n'
    else:
        think_format = ''
    user_content = f'''\
This is the game information:
{global_info}
You are currently taking your attempt {curr_attempt + 1}, and you have a total of {total_attempts} attempts.
Look at the current belief state and give an answer based on it.\
Give an answer that leads to optimal exploration and do not be greedy unless it is the last attempt. Try to maximize the amount of information you have so that you can solve the task correctly.\
Belief: {belief}
Please format your response as: {think_format}<Answer> your answer in the correct format mentioned </Answer>'''

    messages = [
        {"role": "system", "content": 'You are a helpful assistant.'},
        {"role": "user", "content": user_content},
    ]

    if 'qwen' in model_name.lower():
        url = f"http://localhost:{PORT}/v1/chat/completions"
    else:
        url = None

    out = await llm_call(
        model=model_name,
        get_everything=True,
        reasoning_effort='high',
        messages=messages,
        url=url
    )

    content = out['choices'][0]['message']['content']
    match = re.search(r"<Answer>(.*?)</Answer>", content, re.DOTALL)
    action = match.group(1).strip() if match else content.strip()
    
    if 'reasoning' in out['choices'][0]['message'].keys():
        reasoning = out['choices'][0]['message']['reasoning']
    elif 'reasoning_details' in out['choices'][0]['message'].keys() and len(out['choices'][0]['message']['reasoning_details']) > 0:
        reasoning = out['choices'][0]['message']['reasoning_details'][0]['text']
    else:
        reasoning = ''

    return action, reasoning, content

async def run_one_iteration_llm(
        env_name: str,
        model_name: str,
        game_id: int,
        word_limit,
        logs_file,
        info, # 'belief', 'history', or 'both'
    ):
    config = PaprikaConfigHelper.create_config(env_name)
    config['belief_config']['style'] = 'none'
    interaction = PaprikaInteraction(config={})

    import builtins
    _original_print = builtins.print
    builtins.print = lambda *a, **k: None
    try:
        instance_id = await interaction.start_interaction(instance_id=None, scenario_id=None, **config)
    finally:
        builtins.print = _original_print

    first_user_message = interaction.agent_conv.messages[0][1]
    
    if 'instruct' not in model_name.lower():
        pattern = r"(?is)<Think[^>]*>(.*?)</Think>"
        first_user_message = re.sub(pattern, '', first_user_message)
    attempts = -1
    game_history = []
    belief = f'This is the start of the game. No beliefs right now.'
    max_attempts = interaction._instance_dict[instance_id]['max_turns']

    while attempts < max_attempts:
        
        attempts += 1

        history_str = ''
        for gh in game_history:
            history_str += f"Answer:{gh['guess']}\nResponse:{gh['response']}\n"

        if info == 'belief':
            action, action_reasoning, raw_action  = await take_action_belief(first_user_message, belief, model_name, attempts, max_attempts)
        elif info == 'history':
            action, action_reasoning, raw_action  = await take_action_history(first_user_message, history_str, model_name, attempts, max_attempts)
        elif info == 'both':
            action, action_reasoning, raw_action  = await take_action_both(first_user_message, belief, history_str, model_name, attempts, max_attempts)

        message = [
            {"role": "user", "content": f"Output the next action."},
            {"role": "assistant", "content": f"<action>{action}</action>"}
        ]
        done, response, score, additional_data = await interaction.generate_response(instance_id=instance_id, messages=message)
        
        if info == 'belief' or info == 'both':
            belief, belief_reasoning = await update_belief(first_user_message, belief, action, response, model_name, word_limit)
        else:
            belief, belief_reasoning = '', ''

        game_history.append({
            "model": model_name,
            "game_id": str(game_id),
            "env": env_name,
            "attempt": attempts,
            "info": info,
            "raw_guess": raw_action,
            "guess": action,
            "response": response,
            "word_limit": word_limit,
            "score": score,
            "done": done,
            "data": additional_data,
            "belief": belief,
            "action_reasoning": action_reasoning,
            "belief_reasoning": belief_reasoning,
        })

        if "Goal reached" in response:
            break
    
    with open(logs_file, "a") as f:
        for entry in game_history:
            f.write(json.dumps(entry) + "\n")

    print(f'.', end='', flush=True)
    
    return game_history

async def run_multiple_iterations_multiple_games(
        num_games: int,
        word_limits,
        list_envs,
        models,
        logs_file,
        infos,
    ):

    open(logs_file, "w").close()

    semaphore = asyncio.Semaphore(100)

    async def sem_task(*args, **kwargs):
        async with semaphore:
            return await run_one_iteration_llm(*args, **kwargs)

    tasks = []
    for game_id in range(num_games):
        for model in models:
            for env_name in list_envs:
                for word_limit in word_limits:
                    for info in infos:
                        tasks.append(sem_task(env_name, model, game_id, word_limit, logs_file, info))

    print('Start running.')
    results = await asyncio.gather(*tasks)

def main():
    asyncio.run(run_multiple_iterations_multiple_games(
        num_games=40,
        # word_limits=[200, 400, 800], # replace with [None] for no word limits on the belief state
        word_limits=[None],
        list_envs=['mastermind', 'twenty_questions', 'wordle', 'murder_mystery', 'customer_service', 'guess_my_city'],
        # list_envs=['mastermind'],
        models=['google/gemini-2.5-pro', 'deepseek/deepseek-chat', 'deepseek/deepseek-r1'],
        # models=['deepseek/deepseek-r1'],
        logs_file='./logs/paprika_frontier_v6.jsonl',
        infos=['belief', 'history', 'both'],
    ))

if __name__ == "__main__":
    main()