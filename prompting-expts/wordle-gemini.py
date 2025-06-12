import numpy as np
from itertools import permutations
import torch
import random
import sys
sys.path.append('../src')

from optimal_explorer.mdps.combination_lock import CombinationLock
from optimal_explorer.mdps.wordle import Wordle
from optimal_explorer.llm_utils import llm_call
from pprint import pprint
import asyncio
from typing import cast, Any
import time
import json
from copy import deepcopy

def generate_wordle_message_history(state_action_feedback, variation: int = 0): # there are many variations of how you could construct the user prompt form history.
  message_history = []
  # variations are such that I can completely rewrite the game state or I could just append to the prior message.
  if variation == 0: # This one will be just append to prior message history...
    system_prompt =  ("You are playing a wordle game. The rules are:\n"
                    "1. Objective - Guess the secret five-letter English word within six attempts.\n"
                    "2. Enter valid words only - Each guess must be a real, spelled-correctly five-letter word; otherwise Wordle rejects the entry.\n"
                    "3. Color feedback after each guess\n"
                    " - Green - the letter is in the word and in the correct position.\n"
                    " - Yellow - the letter is in the word but in a different position.\n"
                    " - Gray - the letter does not appear in the word at all.\n"
                    "4. Duplicate letters matter - If a letter occurs more than once in the solution, feedback colors account for the exact number and positions of those letters.\n"
                    "5. Respond with ONLY your 5 letter guess, nothing else.")
    # import ipdb; ipdb.set_trace()
    if state_action_feedback is not None:
      message_history = deepcopy(state_action_feedback['state']['message_history'])
      last_message = state_action_feedback['llm_response']["choices"][0]['message']
      message_history.append(last_message)
      feedback = state_action_feedback['feedback']['info']['feedback']
      feedback_str = ''.join(['⬜' if f == 0 else '🟨' if f == 1 else '🟩' for f in feedback])
      guess_str = state_action_feedback["action"]
      message_history.append({"role": "user", "content": f"Feedback: {guess_str} -> {feedback_str}"})
    else:
      message_history.append({"role": "system", "content": system_prompt})
      message_history.append({"role": "user", "content": "Provide your first guess."})

  else:
    raise Exception(f"invalid variation chosen {variation}")
      
  return message_history

async def easy_wordle_game(game_id = 0, model_name="openai/gpt-4o-mini", temperature = 1):
        mdp = Wordle()
        mdp.reset(seed=game_id)  # Use game_id as seed for reproducibility
        history = []
        regret_per_attempt = []
        log_info: dict[str, Any] = dict(errored = False,
                                        game_id = game_id,
                                        state_action_feedback = [],
                                        model_name = model_name,
                                        temperature = temperature,
                                        success = None,
                                        num_turns = -1,
                                        target_word = mdp.target_word)
        done = False
        reward = 0
        while not done:
                state_action_feedback_dict: dict[str, Any] = {"state": None,
                                              "action": None,
                                              "llm_response": None,
                                              "feedback": None,}
                message_history = generate_wordle_message_history(None if len(log_info["state_action_feedback"]) == 0 else log_info['state_action_feedback'][-1])
                # Get LLM's guess
                log_info['state_action_feedback'].append(state_action_feedback_dict) # in case we raise execption but want to inspect the object in debugging.
                state_action_feedback_dict['state'] = {
                        'message_history': message_history,
                        "game_history_state_t-1": deepcopy(history), 
                        }
                # several options the llm call can't time out, we will just always wait right? 
                # and I can manually control c or force terminate every python thing with pgrep -u jbjorner3 python -d ' '
                # what happens if I send a request which is like 1000 gemini 2.5 calls on acident... wost case I lose $20
                llm_response = await llm_call(messages=message_history,
                        model=model_name,
                        temperature=temperature,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        repetition_penalty=1,
                        top_k=0,
                        get_everything=True)
                
                llm_response = cast(Any, llm_response)
                llm_response_content = llm_response['choices'][0]['message']['content']

                state_action_feedback_dict['action'] = llm_response_content
                state_action_feedback_dict['llm_response'] = llm_response
                # Clean up response to get just the guess
                def extract_guess(raw_content: str): # for wordle this is simple.
                        return ''.join(c for c in raw_content if c.isalpha())
                guess = extract_guess(llm_response_content) # function to digest the raw content
                # Make the guess
                obs, reward, done, info = mdp.step(guess)
                if "feedback" not in info:
                        log_info['errored'] = True
                        # pprint(log_info)
                        raise Exception(f"invalid guess '{guess}'", log_info)

                feedback = info['feedback']
                history.append((guess, feedback))
                state_action_feedback_dict['feedback'] = {
                        "obs": deepcopy(obs),
                        "reward": reward,
                        "done": done,
                        "info": info,
                        "game_history_state_t": deepcopy(history),
                }

                # Calculate regret as shortfall against optimal value function
                # V*(s) = 1, V^π(s) = 1 if solved, 0 if not solved
                regret_per_attempt.append(1.0 - (1.0 if done and reward == 1.0 else 0.0))
                # res
        log_info['success'] = reward
        log_info['num_turns'] = len(regret_per_attempt)
        return log_info

model_name = ["google/gemini-2.5-pro-preview", "deepseek/deepseek-r1-0528", "openai/gpt-4o-mini"][0]
num_times = 10
temperature = 1

async def main():
    res = await asyncio.gather(*[easy_wordle_game(i, model_name, temperature) for i in range(num_times)], return_exceptions=False)

    cum_regret = np.mean([contains_success(r) for r in res if isinstance(r, dict)]), len([r for r in res if isinstance(r, dict)])

    print(f"Average number of turns to solve wordle with {model_name} is {cum_regret[0]} over {cum_regret[1]} games.")
    # save this info
    with open(f"wordle_gemini_summary_num_times_{num_times}_temperature_{temperature}.txt", "w") as f:
        json.dump({"model_name": model_name, "num_times": num_times, "temperature": temperature, "cum_regret": cum_regret}, f, indent=2)
    print(f"Saved results to wordle_gemini_results.txt")

    def contains_success(r):
        target_word = r['target_word']
        return int([a['action'].lower() for a in r['state_action_feedback']].index(target_word))

    with open(f"wordle_gemini_{model_name.split('/')[-1]}_num_times_{num_times}_temperature_{temperature}.txt", "w") as f:
        json.dump(res, f, indent=2)

    print(f"Saved results to wordle_gemini_{model_name.split('/')[-1]}_num_times_{num_times}_temperature_{temperature}.txt")

if __name__ == "__main__":
    asyncio.run(main())