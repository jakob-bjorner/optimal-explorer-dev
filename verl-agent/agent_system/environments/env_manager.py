# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory
from copy import deepcopy
import re
import requests

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs


class ComboLockEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = []
        super().__init__(envs, projection_f, config)
    
    def reset(self):
        text_obs, infos = self.envs.reset()

        # self.supervisors = [info['supervisor'] for info in infos]
        self.memory = []
        self.action_or_belief = np.zeros(len(text_obs), dtype=np.bool) # 0 means actions being generated, 1 means beliefs
        self.prior_beliefs = [None] * len(text_obs)
        self.prior_belief_messages: List = [None] * len(text_obs) # this for tracking when a belief is not being generated correctly.
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs
        self.belief_generation_failures = 0
        self.action_generation_failures = 0

        chat = self._build_chat_obs(text_obs, None, None, None, None, None, init=True)
        return {'text': ['']*len(chat), 'image': None, 'anchor': text_obs, 'chat': chat}, infos
    def get_belief_from_output_text(self, text_beliefs_raw: List[str]):
        action_or_belief_texts, valids = self.projection_f(text_beliefs_raw, np.ones(len(text_beliefs_raw)))
        return action_or_belief_texts, valids
    def step(self, text_actions: List[str], is_not_processing, tokenizer):
        # generate_belief = bool(self.step_idx % 2 != 0) This doesn't determine whether belief is generated or not lol. 
        # belief determination is a property of the environment tho, so should maintain belief or not as an array, 
        # where I can change belief or not depending on if a valid action is passed in either case.

        action_or_belief_texts, valids = self.projection_f(text_actions, self.action_or_belief)
        # chat = self._build_chat_obs(beliefs, valids)
        # next_observations = {'text': '', 'image', None, 'anchor': text_obs, 'chat':chat}
        # actions, valids = self.projection_f(text_actions, self.action_or_belief)
        skip_sampling_mdp = self.action_or_belief | ~np.array(valids, dtype=bool) | is_not_processing
        actions = list(zip(action_or_belief_texts, skip_sampling_mdp))
        text_obs, rewards, dones, infos = self.envs.step(actions)

        new_action_or_belief = ~(self.action_or_belief & np.array(valids, dtype=np.bool)) # you go to belief state unless you generate a valid belief while in belief state.
        if self.config.actor_rollout_ref.rollout.single_context and not self.config.actor_rollout_ref.rollout.belief_multiple_messages:
            new_action_or_belief = self.action_or_belief * 0
        self.belief_generation_failures += (self.action_or_belief & ~np.array(valids, dtype=np.bool) & ~is_not_processing).sum()
        self.action_generation_failures += (~self.action_or_belief & ~np.array(valids, dtype=np.bool) & ~is_not_processing).sum()
        self.memory.append(deepcopy((text_obs, rewards, dones, infos, new_action_or_belief, valids, action_or_belief_texts)))
        self.pre_text_obs = text_obs

        # full_text_obs = self.build_text_obs(text_obs)
        chat = self._build_chat_obs(text_obs, text_actions, action_or_belief_texts, valids, self.action_or_belief, new_action_or_belief)
        # add action_valid to infos
        for i, info in enumerate(infos):
            infos[i]["action_or_belief"] = int(self.action_or_belief[i])
            infos[i]['is_action_valid'] = int(valids[i])



        
        beliefs = [belief if valid and self.action_or_belief[i] == 1 else "" for i, (belief, valid) in enumerate(zip(action_or_belief_texts, valids))]
        actions = [action if valid and self.action_or_belief[i] == 0 else "" for i, (action, valid) in enumerate(zip(action_or_belief_texts, valids))]

        self.action_or_belief = new_action_or_belief
        next_observations = {'text': ['']*len(chat), "filtered_belief_generations": beliefs, "filtered_action_generations": actions, 'chat': chat, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos


    def _build_chat_obs(self, text_obs, full_text_actions, actions_or_beliefs, valids, old_action_or_belief, new_action_or_belief, init: bool = False)-> List[Dict[str,str]]:
        postprocess_text_obs = []
        if init:
            # valids wil be none, and actions will also be none, so I just give the chat history which is default for every first interaction.
            for i in range(len(text_obs)):
                postprocess_text_obs.append([{'role': "system", 'content': COMBO_AGENT_FIRST_MESSAGE.format_map({"vocab_list": list(self.config.env.vocab), "max_attempts": self.config.env.max_attempts})},
                                            {'role': "user", 'content': COMBO_FIRST_USER_MESSAGE}])
        else:
            # list of 0 or 1 indicating action or belief being generated.
            # this is updated right before this function call, 
            # so the chat should be preparing the lm call to generate the indicated item.
            # self.action_or_belief 
            # self.memory
            for i in range(len(new_action_or_belief)):
                if new_action_or_belief[i]:
                    # this is belief generation prep
                    # system prompt automatically added when there is no spec.
                    if len(self.memory) == 1:
                        # first belief generation message.
                        prior_belief = COMBO_NO_PRIOR_BELIEF_MESSAGE
                    else:
                        prior_belief = self.prior_beliefs[i]
                    # we may have just come from a long string of belief generation failures, 
                    # so we need to reconstruct the history of failures if this is the case.
                    if old_action_or_belief[i]:
                        # we were previously generating a belief, 
                        # and are still generating a belief, in this case, 
                        # we need to correct some error in the belief generation.
                        new_belief_messages = deepcopy(self.prior_belief_messages[i])

                        new_belief_messages += [{'role': "assistant", 'content': full_text_actions[i]},
                                                {'role': 'user', "content": COMBO_BELIEF_GENERATION_FAILURE_MSG}]
                        if self.config.actor_rollout_ref.rollout.single_context:
                            new_belief_messages = [{'role': 'user', "content": COMBO_BELIEF_GENERATION_FAILURE_MSG}]
                        
                    else:
                        # we are for the first time generating a belief message
                        if valids[i]: 
                            agent_action = full_text_actions[i].split("<action>")[1].split("</action>")[0].strip()
                            env_response = text_obs[i]
                        else:
                            agent_action = "invalid action"
                            if "<action>" in full_text_actions[i] and "</action>" in full_text_actions[i].split("<action>")[1]:
                                # this has a different error message than the other one. lol.
                                content_summary = full_text_actions[i] if len(full_text_actions[i]) < 20 else f"...{full_text_actions[i][-20:]}"
                                env_response = f"Could not parse valid guess from: '{content_summary}'. Please ensure the guess is contained in the final characters of your response, and using only use the characters from the vocab in your guess characters. Do not repeat characters in your guess."
                            else:
                                # so we don't have the tags correct in this one.
                                env_response = 'Could not parse response. Please ensure your response is in the format: <action> ... </action>.'

                        new_belief_messages = [{'role': "user", 'content': COMBO_BELIEF_PROMPT.format(agent_first_message=COMBO_AGENT_FIRST_MESSAGE.format_map({"vocab_list": list(self.config.env.vocab), "max_attempts": self.config.env.max_attempts}),
                                                                                                    belief_state=prior_belief,
                                                                                                    agent_action=agent_action,
                                                                                                    env_response=env_response)}]
                        if self.config.actor_rollout_ref.rollout.single_context:
                            new_belief_messages = [{'role': "user", 'content': env_response}]
                            if self.config.actor_rollout_ref.rollout.belief_multiple_messages:
                                new_belief_messages += [{'role': 'user', 'content': COMBO_BELIEF_PROMPT_SINGLE_CONTEXT}]
                        # prior_belief = self.prior_beliefs[i]
                    self.prior_belief_messages[i] = new_belief_messages
                    postprocess_text_obs.append(new_belief_messages)
                else:
                    # this is action generation prep
                    if self.config.actor_rollout_ref.rollout.single_context and not self.config.actor_rollout_ref.rollout.belief_multiple_messages:
                        env_response = text_obs[i]
                        new_action_messages = [{'role': "user", 'content': env_response}]
                        postprocess_text_obs.append(new_action_messages)
                    else:
                        assert valids[i], f"must be valid, but got {valids[i]=}"
                        # you can only be generating an action after a successful belief generation with the first prompt being a special case in this repo.
                        belief = actions_or_beliefs[i]
                        self.prior_beliefs[i] = belief
                        self.prior_belief_messages[i] = None
                        new_action_messages = [{'role':'user', 'content': COMBO_ACTION_PROMPT.format(agent_first_message=COMBO_AGENT_FIRST_MESSAGE.format_map({"vocab_list": list(self.config.env.vocab), "max_attempts": self.config.env.max_attempts}),
                                                                                                    belief_state=belief)}]
                        if self.config.actor_rollout_ref.rollout.single_context:
                            new_action_messages = [{'role': "user", 'content': COMBO_ACTION_PROMPT_SINGLE_CONTEXT}]
                        postprocess_text_obs.append(new_action_messages)
        return postprocess_text_obs
    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Evaluate if the episodes are successful or not. 
        (Default) implementation is to check info['won'] of the last step.
        
        Returns:
        - success (np.ndarray or torch.Tensor): 1 if the episode is successful, 0 otherwise.
        """
        # figure out how to calculate the regret so we can see how close to R1 performance we are while training.
        def get_regret_end_val_from_episode_rewards(reward_per_traj_data: np.ndarray, max_attempts):
            regret_arr_per_traj = []
            # print(np.mean(reward_per_traj_data))
            for reward in reward_per_traj_data:
                # invert the attempts -> reward calculation. 
                # reward = (max_attempts - attempts + 1) / max_attempts => attempts = -reward * max_attempts + 1 + max_attempts
                if reward == -1.0:
                    attempts = max_attempts
                else:
                    attempts = int(1 + max_attempts - reward * max_attempts)
                # print(attempts)
                regret_arr_per_traj.append([1] * attempts + [0] * (max_attempts - attempts))
            # pprint(regret_arr_per_traj)
            avg_regret = np.mean(regret_arr_per_traj, axis=0).cumsum()
            # sem_regret = np.std(regret_arr_per_traj, axis=0) / np.sqrt(len(avg_regret)) * 1.96
            return avg_regret[-1]

        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        batch_size = len(total_batch_list)
        
        success = defaultdict(list)
        
        for bs in range(batch_size):
            self._process_batch(bs, total_batch_list, total_infos, success)
        
        assert len(success['success_rate']) == batch_size

        regret_tail_value = get_regret_end_val_from_episode_rewards(kwargs['episode_rewards'], self.config.env.max_attempts)

        return {key: np.array(value) for key, value in success.items()} | {"action_generation_failures_success_rate": np.array([self.action_generation_failures/batch_size]*batch_size), 
                                                                           "belief_generation_failures_success_rate": np.array([self.belief_generation_failures/batch_size]*batch_size),
                                                                           "regret_tail_value_success_rate": np.array([regret_tail_value/ batch_size]*batch_size),} # just for the metric calc to be the same.



class NQHotpotQAEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = []
        super().__init__(envs, projection_f, config)

    def reset(self):
        text_obs, infos = self.envs.reset()
        self.questions = text_obs
        # text_obs is the questions in string form, already trimmed, should put the prompt text around this now.

        # self.supervisors = [info['supervisor'] for info in infos]
        self.memory = []
        self.action_or_belief = np.zeros(len(text_obs), dtype=np.bool) # 0 means actions being generated, 1 means beliefs
        self.prior_beliefs = [None] * len(text_obs)
        self.prior_action = [None] * len(text_obs)
        self.prior_obs = [None] * len(text_obs)
        self.prior_belief_messages: List = [None] * len(text_obs) # this for tracking when a belief is not being generated correctly.
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs
        self.belief_generation_failures = 0
        self.action_generation_failures = 0
        self.successful_searches = 0
        is_not_processing = np.zeros(len(text_obs), dtype=np.bool)
        chat = self._build_chat_obs(text_obs, None, None, None, None, None, None, None, is_not_processing, None, init=True)
        return {'text': ['']*len(chat), 'image': None, 'anchor': text_obs, 'chat': chat}, infos



    def step(self, text_actions: List[str], is_not_processing, tokenizer):
        # generate_belief = bool(self.step_idx % 2 != 0) This doesn't determine whether belief is generated or not lol. 
        # belief determination is a property of the environment tho, so should maintain belief or not as an array, 
        # where I can change belief or not depending on if a valid action is passed in either case.
        tags, action_or_belief_texts, valids = self.projection_f(text_actions, self.action_or_belief)


        skip_sampling_mdp = self.action_or_belief | ~np.array(valids, dtype=bool) | is_not_processing
        actions = list(zip(tags, action_or_belief_texts, skip_sampling_mdp))
        text_obs, rewards, dones, infos = self.envs.step(actions) 
        dones = np.logical_or(dones, np.logical_not(valids))
        # perform the search in a grouped fashion, ensure the step isn't done, the question is processing, and that the action is search.
        search_queries = [content for i, (action, content) in enumerate(zip(tags, action_or_belief_texts)) if action == 'search' and not skip_sampling_mdp[i]]
        search_results = self.batch_search(search_queries)
        for i, (action, content) in enumerate(zip(tags, action_or_belief_texts)):
            # update the info dict with information on whether it was a action or belief being generated or not
            infos[i]["action_or_belief"] = int(self.action_or_belief[i])
            infos[i]['is_action_valid'] = int(valids[i])
            if action == 'search' and not skip_sampling_mdp[i]:
                hint = text_obs[i] # the environment tells you the turns remaining
                if self.config.env.is_mem1:
                    text_obs[i] = NQHOTPOTQA_ENV_RESPONSE_SEARCH_MEM1.format_map({"hint": hint, "search_result": search_results.pop(0).strip()})
                else:
                    text_obs[i] = NQHOTPOTQA_ENV_RESPONSE_SEARCH.format_map({"hint": hint, "search_result": search_results.pop(0).strip()})
                # f"{hint}\n\n{search_results.pop(0).strip()}"
                self.successful_searches += 1
        # prune the text_obs here, just in case they are too long.
        for i in range(len(text_obs)):
            if len(text_obs[i]) > self.config.env.max_obs_length:
                # prune to length allowed for observation.
                text_obs[i] = tokenizer.decode(tokenizer.encode(text_obs[i], add_special_tokens=False)[:self.config.env.max_obs_length])

        new_action_or_belief = ~(self.action_or_belief & np.array(valids, dtype=np.bool)) # you go to belief state unless you generate a valid belief while in belief state.
        if self.config.actor_rollout_ref.rollout.single_context and not self.config.actor_rollout_ref.rollout.belief_multiple_messages:
            new_action_or_belief = self.action_or_belief * 0
        if self.config.env.is_mem1:
            new_action_or_belief = self.action_or_belief * 0 # we handle the logic for mem1 with only actions even though we want beliefs to be generated as well.
            # if invalid in mem1, it just terminates.
            is_not_processing = ~np.array(valids, dtype=np.bool) | is_not_processing
            dones = [(d or not v) for d, v in zip(dones, valids)]
        self.belief_generation_failures += (self.action_or_belief & ~np.array(valids, dtype=np.bool) & ~is_not_processing).sum()
        self.action_generation_failures += (~self.action_or_belief & ~np.array(valids, dtype=np.bool) & ~is_not_processing).sum()
        self.memory.append(deepcopy((text_obs, rewards, dones, infos, new_action_or_belief, valids, action_or_belief_texts)))
        self.pre_text_obs = text_obs

        # full_text_obs = self.build_text_obs(text_obs)
        chat = self._build_chat_obs(text_obs, text_actions, tags, infos, action_or_belief_texts, valids, self.action_or_belief, new_action_or_belief, is_not_processing, tokenizer)
        

        beliefs = [belief if valid and tag == 'belief' else "" for i, (tag, belief, valid) in enumerate(zip(tags, action_or_belief_texts, valids))]
        actions = [action if valid and (tag == 'search' or tag == "answer") else "" for i, (tag, action, valid) in enumerate(zip(tags, action_or_belief_texts, valids))]
        self.action_or_belief = new_action_or_belief
        next_observations = {'text': ['']*len(chat), "filtered_belief_generations": beliefs, "filtered_action_generations": actions, 'chat': chat, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def batch_search(self, queries: List[str]) -> list[str]:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']
        
        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):
        
        payload = {
            "queries": queries,
            "topk": self.config.env.topk,
            "return_scores": True
        }
        try:
            return requests.post(self.config.env.search_url, json=payload).json()
        except Exception as e:
            print(f"Error in batch_search: {e}")
            return []

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
    def _build_chat_obs(self, text_obs, full_text_actions, tags, infos, actions_or_beliefs, valids, old_action_or_belief, new_action_or_belief, is_not_processing, tokenizer, init: bool = False)-> List[Dict[str,str]]:
        postprocess_text_obs = []
        # valids wil be none, and actions will also be none, so I just give the chat history which is default for every first interaction.
        for i in range(len(text_obs)):
            if is_not_processing[i]:
                postprocess_text_obs.append([{'role': "user", 'content': "yo"}]) # this will be replaced at the rollout_loop.py level, and this if statement is just here to filter from going through the logic needlessly
                continue
            if init:
                if self.config.env.is_mem1:
                    if self.config.env.force_full_step_len:
                        postprocess_text_obs.append([{'role': "user", 'content': get_NQHOTPOTQA_FULL_AGENT_FIRST_MESSAGE_MEM1(self.questions[i], self.config.actor_rollout_ref.rollout.instruct)}])
                    else:
                        postprocess_text_obs.append([{'role': "user", 'content': get_NQHOTPOTQA_AGENT_FIRST_MESSAGE_MEM1(self.questions[i], self.config.actor_rollout_ref.rollout.instruct)}])
                    # need to add the generation template manually, because it was done wrong in MEM1.
                else:
                    if self.config.env.force_full_step_len:
                        postprocess_text_obs.append([{'role': "system", 'content': NQHOTPOTQA_FULL_AGENT_FIRST_MESSAGE.format_map({"question": self.questions[i]})},
                                                    {'role': "user", 'content': NQHOTPOTQA_FULL_FIRST_USER_MESSAGE}])
                    else:
                        postprocess_text_obs.append([{'role': "system", 'content': NQHOTPOTQA_AGENT_FIRST_MESSAGE.format_map({"question": self.questions[i]})},
                                                    {'role': "user", 'content': NQHOTPOTQA_FIRST_USER_MESSAGE}])
            else:
                # list of 0 or 1 indicating action or belief being generated.
                # this is updated right before this function call, 
                # so the chat should be preparing the lm call to generate the indicated item.
                # self.action_or_belief 
                # self.memory
                if new_action_or_belief[i]:
                    # this is belief generation prep
                    # system prompt automatically added when there is no spec.
                    if len(self.memory) == 1:
                        # first belief generation message.
                        prior_belief = NQHOTPOTQA_NO_PRIOR_BELIEF_MESSAGE
                    else:
                        prior_belief = self.prior_beliefs[i]
                    # we may have just come from a long string of belief generation failures, 
                    # so we need to reconstruct the history of failures if this is the case.
                    if old_action_or_belief[i]:
                        # we were previously generating a belief, 
                        # and are still generating a belief, in this case, 
                        # we need to correct some error in the belief generation.
                        new_belief_messages = deepcopy(self.prior_belief_messages[i])
                        # just regenerate, you don't want to throw anything away, and you didn't do it right, 
                        # so yeah, it might happen infinitely whatever. with temp 1 in our training and test setting, should be fine.
                        # new_belief_messages += [{'role': "assistant", 'content': full_text_actions[i]},
                        #                         {'role': 'user', "content": NQHOTPOTQA_BELIEF_GENERATION_FAILURE_MSG}]
                        if self.config.actor_rollout_ref.rollout.single_context:
                            new_belief_messages = [{'role': 'user', "content": NQHOTPOTQA_BELIEF_GENERATION_FAILURE_MSG}]
                    else:
                        # we are for the first time generating a belief message
                        if valids[i]: 
                            agent_action = actions_or_beliefs[i]
                            env_response = text_obs[i]
                        else:
                            agent_action = "invalid action"
                            env_response = NQHOTPOTQA_ENV_RESPONSE
                        self.prior_action[i] = agent_action
                        self.prior_obs[i] = env_response

                        if self.config.env.force_full_step_len:
                            agent_first_message = NQHOTPOTQA_FULL_AGENT_FIRST_MESSAGE.format_map({"question": self.questions[i]})
                        else:
                            agent_first_message = NQHOTPOTQA_AGENT_FIRST_MESSAGE.format_map({"question": self.questions[i]})
                        new_belief_messages = [{'role': "user", 'content': NQHOTPOTQA_BELIEF_PROMPT.format(agent_first_message=agent_first_message,
                                                                                                    belief_state=prior_belief,
                                                                                                    agent_action=agent_action,
                                                                                                    env_response=env_response)}]
                        if self.config.actor_rollout_ref.rollout.single_context:
                            new_belief_messages = [{'role': "user", 'content': env_response}]
                            if self.config.actor_rollout_ref.rollout.belief_multiple_messages:
                                new_belief_messages += [{'role': 'user', 'content': NQHOTPOTQA_BELIEF_PROMPT_SINGLE_CONTEXT}]
                        # prior_belief = self.prior_beliefs[i]
                    self.prior_belief_messages[i] = new_belief_messages
                    postprocess_text_obs.append(new_belief_messages)
                else:
                    # this is action generation prep
                    if self.config.env.is_mem1:
                        # you can only be generating an action after a successful belief generation with the first prompt being a special case in this repo.
                        # this extraction strategy is taken from mem1
                        belief = "<think>" + full_text_actions[i].split('<think>')[1] if '<think>' in full_text_actions[i] else full_text_actions[i]
                        
                        new_action_messages = [{'role': "user", 'content': (get_NQHOTPOTQA_FULL_AGENT_FIRST_MESSAGE_MEM1 if self.config.env.force_full_step_len else get_NQHOTPOTQA_AGENT_FIRST_MESSAGE_MEM1)(self.questions[i], self.config.actor_rollout_ref.rollout.instruct)},
                                               {'role': 'assistant', "content": belief},
                                               {'role': 'user', 'content': text_obs[i]}]
                        postprocess_text_obs.append(new_action_messages)
                    else:
                        if self.config.actor_rollout_ref.rollout.single_context and not self.config.actor_rollout_ref.rollout.belief_multiple_messages:
                            env_response = text_obs[i]
                            # for the single_context withoutout belief messages (ie Vanilla setting), we modify the environment response, such that it's hint is consistent with ABBEL's hint formatting not MEM1's.
                            
                            if "<hint>" in env_response and "</hint>" in env_response:
                                env_response = env_response.split("<hint>")[0] + env_response.split("</hint>")[1]
                                # this hint is on a different step because we give it during when ABBEL has its belief generation step.
                                hint = "It is your last step." if infos[i]['steps_remaining'] == 1 else f"You have {infos[i]['steps_remaining'] - 1} steps remaining."
                                env_response += "\nRemember if it is your last step you must answer. " + hint
                            new_action_messages = [{'role': "user", 'content': env_response}]
                            postprocess_text_obs.append(new_action_messages)
                        else:
                            assert valids[i], f"must be valid, but got {valids[i]=}"
                            # you can only be generating an action after a successful belief generation with the first prompt being a special case in this repo.
                            belief = actions_or_beliefs[i]
                            self.prior_beliefs[i] = belief
                            self.prior_belief_messages[i] = None
                            if self.config.env.force_full_step_len:
                                agent_first_message = NQHOTPOTQA_FULL_AGENT_FIRST_MESSAGE.format_map({"question": self.questions[i]})
                            else:
                                agent_first_message = NQHOTPOTQA_AGENT_FIRST_MESSAGE.format_map({"question": self.questions[i]})
                            hint = "It is your last step." if infos[i]['is_last_step'] else f"You have {infos[i]['steps_remaining']} steps remaining."
                            if self.config.env.prompt_info_type == 1 and self.prior_action[i] is not None:
                                # , prior_obs=self.prior_obs[i]
                                new_action_messages = [{'role':'user', 'content': NQHOTPOTQA_ACTION_PROMPT_TYPE_1.format(agent_first_message=agent_first_message, prior_action=self.prior_action[i], belief_state=belief, hint=hint)}]
                            elif self.config.env.prompt_info_type == 2 and self.prior_action[i] is not None:
                                new_action_messages = [{'role':'user', 'content': NQHOTPOTQA_ACTION_PROMPT_TYPE_2.format(agent_first_message=agent_first_message, prior_action=self.prior_action[i], prior_env_response=self.prior_obs[i], belief_state=belief, hint=hint)}]
                            else:
                                new_action_messages = [{'role':'user', 'content': (NQHOTPOTQA_FULL_ACTION_PROMPT if self.config.env.force_full_step_len else NQHOTPOTQA_ACTION_PROMPT).format(agent_first_message=agent_first_message, belief_state=belief, hint=hint)}]

                            if self.config.actor_rollout_ref.rollout.single_context:
                                new_action_messages = [{'role': "user", 'content': NQHOTPOTQA_ACTION_PROMPT_SINGLE_CONTEXT}]
                            postprocess_text_obs.append(new_action_messages)
        return postprocess_text_obs
    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Evaluate if the episodes are successful or not. 
        (Default) implementation is to check info['won'] of the last step.
        
        Returns:
        - success (np.ndarray or torch.Tensor): 1 if the episode is successful, 0 otherwise.
        """
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        batch_size = len(total_batch_list)
        
        success = defaultdict(list)
        
        for bs in range(batch_size):
            self._process_batch(bs, total_batch_list, total_infos, success)
        
        assert len(success['success_rate']) == batch_size

        return {key: np.array(value) for key, value in success.items()} | {"action_generation_failures_success_rate": np.array([self.action_generation_failures/batch_size]*batch_size), 
                                                                           "belief_generation_failures_success_rate": np.array([self.belief_generation_failures/batch_size]*batch_size),
                                                                           "successful_searches_success_rate": np.array([self.successful_searches/batch_size]*batch_size)} # just for the metric calc to be the same.

class ColabBenchEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = []
        super().__init__(envs, projection_f, config)

    def reset(self):
        text_obs, infos = self.envs.reset()
        self.task_desciptions = text_obs
        # text_obs is the questions in string form, already trimmed, should put the prompt text around this now.

        # self.supervisors = [info['supervisor'] for info in infos]
        self.memory = []
        self.action_or_belief = np.zeros(len(text_obs), dtype=np.bool) # 0 means actions being generated, 1 means beliefs
        self.prior_beliefs = [None] * len(text_obs)
        self.prior_belief_messages: List = [None] * len(text_obs) # this for tracking when a belief is not being generated correctly.
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs
        self.belief_generation_failures = 0
        self.action_generation_failures = 0
        self.successful_searches = 0
        is_not_processing = np.zeros(len(text_obs), dtype=np.bool)

        chat = self._build_chat_obs(text_obs, None, None, None, None, None, None, None, is_not_processing, None, init=True)
        return {'text': ['']*len(chat), 'image': None, 'anchor': text_obs, 'chat': chat}, infos

    def get_belief_from_output_text(self, text_beliefs_raw: List[str]):
        tags, action_or_belief_texts, valids = self.projection_f(text_beliefs_raw, np.ones(len(text_beliefs_raw)))
        return action_or_belief_texts, valids

    def step(self, text_actions: List[str], is_not_processing, tokenizer):
        # generate_belief = bool(self.step_idx % 2 != 0) This doesn't determine whether belief is generated or not lol. 
        # belief determination is a property of the environment tho, so should maintain belief or not as an array, 
        # where I can change belief or not depending on if a valid action is passed in either case.
        tags, action_or_belief_texts, valids = self.projection_f(text_actions, self.action_or_belief)
        skip_sampling_mdp = self.action_or_belief | ~np.array(valids, dtype=bool) | is_not_processing
        actions = list(zip(tags, action_or_belief_texts, skip_sampling_mdp, skip_sampling_mdp*False))
        text_obs, rewards, dones, infos = self.envs.step(actions) 
        dones = np.logical_or(dones, np.logical_not(valids)) # we inheret the if invalid just terminate logic, and this helped us a lot.
        
        # ok, converting the nqhotpotqa code to colabbench. Colab bench should be much simpler. 
        # Does it need to support mem1? (I'll say yes so long as its easy for now.)
        # Should really put the logic handling the single context and belief stuff in a general parent class.
        for i, (action, content) in enumerate(zip(tags, action_or_belief_texts)):
            # update the info dict with information on whether it was a action or belief being generated or not
            infos[i]["action_or_belief"] = int(self.action_or_belief[i])
            infos[i]['is_action_valid'] = int(valids[i])


        new_action_or_belief = ~(self.action_or_belief & np.array(valids, dtype=np.bool)) # you go to belief state unless you generate a valid belief while in belief state.
        if self.config.actor_rollout_ref.rollout.single_context and not self.config.actor_rollout_ref.rollout.belief_multiple_messages:
            new_action_or_belief = self.action_or_belief * 0
        if self.config.env.full_history_belief:
            new_action_or_belief = self.action_or_belief * 0
            # and we populate the belief for a new action prompt with the ground truth belief of our choice. this is done in the build_chat_obs
        if self.config.env.is_mem1:
            new_action_or_belief = self.action_or_belief * 0 # we handle the logic for mem1 with only actions even though we want beliefs to be generated as well.
            # if invalid in mem1, it just terminates.
            is_not_processing = ~np.array(valids, dtype=np.bool) | is_not_processing
            dones = [(d or not v) for d, v in zip(dones, valids)]
        # not sure, but the logic for the below to metrics might be wrong for mem1. not relevant enough to fix.
        self.belief_generation_failures += (self.action_or_belief & ~np.array(valids, dtype=np.bool) & ~is_not_processing).sum()
        self.action_generation_failures += (~self.action_or_belief & ~np.array(valids, dtype=np.bool) & ~is_not_processing).sum()
        self.memory.append(deepcopy((text_obs, rewards, dones, infos, new_action_or_belief, valids, action_or_belief_texts)))
        self.pre_text_obs = text_obs

        # full_text_obs = self.build_text_obs(text_obs)
        chat = self._build_chat_obs(text_obs, text_actions, tags, infos, action_or_belief_texts, valids, self.action_or_belief, new_action_or_belief, is_not_processing, tokenizer)
        

        beliefs = [belief if valid and tag == 'belief' else "" for i, (tag, belief, valid) in enumerate(zip(tags, action_or_belief_texts, valids))]
        actions = [action if valid and (tag == 'search' or tag == "answer") else "" for i, (tag, action, valid) in enumerate(zip(tags, action_or_belief_texts, valids))]
        self.action_or_belief = new_action_or_belief
        next_observations = {'text': ['']*len(chat), "filtered_belief_generations": beliefs, "filtered_action_generations": actions, 'chat': chat, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def _build_chat_obs(self, text_obs, full_text_actions, tags, infos, actions_or_beliefs, valids, old_action_or_belief, new_action_or_belief, is_not_processing, tokenizer, init: bool = False)-> List[Dict[str,str]]:
        postprocess_text_obs = []
        # valids wil be none, and actions will also be none, so I just give the chat history which is default for every first interaction.
        for i in range(len(text_obs)):
            if is_not_processing[i]:
                postprocess_text_obs.append([{'role': "user", 'content': "yo"}]) # this will be replaced at the rollout_loop.py level, and this if statement is just here to filter from going through the logic needlessly
                continue
            if init:
                # if self.config.env.is_mem1:
                #     postprocess_text_obs.append([{'role': "user", 'content': get_COLABBENCH_AGENT_FIRST_MESSAGE_MEM1(self.task_desciptions[i], self.config.actor_rollout_ref.rollout.instruct)}])
                #     # need to add the generation template manually, because it was done wrong in MEM1.
                # else:
                
                postprocess_text_obs.append([{'role': "system", 'content': COLABBENCH_AGENT_FIRST_MESSAGE.format(max_attempts=self.config.env.max_attempts)},
                                            {'role': "user", 'content': self.task_desciptions[i]}])
            else:
                # list of 0 or 1 indicating action or belief being generated.
                # this is updated right before this function call, 
                # so the chat should be preparing the lm call to generate the indicated item.
                # self.action_or_belief 
                # self.memory
                if new_action_or_belief[i]:
                    # this is belief generation prep
                    # system prompt automatically added when there is no spec.
                    if len(self.memory) == 1:
                        # first belief generation message.
                        prior_belief = COLABBENCH_NO_PRIOR_BELIEF_MESSAGE
                    else:
                        prior_belief = self.prior_beliefs[i]
                    # we may have just come from a long string of belief generation failures, 
                    # so we need to reconstruct the history of failures if this is the case.
                    if old_action_or_belief[i]:
                        # we were previously generating a belief, 
                        # and are still generating a belief, in this case, 
                        # we need to correct some error in the belief generation.
                        new_belief_messages = deepcopy(self.prior_belief_messages[i])
                        # just regenerate, you don't want to throw anything away, and you didn't do it right, 
                        # so yeah, it might happen infinitely whatever. with temp 1 in our training and test setting, should be fine.
                        # new_belief_messages += [{'role': "assistant", 'content': full_text_actions[i]},
                        #                         {'role': 'user', "content": COLABBENCH_BELIEF_GENERATION_FAILURE_MSG}]
                        if self.config.actor_rollout_ref.rollout.single_context:
                            new_belief_messages = [{'role': 'user', "content": COLABBENCH_BELIEF_GENERATION_FAILURE_MSG}]
                    else:
                        # we are for the first time generating a belief message
                        if valids[i]: 
                            agent_action = actions_or_beliefs[i]
                            env_response = text_obs[i]
                        else:
                            agent_action = "invalid action"
                            env_response = COLABBENCH_ENV_RESPONSE

                        agent_first_message = COLABBENCH_AGENT_FIRST_MESSAGE.format(max_attempts=self.config.env.max_attempts)
                        new_belief_messages = [{'role': "user", 'content': COLABBENCH_BELIEF_PROMPT.format(agent_first_message=agent_first_message,
                                                                                                           first_user_query=self.task_desciptions[i],
                                                                                                           belief_state=prior_belief,
                                                                                                           agent_action=agent_action,
                                                                                                           env_response=env_response)}]
                        if self.config.actor_rollout_ref.rollout.single_context:
                            new_belief_messages = [{'role': "user", 'content': env_response}]
                            if self.config.actor_rollout_ref.rollout.belief_multiple_messages:
                                new_belief_messages += [{'role': 'user', 'content': COLABBENCH_BELIEF_PROMPT_SINGLE_CONTEXT}]
                        # prior_belief = self.prior_beliefs[i]
                    self.prior_belief_messages[i] = new_belief_messages
                    postprocess_text_obs.append(new_belief_messages)
                else:
                    # this is action generation prep
                    # if self.config.env.is_mem1:
                    #     # you can only be generating an action after a successful belief generation with the first prompt being a special case in this repo.
                    #     # this extraction strategy is taken from mem1
                    #     belief = "<think>" + full_text_actions[i].split('<think>')[1] if '<think>' in full_text_actions[i] else full_text_actions[i]
                        
                    #     new_action_messages = [{'role': "user", 'content': (get_COLABBENCH_FULL_AGENT_FIRST_MESSAGE_MEM1 if self.config.env.force_full_step_len else get_COLABBENCH_AGENT_FIRST_MESSAGE_MEM1)(self.task_desciptions[i], self.config.actor_rollout_ref.rollout.instruct)},
                    #                            {'role': 'assistant', "content": belief},
                    #                            {'role': 'user', 'content': text_obs[i]}]
                    #     postprocess_text_obs.append(new_action_messages)
                    # else:
                    if self.config.env.full_history_belief:
                        env_response = text_obs[i]
                        if len(self.memory) == 1:
                            prior_belief = COLABBENCH_NO_PRIOR_BELIEF_MESSAGE
                        else:
                            prior_belief = self.prior_beliefs[i]
                        # from when I only wanted the action saved in the history
                        # action = actions_or_beliefs[i]
                        # belief = prior_belief + "\n<ask>" + action.strip() + "</ask>\n<environment>" + env_response.strip() + "</environment>"
                        # to now having the thinking also saved along with the action
                        action = full_text_actions[i]
                        belief = prior_belief + "\n" + action.strip() + "\n<environment>" + env_response.strip() + "</environment>"
                        agent_first_message = COLABBENCH_AGENT_FIRST_MESSAGE.format(max_attempts=self.config.env.max_attempts)
                        hint = "It is your last step." if infos[i]['is_last_step'] else f"You have {infos[i]['steps_remaining']} steps remaining."
                        new_action_messages = [{'role':'user', 'content': COLABBENCH_ACTION_PROMPT.format(agent_first_message=agent_first_message, first_user_query=self.task_desciptions[i], belief_state=belief, hint=hint)}]
                        self.prior_beliefs[i] = belief
                        postprocess_text_obs.append(new_action_messages)
                    elif self.config.actor_rollout_ref.rollout.single_context and not self.config.actor_rollout_ref.rollout.belief_multiple_messages:
                        env_response = text_obs[i]
                        hint = "It is your last step." if infos[i]['is_last_step'] else f"You have {infos[i]['steps_remaining']} steps remaining."
                        new_action_messages = [{'role': "user", 'content': env_response + "\n" + COLABBENCH_TURNS_REMAINING_HINT.format(hint=hint).strip()}]
                        postprocess_text_obs.append(new_action_messages)
                    else:
                        assert valids[i], f"must be valid, but got {valids[i]=}"
                        # you can only be generating an action after a successful belief generation with the first prompt being a special case in this repo.
                        belief = actions_or_beliefs[i]
                        self.prior_beliefs[i] = belief
                        self.prior_belief_messages[i] = None
                        agent_first_message = COLABBENCH_AGENT_FIRST_MESSAGE.format(max_attempts=self.config.env.max_attempts)
                        hint = "It is your last step." if infos[i]['is_last_step'] else f"You have {infos[i]['steps_remaining']} steps remaining."
                        new_action_messages = [{'role':'user', 'content': COLABBENCH_ACTION_PROMPT.format(agent_first_message=agent_first_message, first_user_query=self.task_desciptions[i], belief_state=belief, hint=hint)}]
                        if self.config.actor_rollout_ref.rollout.single_context:
                            new_action_messages = [{'role': "user", 'content': COLABBENCH_ACTION_PROMPT_SINGLE_CONTEXT + COLABBENCH_TURNS_REMAINING_HINT.format(hint=hint)}]
                        postprocess_text_obs.append(new_action_messages)
        return postprocess_text_obs
    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Evaluate if the episodes are successful or not. 
        (Default) implementation is to check info['won'] of the last step.
        
        Returns:
        - success (np.ndarray or torch.Tensor): 1 if the episode is successful, 0 otherwise.
        """
        total_infos = kwargs['total_infos']
        total_batch_list = kwargs['total_batch_list']
        batch_size = len(total_batch_list)
        
        success = defaultdict(list)
        
        for bs in range(batch_size):
            self._process_batch(bs, total_batch_list, total_infos, success)
        
        assert len(success['success_rate']) == batch_size

        return {key: np.array(value) for key, value in success.items()} | {"action_generation_failures_success_rate": np.array([self.action_generation_failures/batch_size]*batch_size), 
                                                                           "belief_generation_failures_success_rate": np.array([self.belief_generation_failures/batch_size]*batch_size),
                                                                           "successful_searches_success_rate": np.array([self.successful_searches/batch_size]*batch_size)} # just for the metric calc to be the same.

def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    if "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True)
        _val_envs = build_gymcards_envs(env_name=config.env.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False)
        
        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            'eval_dataset': 'eval_in_distribution', # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "combolock" in config.env.env_name.lower():
        from agent_system.environments.env_package.combolock import build_combolock_envs, combolock_projection
        _envs = build_combolock_envs(config.env.max_attempts, config.env.vocab, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, )
        _val_envs = build_combolock_envs(config.env.max_attempts, config.env.vocab, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False)

        projection_f = partial(combolock_projection, vocab=config.env.vocab)
        envs = ComboLockEnvironmentManager(_envs, projection_f, config)
        val_envs = ComboLockEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "nqhotpotqa" in config.env.env_name.lower():
        from agent_system.environments.env_package.nqhotpotqa import build_nqhotpotqa_envs, nqhotpotqa_projection
        _envs = build_nqhotpotqa_envs(split=config.env.split, num_objectives=config.env.num_objectives, force_full=config.env.force_full_step_len, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, )
        _val_envs = build_nqhotpotqa_envs(split=config.env.split,  num_objectives=config.env.num_objectives, force_full=config.env.force_full_step_len, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False)

        projection_f = partial(nqhotpotqa_projection)
        envs = NQHotpotQAEnvironmentManager(_envs, projection_f, config)
        val_envs = NQHotpotQAEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "colabbench" in config.env.env_name.lower():
        from agent_system.environments.env_package.colabbench import build_colabbench_envs, colabbench_projection
        _envs = build_colabbench_envs(split=config.env.split, hostname=config.env.hostname, port=config.env.port, model_id=config.env.model_id, task_type=config.env.task_type, max_steps=config.env.max_attempts, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, )
        _val_envs = build_colabbench_envs(split=config.env.split, hostname=config.env.hostname, port=config.env.port, model_id=config.env.model_id, task_type=config.env.task_type, max_steps=config.env.max_attempts, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False)

        projection_f = partial(colabbench_projection)
        envs = ColabBenchEnvironmentManager(_envs, projection_f, config)
        val_envs = ColabBenchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)