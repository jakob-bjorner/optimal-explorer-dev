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

import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from verl.models.transformers.qwen2_vl import get_rope_index
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict
from torch.nn.utils.rnn import pad_sequence
import tensordict as td
from copy import deepcopy, copy
import re
import ast
from itertools import product, permutations
import os
import asyncio
import time
import math
from collections import defaultdict
from operator import itemgetter

class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.maxes = None

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        if "chat" in obs:
            chat = np.array(obs['chat'][item])
        else:
            chat = np.array([{
                "content": obs_content,
                "role": "user",
            }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_len)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            episode_penalties: None|np.ndarray = None,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        
        Returns:
            DataProto: Collected and organized trajectory data
        """
        not_is_belief_grading_context_list = [not trajectory_list[0]['info'].get('is_belief_grading_context', False) for trajectory_list in total_batch_list]
        batch_size = len(total_batch_list)

        episode_rewards_mean = np.mean(episode_rewards[not_is_belief_grading_context_list])
        episode_rewards_min = np.min(episode_rewards[not_is_belief_grading_context_list])
        episode_rewards_max = np.max(episode_rewards[not_is_belief_grading_context_list])
        
        belief_episode_rewards_mean = np.mean(episode_rewards[np.logical_not(not_is_belief_grading_context_list)])
        belief_episode_length_mean = np.mean([trajectory_list[0].get('filtered_belief_generations_len', 0) for not_is_belief_grading_context, trajectory_list in zip(not_is_belief_grading_context_list, total_batch_list) if not not_is_belief_grading_context])
        if np.isnan(belief_episode_length_mean): #
            belief_lengths = [context.get('filtered_belief_generations_len', 0) for trajectory_list in total_batch_list for context in trajectory_list if context.get('filtered_belief_generations_len', 0) != 0]
            belief_episode_length_mean = np.mean(belief_lengths)
        episode_lengths_mean = np.mean(episode_lengths[not_is_belief_grading_context_list])
        episode_lengths_min = np.min(episode_lengths[not_is_belief_grading_context_list])
        episode_lengths_max = np.max(episode_lengths[not_is_belief_grading_context_list])

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    # if not_is_belief_grading_context_list[bs]:
                    data['episode_rewards'] = episode_rewards[bs]
                    if episode_penalties is not None:
                        data['episode_penalties'] = episode_penalties[bs]
                    data['episode_rewards_mean'] = episode_rewards_mean
                    data['episode_rewards_min'] = episode_rewards_min
                    data['episode_rewards_max'] = episode_rewards_max

                    data['belief_episode_rewards_mean'] = belief_episode_rewards_mean
                    data['belief_episode_length_mean'] = belief_episode_length_mean
                    # episode_lengths
                    # if not_is_belief_grading_context_list[bs]:
                    data['episode_lengths'] = episode_lengths[bs]
                    data['episode_lengths_mean'] = episode_lengths_mean
                    data['episode_lengths_min'] = episode_lengths_min
                    data['episode_lengths_max'] = episode_lengths_max
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """
        start_time = time.time()
        print(start_time)
        # Initial observations from the environment
        obs, infos = envs.reset()

        # Initialize trajectory collection
        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        if len(gen_batch.batch) != lenght_obs and self.config.env.rollout.n > 0:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"

        batch_size = len(gen_batch.batch['input_ids'])
        batch_output = None
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        prompt_too_long = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        belief_lengths = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.int32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        if self.maxes is None:
            self.maxes = [-5] * batch_size

        # need a completely different loop to handle single context stuff, because else it will be really messy. Almost none will be shared I think.
        if self.config.actor_rollout_ref.actor.single_context:
            episode_penalties = None
            messages_list: list[list[dict[str, str]]] = [[] for _ in range(batch_size)] # this will contain full message history
            input_ids_list: list[list[int]] = [[] for _ in range(batch_size)] # this will be the list of input_ids which I feed into the vllm loop every time. 
            attention_mask_list: list[list[int]] = [[] for _ in range(batch_size)]
            # position_ids_list: list[list[int]] = [[] for _ in range(batch_size)]
            loss_mask_list: list[list[int]] = [[] for _ in range(batch_size)]
            for i in range(len(obs['chat'])):
                starting_chat: list[dict[str, str]] = obs['chat'][i]
                messages_list[i].extend(starting_chat)

                starting_chat_input_ids: list[int] = self.tokenizer.apply_chat_template(starting_chat, add_generation_prompt=False, tokenize=True)
                input_ids_list[i].extend(starting_chat_input_ids)
                attention_mask_list[i].extend([1] * len(starting_chat_input_ids))
                # note, loss_mask_list is empty because the prompt and response input_ids are segmented.

            BASE_CHAT_HISTORY = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "I am a user."}]
            base_conv_wo_gen_prompt_end_pos = len(self.tokenizer.apply_chat_template(BASE_CHAT_HISTORY, add_generation_prompt=False, tokenize=False))
            generation_prompt_ids: list[int] = self.tokenizer.apply_chat_template(BASE_CHAT_HISTORY, add_generation_prompt=True, tokenize=True)[len(self.tokenizer.apply_chat_template(BASE_CHAT_HISTORY, add_generation_prompt=False, tokenize=True)):]
            def _update_input_ids(new_input_ids: List[int], index: int, should_add_loss_mask: bool) -> None:
                input_ids: list[int] = input_ids_list[index]
                attention_mask: list[int] = attention_mask_list[index]
                # position_ids: list[int] = position_ids_list[index]
                loss_mask: list[int] = loss_mask_list[index]

                input_ids += new_input_ids
                attn_mask = [1] * len(new_input_ids)
                attention_mask += attn_mask
                loss_mask += [int(should_add_loss_mask)] * len(new_input_ids)
                # position_ids += (compute_position_id_with_mask(torch.tensor(attn_mask)) + (position_ids[-1] + 1)).tolist()
                assert len(input_ids) == len(attention_mask), f"""context has different length of {len(input_ids)=}, {len(attention_mask)=},"""
            def add_assistant_message(
                index: int,
                content_str: str,
                content_ids: list[int],
                should_add_loss_mask: bool = True,
            ) -> None:
                content_str = content_str.replace("<|im_end|>", "")
                messages_list[index].append(dict(role="assistant", content=content_str))
                _update_input_ids(content_ids, index, should_add_loss_mask=should_add_loss_mask)
                if content_ids[-1] != self.tokenizer.encode("<|im_end|>", add_special_tokens=False)[0]:
                    _update_input_ids(self.tokenizer.encode("<|im_end|>", add_special_tokens=False), index, should_add_loss_mask=False)

                _update_input_ids(self.tokenizer.encode("\n", add_special_tokens=False), index, should_add_loss_mask=False)
            def add_user_messages(
                index: int,
                new_messages: list[dict[str, str]], 
            ) -> None:
                new_messages = deepcopy(new_messages)
                messages_list[index].extend(new_messages)
                content_str = self.tokenizer.apply_chat_template([*BASE_CHAT_HISTORY, *new_messages], add_generation_prompt=False, tokenize=False)
                content_ids = self.tokenizer.encode(content_str[base_conv_wo_gen_prompt_end_pos:], add_special_tokens=False)
                _update_input_ids(content_ids, index, should_add_loss_mask=False)
            def add_generation_prompt(
                index: int,
            ):
                temp_generation_prompt_ids = generation_prompt_ids
                cur_generation_prompt_ids = [] if input_ids_list[index][-len(temp_generation_prompt_ids):] == temp_generation_prompt_ids else temp_generation_prompt_ids
                if cur_generation_prompt_ids:
                    _update_input_ids(cur_generation_prompt_ids, index, should_add_loss_mask=False)
            def prepare_data_for_data_proto(input_ids_list: list[list[int]], attention_mask_list: list[list[int]]):
                input_ids = pad_sequence([torch.tensor(t) for t in input_ids_list], batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side='left')
                attention_mask = pad_sequence([torch.tensor(t) for t in attention_mask_list], batch_first=True, padding_value=0, padding_side='left')
                position_ids = compute_position_id_with_mask(attention_mask)
                return input_ids, attention_mask, position_ids
            prompt_input_ids_list = deepcopy(input_ids_list)
            small_input_ids: list[int] = self.tokenizer.apply_chat_template([{'role':"user", 'content': "What is 2 + 2? Please answer quickly."}], add_generation_prompt=True, tokenize=True)
            small_attention_mask = [1] * len(small_input_ids)

            for _step in range(self.config.env.max_steps):
                active_masks = np.logical_not(is_done)
                # need to construct ["input_ids", "attention_mask", "position_ids"] in every loop
                # then put them in batched_input object to feed to 
                # batch_input = None
                # batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs) # need to do the logic of creating the next batch_input object, which should contain the tensors padded to length for all conversations.
                trimmed_input_ids_list = list(input_ids_list)
                trimmed_attention_mask_list = list(attention_mask_list)
                for i in range(batch_size):
                    if is_done[i]:
                        trimmed_input_ids_list[i] = small_input_ids
                        trimmed_attention_mask_list[i] = small_attention_mask
                    else:
                        add_generation_prompt(i)
                        
                # input_ids = pad_sequence([torch.tensor(t) for t in trimmed_input_ids_list], batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side='left')
                # attention_mask = pad_sequence([torch.tensor(t) for t in trimmed_attention_mask_list], batch_first=True, padding_value=0, padding_side='left')
                # position_ids = compute_position_id_with_mask(attention_mask)
                input_ids, attention_mask, position_ids = prepare_data_for_data_proto(trimmed_input_ids_list, trimmed_attention_mask_list)
                batch_input = DataProto(
                    td.TensorDict(dict(
                        input_ids = input_ids,
                        attention_mask = attention_mask,
                        position_ids = position_ids,
                    ), batch_size=batch_size),
                    meta_info=gen_batch.meta_info, # I missing data_source and index fields. Not sure if important.
                )

                batch_output = actor_rollout_wg.generate_sequences(batch_input)

                batch_input.non_tensor_batch['uid'] = uid_batch
                batch_input.non_tensor_batch['traj_uid'] = traj_uid
                batch_input.pop(batch_keys=["attention_mask", 'input_ids', "position_ids"])
                batch = batch_input.union(batch_output)

                text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
                next_obs, rewards, dones, infos = envs.step(text_actions, is_done, self.tokenizer)
                if len(rewards.shape) == 2:
                    rewards = rewards.squeeze(1)
                if len(dones.shape) == 2:
                    # dones is numpy, delete a dimension
                    dones = dones.squeeze(1)

                # create a temp save of each of these, 
                # just in case we create a messsage history which is clearly too long, 
                # and have to restore it. 
                # This is in place of implementing an undo for the add user and add assistant message operations.
                # temp_messages_list = deepcopy(messages_list)
                # temp_input_ids_list = deepcopy(input_ids_list)
                # temp_attention_mask_list = deepcopy(attention_mask_list)
                # temp_loss_mask_list = deepcopy(loss_mask_list)
                for i in range(batch_size): 
                    if is_done[i]:
                        continue
                    response_len = len(batch.batch['responses'][0])
                    add_assistant_message(i, text_actions[i], batch.batch['responses'][i][batch.batch['attention_mask'][i][-response_len:] == 1], should_add_loss_mask=True)
                    add_user_messages(i, next_obs['chat'][i])
                    # if the assistant message is too long, I shouldn't really restore it, 
                    # I should rather clip it to length just in case the sequence is too long,
                    # and we want to apply negative penalty to what was just generated. 
                    # I should note that this part of the implementation is somewhat controversal 
                    # because of DAPO's findings that you should to ignore overlong generations
                    # but it feels right to do here.
                    if len(input_ids_list[i]) >= self.config.data.max_prompt_length:
                        prompt_too_long[i] = (True)
                        # and need to prune, but the messages_list can stay unpruned for book keeping.
                        input_ids_list[i] = input_ids_list[i][:self.config.data.max_prompt_length]
                        attention_mask_list[i] = attention_mask_list[i][:self.config.data.max_prompt_length]
                        loss_mask_list[i] = loss_mask_list[i][:self.config.data.max_prompt_length - len(prompt_input_ids_list[i])]
                # need to remember to not reward the sequences which are too long even if they just got a reward.

                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)

                # Create reward tensor, only assign rewards for active environments
                episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks) * np.logical_not(prompt_too_long)
                episode_lengths[active_masks] += 1

                assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
                
                # Update done states
                is_done = np.logical_or(is_done, dones)
                is_done = np.logical_or(is_done, prompt_too_long) # need this updated so the active masks are updated, which is used for getting the loss tokens
                
                obs = next_obs

                # Break if all environments are done
                if is_done.all():
                    break

            prompt_input_ids = pad_sequence([torch.tensor(t) for t in prompt_input_ids_list], batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side='left')
            # do I need to pad to prompt length? I don't think I do, so I won't If something breaks that would suck.

            prompt_attention_mask = pad_sequence([torch.tensor([1] * len(t)) for t in prompt_input_ids_list], batch_first=True, padding_value=0, padding_side='left')
            response_input_ids = pad_sequence([torch.tensor(in_ids[len(p_ids):]) for p_ids, in_ids in zip(prompt_input_ids_list, input_ids_list)], batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side='right')
            response_mask = pad_sequence([torch.tensor(in_mask[len(p_ids):]) for p_ids, in_mask in zip(prompt_input_ids_list, attention_mask_list)], batch_first=True, padding_value=0, padding_side='right')
            
            complete_input_ids = torch.concat([prompt_input_ids, response_input_ids], dim=-1)
            complete_attention_mask = torch.concat([prompt_attention_mask, response_mask], dim=-1)

            # for i in range(batch_size):
            #     ...
            # I need to separate the prompt_ids and the response_ids, and make a tensor just for the responses
            batch.batch['responses'] = response_input_ids
            batch.batch['input_ids'] = complete_input_ids 
            batch.batch['attention_mask'] = complete_attention_mask
            batch.batch['position_ids'] = compute_position_id_with_mask(complete_attention_mask)
            loss_mask = pad_sequence([torch.tensor(t) for t in loss_mask_list], batch_first=True, padding_value=0, padding_side='right')
            batch.batch['loss_mask'] = loss_mask
            batch.pop(batch_keys=["rollout_log_probs"])
            batch.non_tensor_batch['data_source'] = gen_batch.non_tensor_batch['data_source']
            batch.non_tensor_batch['active_masks'] = np.ones(batch_size, dtype=bool) # here we concat the input_ids every time, so all batch elements are always active.
            batch.non_tensor_batch["info"] = infos
            # batch.non_tensor_batch['messages'] = np.array(messages_list)
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])
            # total_batch_list = to_list_of_dict(batch)
            # total_infos = infos
        else:
            # Trajectory collection loop
            for _step in range(self.config.env.max_steps):
                active_masks = np.logical_not(is_done)

                batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                batch_input = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                batch_input.meta_info = gen_batch.meta_info
                input_ids_str = self.tokenizer.batch_decode(batch_input.batch['input_ids'], skip_special_tokens=True)
                batch_output = actor_rollout_wg.generate_sequences(batch_input)



                batch.non_tensor_batch['uid'] = uid_batch
                batch.non_tensor_batch['traj_uid'] = traj_uid

                batch = batch.union(batch_output)
                
                text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)

                next_obs, rewards, dones, infos = envs.step(text_actions, is_done, self.tokenizer)

                if len(rewards.shape) == 2:
                    rewards = rewards.squeeze(1)
                if len(dones.shape) == 2:
                    # dones is numpy, delete a dimension
                    dones = dones.squeeze(1)

                
                if 'chat' in next_obs:
                    for i in range(batch_size):# this is specific to combo lock... other envs don't have chat.
                        input_ids = self.tokenizer.apply_chat_template(
                            next_obs['chat'][i],
                            add_generation_prompt=True,
                            tokenize=True
                        )
                        if not (is_done[i] or dones[i]) and len(input_ids) >= self.config.data.max_prompt_length:
                            prompt_too_long[i] = True


                if 'is_action_valid' in infos[0]:
                    batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
                else:
                    batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

                # Create reward tensor, only assign rewards for active environments
                episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
                episode_lengths[active_masks] += 1

                assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
                batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
                batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
                if 'chat' in next_obs:
                    batch.non_tensor_batch['response_ids_str'] = text_actions
                    batch.non_tensor_batch["response_ids_token_len"] = (self.tokenizer.pad_token_id != batch.batch['responses']).sum(-1).cpu().numpy()
                    batch.non_tensor_batch['input_ids_str'] = input_ids_str
                    batch.non_tensor_batch["input_ids_token_len"] = (self.tokenizer.pad_token_id != batch.batch['input_ids']).sum(-1).cpu().numpy()
                if "filtered_belief_generations" in next_obs:
                    # needed to eventually calculate belief len data per turn easier. 
                    # This will be used in conjunction with the response_ids, which can give the thinking and answer/search tags
                    batch.non_tensor_batch['filtered_belief_generations'] = next_obs['filtered_belief_generations']
                    new_belief_lengths = [len(bt) for bt in self.tokenizer(next_obs['filtered_belief_generations']).input_ids]
                    belief_lengths = [belief_len_list + [new_belief_len] for belief_len_list, new_belief_len in zip(belief_lengths,new_belief_lengths)]
                    batch.non_tensor_batch['filtered_belief_generations_len'] = np.array(new_belief_lengths)
                    batch.non_tensor_batch['filtered_action_generations'] = next_obs['filtered_action_generations'] 
                batch.non_tensor_batch['step'] = np.ones(batch_size, dtype=int) * _step
                batch.non_tensor_batch["info"] = infos

                # Update episode lengths for active environments
                batch_list: list[dict] = to_list_of_dict(batch)

                for i in range(batch_size):
                    total_batch_list[i].append(batch_list[i])
                    total_infos[i].append(infos[i])

                # Update done states

                is_done = np.logical_or(is_done, dones)
                is_done = np.logical_or(is_done, prompt_too_long) # need this updated so the active masks are updated, which is used for getting the loss tokens
                
                # Update observations for next step
                if 'chat' in next_obs:
                    new_next_obs_chat = []
                    for i in range(batch_size):
                        if is_done[i]:
                            new_next_obs_chat.append([{'role':"user", 'content': "What is 2 + 2? Please answer quickly."}])
                        else:
                            new_next_obs_chat.append(next_obs['chat'][i])
                    next_obs['chat'] = new_next_obs_chat

                obs = next_obs

                # Break if all environments are done
                if is_done.all():
                    break
            # if they don't terminate, we give a -1 reward in the combolock setting.
        
        print(time.time() - start_time)
        # should look at how many tokens were generated and how many were used as prompt tokens. I guess this is documented.
        if self.config.env.non_terminal_penalty:
            episode_rewards[np.logical_not(is_done) | prompt_too_long] = -self.config.env.non_terminal_penalty
        # does episode reward not count towards GRPO? No it does, funny enough, the reward thing I think we record per step isn't used tho. seems just for logging.
        # breakpoint()
        # need to consider if this if statement can be commented out...
        # if self.config.env.belief_length_penalty: # 0.1
        # only want to further penalize the runs which did terminate, and terminated with some correct output. to make them correct and smaller.
        # I think this could lead to reward hacking if the model just records a single objective instead, or just focuses on a single objective.
        # is the idea make them all smaller? or just make them smaller than a particular size?
        max_belief_lengths = np.array([max(belief_lens + [0]) for belief_lens in belief_lengths])
        # if max_belief_lengths.max() != max_belief_lengths.min():
        belief_penalties = (max_belief_lengths - max_belief_lengths.mean())
        # belief_penalties = max_belief_lengths
        belief_penalties[max_belief_lengths == 0] = 0
        # belief_penalties = belief_penalties / (belief_penalties.max() - belief_penalties.min()) rm normalization
        episode_penalties = np.zeros_like(episode_rewards)
        episode_penalties[episode_rewards > 0] = belief_penalties[episode_rewards > 0]
        # the below line is soft depricated. Should instead use trainer.post_normalization_length_penalty
        episode_rewards[episode_rewards > 0] += -self.config.env.belief_length_penalty * belief_penalties[episode_rewards > 0]
        # we want to reward the sequences
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    )
        success["non_terminal_trajectories_success_rate"] = (np.logical_not(is_done) | prompt_too_long).mean(keepdims=True) # this should be prompt too long
        # add metrics as KV pairs to the success dict.
        
        # second RL after the RL steps. this could be done every few steps or something more heuristicy.
        # figure out how to separate out the belief states, 
        # so we can reprompt the model to generate more belief states, 
        # and grade the outputs of all the belief states.
        # would be nice to have a simple setting for belief state grading,
        # but for now dealing with the full setting and see if it just works. 
        # not sure how to make a simpler setting easily out of combination lock... 
        # Perhaps only 3 vocab tokens to guess, so make sure that the belief 
        # contains all the digits, like a string matching thing.  (This will be backup plan. )
        
        # implementing the belief grade call notes:
        # total_batch_list is a list length 4 * 2 for effective batch size. each containing a list of dicts which correspond to the contexts.
        # they seem to have the batch and non_tensor_info in a flat dict, so can just loop through here and look for active belief (ie not ['info']['action_or_belief']])
        # lets just assume I can do this right now, and see what I'd need to do to generatemore sequences from here.
        # Flag belief contexts for metrics info['is_belief_grading_context'] = True

        # after beliefs are generated, use this function to parse them in the same manor as is done in the environment 
        # function use self.envs.get_belief_from_output_text(self, text_beliefs_raw: List[str]):
        # ensure these are all true, before you pass back the stats.
        # pad total_episode_rewards and total_episode_lengths use zero or whatever doesn't matter.
        # populate total_traj_uid with new uids total_traj_uid
        # add belief_grading_contexts to total_batch_list, ensure they have info flag. 
        # GRPO should handle them nicely. 
        # Ensure the uid and traj_uid are new, and don't conflict with the prior ones.

        # Try this function before returning 
        # self.gather_rollout_data(
        #     total_batch_list=total_batch_list,
        #     episode_rewards=total_episode_rewards,
        #     episode_lengths=total_episode_lengths,
        #     success=total_success,
        #     traj_uid=total_traj_uid,
        # )

        # before restarting the session to get the action_or_belief information, I want to check that generation works nicely on this single GPU setting.
        # keys_for_generation = ["input_ids", "attention_mask", "position_ids", "raw_prompt"]
        # gen_for_belief_grading = DataProto.from_single_dict(data=collate_fn([{k: e[k] for k in keys_for_generation} for e in [total_batch_list[0][0], total_batch_list[0][3], total_batch_list[2][2]]]))
        # yup seems good, I can generate some beliefs then.
        # We are going to reward for the correct posterior, and that is it. 
        # This can encourage parsable posteriors over time, and will essentially help the model to just record all the info in a very easy mannor.
        # here
        # generate a pair for each, re label the traj_uid
        if self.config.trainer.belief_state_grading:
            flattened_valid_belief_contexts = [c for trajectory in total_batch_list for c in trajectory if (c['active_masks'] and c['info']["is_action_valid"] and c['info']["action_or_belief"] and "Belief generation failed to parse" not in c['input_ids_str'])] 
            # technically we don't have to filter on the successful beliefs, but lets do this for now.
            if len(flattened_valid_belief_contexts) >= batch_size:
                # we want to create a deepcopy of the full set of valid beleif contexts, then remove unnecessary info, and populate with new traj_uid and uid info and reshuffle into total_batch_list format.
                # take a subset divisible by batch_size just for safety.
                subset_size = (len(flattened_valid_belief_contexts) // batch_size) * batch_size

                flattened_valid_belief_contexts = deepcopy(flattened_valid_belief_contexts[:subset_size])
                # set most of these to none. there are some other things we need to do, like make sure the attention_mask, input_ids, are cut to length, and remove the response, and rollout_log_probs, and change uid to parent_uid and traj_uid to parent_traj_uid 
                # (['attention_mask', 'prompts', 'input_ids', 'responses', 'rollout_log_probs', 'position_ids', 'anchor_obs', 'index', 'data_source', 'uid', 'traj_uid', 'raw_prompt', 'is_action_valid', 'rewards', 'active_masks', 'response_ids_str', 'response_ids_token_len', 'input_ids_str', 'input_ids_token_len', 'filtered_belief_generations', 'filtered_action_generations', 'step', 'info'])
                # set to nan 'response_ids_str', 'response_ids_token_len', 'input_ids_str', 'input_ids_token_len', 'anchor_obs' => "", ("is_action_valid", 'rewards', 'active_masks', 'filtered_belief_generations', 'filtered_action_generations', 'step') will all have new values, and need to change things in info
                for belief_context_dict in flattened_valid_belief_contexts:
                    belief_context_dict['response_ids_token_len'] = np.nan
                    belief_context_dict.pop('rewards')
                    belief_context_dict['anchor_obs'] = ''
                    # belief_context_dict["input_ids"] = belief_context_dict["input_ids"][:self.config.data.max_prompt_length]
                    # belief_context_dict["attention_mask"] = belief_context_dict["attention_mask"][:self.config.data.max_prompt_length]
                    # belief_context_dict['position_ids'] = belief_context_dict['position_ids'][:self.config.data.max_prompt_length]
                    belief_context_dict['info']["parent_uid"] = belief_context_dict['uid'] # 'uid', 'traj_uid'
                    belief_context_dict['info']["parent_traj_uid"] = belief_context_dict['traj_uid'] # 'uid', 'traj_uid'
                    new_uid = str(uuid.uuid4())
                    belief_context_dict["uid"] = new_uid # want to give traj ids after, because we deepcopy these contexts to create other traj_uids.
                    belief_context_dict["traj_uid"] = '' # this should be populated after the group size is increased.
            
                keys_for_generation = ["input_ids", "attention_mask", "position_ids"]
                input_for_belief_gen = DataProto.from_single_dict(data=collate_fn([{k: e[k] for k in keys_for_generation} for e in flattened_valid_belief_contexts]))
                input_for_belief_gen.batch["input_ids"] = input_for_belief_gen.batch["input_ids"][:, :self.config.data.max_prompt_length]
                input_for_belief_gen.batch["attention_mask"] = input_for_belief_gen.batch["attention_mask"][:, :self.config.data.max_prompt_length]
                input_for_belief_gen.batch['position_ids'] = input_for_belief_gen.batch['position_ids'][:, :self.config.data.max_prompt_length]
                belief_gen_outputs = actor_rollout_wg.generate_sequences(input_for_belief_gen)
                new_belief_response_strs = self.tokenizer.batch_decode(belief_gen_outputs.batch['responses'], skip_special_tokens=True)
                new_belief_action_or_belief_texts, new_belief_valids = envs.get_belief_from_output_text(new_belief_response_strs)
                # new_belief_action_or_belief_texts, new_belief_valids = envs.projection_f(new_belief_response_strs, np.ones(len(new_belief_response_strs)))

                new_belief_contexts = deepcopy(flattened_valid_belief_contexts)
                for i, new_belief_context_dict in enumerate(new_belief_contexts):
                    for k in ["attention_mask","input_ids","position_ids","prompts","responses","rollout_log_probs"]:
                        new_belief_context_dict[k] = belief_gen_outputs.batch[k][i]
                    new_belief_context_dict['is_action_valid'] = new_belief_valids[i]
                    new_belief_context_dict['info'] = deepcopy(new_belief_context_dict['info']) # this shouldn't be necessary
                    new_belief_context_dict['info']["is_action_valid"] = int(new_belief_valids[i])
                    new_belief_context_dict['filtered_belief_generations'] = new_belief_action_or_belief_texts[i]
                    new_belief_context_dict['filtered_belief_generations_len'] = self.tokenizer.encode(new_belief_action_or_belief_texts[i]).__len__()
                    new_belief_context_dict['response_ids_str'] = new_belief_response_strs[i]
                all_belief_contexts = flattened_valid_belief_contexts + new_belief_contexts
                for c in all_belief_contexts:
                    c['info'].update({"is_belief_grading_context": True})
                if self.config.env.env_name == "combolock" and self.config.trainer.belief_state_grading_type < 0:
                    from agent_system.environments.prompts.combolock import COMBO_BELIEF_GRADING_PROMPT, COMBO_BELIEF_GRADING_PROMPT_FILLER_BELIEF
                    grading_prompts = [COMBO_BELIEF_GRADING_PROMPT.format(belief=c['filtered_belief_generations']) if c['is_action_valid'] else "" for c in all_belief_contexts] 
                    # I decide not to filter out the invalids here, and just grade everything because its less book keeping. shouldn't be too bad when the code is working well. < 1/6 beliefs seem to fail.
                    # we don't need to grade the invalid cases, just reward them -1.
                    # old_padding_side = self.tokenizer.padding_side
                    # self.tokenizer.padding_side = "left"
                    # grading_inputs = self.tokenizer(grading_prompts, return_tensors='pt', padding="max_length", max_length=self.config.data.max_prompt_length)
                    # self.tokenizer.padding_side = old_padding_side

                    # input_for_belief_grading = DataProto.from_single_dict(data={'input_ids': grading_inputs['input_ids'], "attention_mask": grading_inputs['attention_mask'], "position_ids": compute_position_id_with_mask(grading_inputs['attention_mask'])})
                    # input_for_belief_grading.meta_info['extra_sample_params'] = {'stop': ['```'], "include_stop_str_in_output": True, "detokenize": True}
                    # belief_grading_outputs = actor_rollout_wg.generate_sequences(input_for_belief_grading)
                    # belief_grading_response_strs = self.tokenizer.batch_decode(belief_grading_outputs.batch['responses'], skip_special_tokens=True)

                    from openai import AsyncOpenAI
                    from dotenv import load_dotenv
                    load_dotenv('/nas/ucb/jbjorner3/dev/optimal-explorer-dev/.env')

                    client = AsyncOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ['OPENROUTER_API_KEY'],
                    )
                    async def generate(prompt, timeout): # want something that waits for a total timeout and if it doesn't work just return an empty string.
                        if prompt:
                            for _ in range(3):
                                try:
                                    return (await asyncio.wait_for(client.chat.completions.create(
                                        extra_body={
                                            "google": {
                                                "thinking_config": {
                                                    "thinking_budget": 0
                                                }
                                            }
                                        },
                                        model="google/gemini-3-flash-preview",
                                        messages=[{"role": "user", "content":prompt}],
                                        ), timeout=timeout)).choices[0].message.content
                                except Exception as e:
                                    if isinstance(e, asyncio.TimeoutError):
                                        return ""
                                    else:
                                        await asyncio.sleep(10)
                        return ""
                    async def generate_all(prompts):
                        return await asyncio.gather(*[generate(p, 40) for p in prompts])
                    belief_grading_response_strs = asyncio.run(generate_all(grading_prompts))

                    # then we parse the strs, and get the ground truth 

                    pattern = r"in position 1: (.*)\n.*in position 2: (.*)\n.*in position 3: (.*)" # this is specific to the prompt we use, but whatever. storing it here for now.
                    program = re.compile(pattern)
                    def get_posterior_from_response_str(response_str):
                        match = program.search(response_str)
                        if match and len(match.groups()) == 3:
                            try:
                                position_possibilities = [ast.literal_eval(possibility_str) for possibility_str in match.groups()]
                                position_possibilities = [[int(s) for s in l] for l in position_possibilities]
                            except:
                                return None
                            return position_possibilities
                        else:
                            return None
                    belief_representation_extracted = list(map(get_posterior_from_response_str, belief_grading_response_strs))

                    # need to compare the belief_representations to the true beliefs that they should have after the feedback they have just been given.
                    # I'll only grade the states which are validly parsed. I'll want to record the fraction of states graded. 
                    # We know that the first states are always valid with [0-9], [0-9], [0-9] for all.
                    # will start with tree search from each starting trajectory, and append to the set that I can grade.
                    # at the end I want to have graded all the.
                    for c, extract, belief_grading_response_str in zip(all_belief_contexts, belief_representation_extracted, belief_grading_response_strs):
                        c['info'].update({"belief_representation_extracted": extract, "belief_grading_response_str": belief_grading_response_str})
                    primary_belief_contexts, secondary_belief_contexts = all_belief_contexts[:len(flattened_valid_belief_contexts)], all_belief_contexts[len(flattened_valid_belief_contexts):]
                    valid_codes = set(permutations(range(10), 3))
                    

                    def normalize_from_possibles_list(possibles_list):
                        return set(k for k in product(*possibles_list) if k in valid_codes)
                    def get_reward_from_possibles_list(possibles_list, true_belief):
                        if possibles_list is None:
                            return 0
                        else:
                            return float(true_belief == normalize_from_possibles_list(possibles_list))
                    new_total_batch_list = copy(total_batch_list)
                    new_episode_rewards = episode_rewards.tolist()
                    new_episode_lengths = episode_lengths.tolist()
                    new_traj_uid = traj_uid.tolist()
                    i = 0
                    parsable_belief_states = 0
                    while i < len(primary_belief_contexts):
                        primary_belief_context = primary_belief_contexts[i]
                        secondary_belief_context = secondary_belief_contexts[i]
                        true_belief = normalize_from_possibles_list([[int(s) for s in l] for l in primary_belief_context['info']['posterior']])
                        primary_reward = get_reward_from_possibles_list(primary_belief_context['info']['belief_representation_extracted'], true_belief)
                        secondary_reward = 0.0 if not secondary_belief_context['is_action_valid'] else get_reward_from_possibles_list(secondary_belief_context['info']['belief_representation_extracted'], true_belief)
                        primary_traj_uid = str(uuid.uuid4())
                        secondary_traj_uid = str(uuid.uuid4())

                        primary_belief_context['traj_uid'] = primary_traj_uid
                        secondary_belief_context['traj_uid'] = secondary_traj_uid
                        primary_belief_context['rewards'] = primary_reward
                        secondary_belief_context['rewards'] = secondary_reward
                        
                        new_total_batch_list.extend([[primary_belief_context], [secondary_belief_context]])
                        new_episode_rewards.extend([primary_reward, secondary_reward])
                        new_episode_lengths.extend([1, 1])
                        new_traj_uid.extend([primary_traj_uid, secondary_traj_uid])
                        if primary_reward == 0.0:
                            # we skip the rest of the trajectory.
                            while i < len(primary_belief_contexts) and primary_belief_contexts[i]['info']["parent_traj_uid"] == primary_belief_context['info']["parent_traj_uid"]:
                                i += 1
                            continue
                        else:
                            parsable_belief_states += 1
                        i += 1
                    total_batch_list = new_total_batch_list
                    episode_rewards = np.array(new_episode_rewards)
                    episode_lengths = np.array(new_episode_lengths)
                    traj_uid = np.array(new_traj_uid)
                    success['fraction_parsable_belief_states_success_rate'] = np.array([parsable_belief_states] * len(primary_belief_contexts)) / len(primary_belief_contexts)
                elif self.config.env.env_name == "colabbench" or (self.config.env.env_name == "combolock" and self.config.trainer.belief_state_grading_type >= 0):
                    if self.config.trainer.belief_state_grading_type >= 0:

                        # possibilities to try for rebuttal:
                        # auto encoder: decoder: log P(x | z); encoder  log P(z | x)
                        # belief grading with log P(o_t, a_t, b_t | b_t+1) = P(o_t | a_t, b_t, b_t+t) P(a_t | b_t, b_t+t) P(b_t | b_t+1)
                        # belief_t = {"states": [state_1, state_2, state_3], "obs": [obs_1, obs_2, obs_3], "action": [act_1, act_2, act_3]}
                        # belief_t-1 = {"states": [state_1, state_2], "obs": [obs_1, obs_2], "action": [act_1, act_2]}
                        # A(b_t, b_t+1)
                        # outbased_reward -> importance what info stored.
                        # belief grading with log P(o_t | a_t, b_t, b_t+1) # won't encourage storing b_t information.
                        # action grading with surprisal -log P(o_t | a_t, b_t) surprise of dynamics model.
                        #   This is the action yeilds information that is not already present from the belief. 
                        #   This doesn't depend on the belief does it? 
                        #   It could be that conditioned on the belief this form of reward shaping is better than conditioning on full length?
                        #   This feels like a messy claim to verify. Not sure if its even true. ?
                        #   vanilla surpisal -log P(o_t | a_t, ..., a_0)
                        #   surprisal of predicting the next belief -log P(b_t+1 | b_t) information gain of world model.
                        #   Multiple observations and actions. Caching. (do all the actions within the span.)
                        # I'll focus on the first two belief grading options up
                        # need to generate the grade for all beliefs
                        from agent_system.environments.prompts.colabbench import COLABBENCH_BELIEF_GRADING_0_NO_LOSS, COLABBENCH_BELIEF_GRADING_1_LOSS, COLABBENCH_BELIEF_GRADING_2_NO_LOSS, COLABBENCH_BELIEF_GRADING_3_LOSS, COLABBENCH_BELIEF_GRADING_4_NO_LOSS, COLABBENCH_BELIEF_GRADING_5_LOSS, COLABBENCH_BELIEF_GRADING_REF_RECONSTRUCTION_0_NO_LOSS, COLABBENCH_BELIEF_GRADING_REF_RECONSTRUCTION_1_LOSS
                        from agent_system.environments.prompts.combolock import COMBO_BELIEF_GRADING_0_NO_LOSS, COMBO_BELIEF_GRADING_1_LOSS, COMBO_BELIEF_GRADING_2_NO_LOSS, COMBO_BELIEF_GRADING_3_LOSS, COMBO_BELIEF_GRADING_4_NO_LOSS, COMBO_BELIEF_GRADING_5_LOSS
                        input_ids_list = []
                        labels_list = []
                        def extract(tag, s):
                            return s.split(f"<{tag}>")[1].split(f"</{tag}>")[0]
                        def prepare_data_for_data_proto(input_ids_list: list[list[int]], labels_list: list[list[int]]):
                            input_ids = pad_sequence([torch.tensor(t) for t in input_ids_list], batch_first=True, padding_value=self.tokenizer.pad_token_id, padding_side='left')
                            labels = pad_sequence([torch.tensor(t) for t in labels_list], batch_first=True, padding_value=-100, padding_side='left')
                            attention_mask = pad_sequence([torch.tensor([1]*len(t)) for t in input_ids_list], batch_first=True, padding_value=0, padding_side='left')
                            position_ids = compute_position_id_with_mask(attention_mask)
                            return input_ids, labels, attention_mask, position_ids
                        
                        for c in all_belief_contexts:
                            if c['is_action_valid']:
                                if self.config.env.env_name == "combolock":
                                    true_prior_obs = c['input_ids_str'].split("Environment feedback:")[1].split("Now update your belief")[0].strip()
                                    true_prior_belief = extract("belief", c['input_ids_str'])
                                    true_prior_action = extract("action", c['input_ids_str'])
                                    prompt_parts=[
                                        COMBO_BELIEF_GRADING_0_NO_LOSS.format(future_belief=c['filtered_belief_generations']),
                                        COMBO_BELIEF_GRADING_1_LOSS.format(prior_belief=true_prior_belief), 
                                        COMBO_BELIEF_GRADING_2_NO_LOSS, 
                                        COMBO_BELIEF_GRADING_3_LOSS.format(prior_action=true_prior_action), 
                                        COMBO_BELIEF_GRADING_4_NO_LOSS, 
                                        COMBO_BELIEF_GRADING_5_LOSS.format(prior_obs=true_prior_obs)
                                    ]
                                else:
                                    true_prior_obs = extract("environment", c['input_ids_str'])
                                    true_prior_belief = extract("belief", c['input_ids_str'])
                                    true_prior_action = extract("action", c['input_ids_str'])
                                    prompt_parts=[
                                        COLABBENCH_BELIEF_GRADING_0_NO_LOSS.format(future_belief=c['filtered_belief_generations'], first_user_query=c['info']['problem_description']),
                                        COLABBENCH_BELIEF_GRADING_1_LOSS.format(prior_belief=true_prior_belief), 
                                        COLABBENCH_BELIEF_GRADING_2_NO_LOSS, 
                                        COLABBENCH_BELIEF_GRADING_3_LOSS.format(prior_action=true_prior_action), 
                                        COLABBENCH_BELIEF_GRADING_4_NO_LOSS, 
                                        COLABBENCH_BELIEF_GRADING_5_LOSS.format(prior_obs=true_prior_obs)
                                    ]
                                
                                if int(self.config.trainer.belief_state_grading_type) == 0:
                                    record_prob_for_parts=[0,1,0,1,0,1]
                                elif int(self.config.trainer.belief_state_grading_type) == 1:
                                    record_prob_for_parts=[0,0,0,0,0,1] # lets ensure that this is properly triggered
                                elif int(self.config.trainer.belief_state_grading_type) == 2:
                                    record_prob_for_parts=[0,0,0,1,0,1]
                                else:
                                    # new int 3 case, where I change the prompt parts. I now want the log prob of the correct answer based on the 
                                    if self.config.env.env_name == "combolock":
                                        raise NotImplemented
                                    else:
                                        prompt_parts=[
                                            COLABBENCH_BELIEF_GRADING_REF_RECONSTRUCTION_0_NO_LOSS.format(belief_state=c['filtered_belief_generations'], first_user_query=c['info']['problem_description']),
                                            COLABBENCH_BELIEF_GRADING_REF_RECONSTRUCTION_1_LOSS.format(code=c['info']['ground_truth'].strip()),
                                        ]
                                    record_prob_for_parts = [0,1]
                                    ...
                                prompt_parts_tokenized = [self.tokenizer.encode(s) for s in prompt_parts]
                                input_ids_list.append(sum(prompt_parts_tokenized, []))
                                labels_list.append(sum([list(ids) if record else [-100] * len(ids) for record, ids in zip(record_prob_for_parts, prompt_parts_tokenized)], []))
                            else:
                                # this is done to keep the size divisible by the number of gpus even if the belief generated isn't valid, which should be rare
                                input_ids_list.append([1, 1])
                                labels_list.append([-100, -100]) # 2 because label will remove 1, and might need non empty tensor for some operatoin.
                                # this is so that the function below still runs if there are no valid beliefs generated.
                        # actor_rollout_wg.compute_log_prob or actor_rollout_wg.compute_ref_log_prob
                        # need the following keys, and log prob is only computed over responses, but these can be set equal to labels. 
                        #  ["responses", "input_ids", "attention_mask", "position_ids"]
                        input_ids, labels, attention_mask, position_ids = prepare_data_for_data_proto(input_ids_list, labels_list)
                        input_for_belief_grading = DataProto(
                            td.TensorDict(dict(
                                input_ids = input_ids,
                                responses = labels, 
                                attention_mask = attention_mask,
                                position_ids = position_ids,
                            ), batch_size=len(input_ids)),
                            meta_info=gen_batch.meta_info, # I missing data_source and index fields. Not sure if important.
                        )
                        log_prob_prior_info_given_future_belief = actor_rollout_wg.compute_log_prob(input_for_belief_grading).batch['old_log_probs']
                        # some implementations of the log_prob don't work with the -100 labels, so actually I have to post process the log_probs.
                        log_prob_prior_info_given_future_belief[labels[:, 1:] == -100] = 0.0
                        belief_grades = log_prob_prior_info_given_future_belief.sum(-1) / (64 if self.config.trainer.div_by_const else (labels[:, 1:] != -100).sum(-1) )
                        belief_grade_token_mask = belief_grades.isnan()
                        belief_grades[belief_grade_token_mask] = -5 # this is just empty seq, so shouldn't matter what value I set it to. doing -10 for safety tho.

                        

                        for c, belief_grade in zip(all_belief_contexts, belief_grades):
                            belief_grade = min(belief_grade.item(), self.config.trainer.ceiling_belief_grading_reward)
                            belief_grade = max(belief_grade, -5) # -5 is a guess at the rough poorest performance we can expect.
                            c['info']['belief_grade'] = belief_grade
                        
                        #     if int(self.config.trainer.belief_state_grading_type) == 3:
                        #         c['info']['belief_grade'] = min((belief_grade.item() // 0.1) * 0.1, self.config.trainer.ceiling_belief_grading_reward) # rounding to nearest 0.1 because we use GRPO normalizing by std, which will take the difference too hard.
                        #     else:
                        #         c['info']['belief_grade'] = min((belief_grade.item() // 0.2) * 0.2, self.config.trainer.ceiling_belief_grading_reward) # rounding to nearest 0.2 because we use GRPO normalizing by std, which will take the difference too hard.

                        primary_belief_contexts, secondary_belief_contexts = all_belief_contexts[:len(flattened_valid_belief_contexts)], all_belief_contexts[len(flattened_valid_belief_contexts):]
                        new_total_batch_list = copy(total_batch_list)
                        new_episode_rewards = episode_rewards.tolist()
                        new_episode_lengths = episode_lengths.tolist()
                        # if you have penalties enabled for belief lengths, you shouldn't have them enabled for overall episode length. This encourages very short trajectories.
                        new_episode_penalties = np.zeros_like(episode_penalties).tolist()
                        new_traj_uid = traj_uid.tolist()
                        i = 0
                        belief_states_graded_in_chain = 0
                        new_max_updates = list(self.maxes)
                        old_maxes = list(self.maxes)
                        log_obs_list = list()
                        advantage_magnitude_list = list()



                        # ok, so for each environment I maintain a max, then I update their max upon getting a new highest reward.
                        # the difference should be much fewer gradient updates, but how should I implement this change because if
                        # the model updates whenever there is something slightly above the current max, and there is a cap on what 
                        # could be considered the max, it will asymptotically converge, perhaps I set a quantization or rounding of 
                        # 0.0001, and then itll happen in finite time. The other issue is that as opposed to a ppo with value model 
                        # implementation, I have to have something to compare this belief grade to, perhaps it isn't so bad, two 
                        # rewards which are very low  won't be able to be improved, which feels wrong, this however connects to the
                        # mountain car game where the model climbing up the hill will not recieve the reward of log obs reconstruction 
                        # improvement again after a certain threshold. Having the negative baseline of a poorer performing policy seems 
                        # fine.
                        uid_to_idx = {uid_i: i for i, uid_i in enumerate(uid_batch)}
                        ema_factor = 0.9
                        while i < len(primary_belief_contexts):
                            primary_belief_context = primary_belief_contexts[i]
                            secondary_belief_context = secondary_belief_contexts[i]
                            if "invalid" in extract("action", primary_belief_context['input_ids_str']):
                                i += 1
                                continue
                            primary_reward = primary_belief_context['info']['belief_grade']
                            secondary_reward = -5.0 if not secondary_belief_context['is_action_valid'] else secondary_belief_context['info']['belief_grade']

                            primary_traj_uid = str(uuid.uuid4())
                            secondary_traj_uid = str(uuid.uuid4())

                            primary_belief_context['traj_uid'] = primary_traj_uid
                            secondary_belief_context['traj_uid'] = secondary_traj_uid
                            
                            # update the primary and secondary_reward here if they are more than the prior maxes.
                            assert primary_belief_context['info']["parent_uid"] == secondary_belief_context['info']["parent_uid"]
                            idx_of_uid_of_belief = uid_to_idx[primary_belief_context['info']["parent_uid"]]
                            log_obs_list.extend([primary_reward, secondary_reward])
                            og_primary_reward = primary_reward
                            # uncomment below for supporting potential based reward shaping.
                            # primary_phi = ema_factor * old_maxes[idx_of_uid_of_belief] + (1-ema_factor) * max(primary_reward, old_maxes[idx_of_uid_of_belief])
                            # secondary_phi = ema_factor * old_maxes[idx_of_uid_of_belief] + (1-ema_factor) * max(secondary_reward, old_maxes[idx_of_uid_of_belief])
                            # if primary_phi > old_maxes[idx_of_uid_of_belief] or secondary_phi  > old_maxes[idx_of_uid_of_belief]: 
                            #     new_max_updates[idx_of_uid_of_belief] = max(primary_phi, secondary_phi, new_max_updates[idx_of_uid_of_belief])
                            # else:
                            #     # I could skip these to speed up the runtime, but want to document them without changing too much code.
                            #     assert primary_phi == secondary_phi
                            # primary_reward = primary_phi
                            # secondary_reward = secondary_phi

                            # I want to bound the advantage that I will apply after GRPO without normalization is applied. for these two rewards, I'll find their mean and subtract it, so I just need to ensure they are less than 0.7 away from their mean, otherwise, I'll clip them
                            # a_p = p - (p + s) / 2, well this can be done by finding their distance, then adding to the smaller one double the difference above 0.7 that the mean difference is.
                            if primary_reward < secondary_reward:
                                primary_reward += max(0, abs((primary_reward-secondary_reward)/2)-0.7/5) * 2
                            else:
                                secondary_reward += max(0, abs((primary_reward-secondary_reward)/2)-0.7/5) * 2


                            advantage_magnitude_list.extend([abs((primary_reward - secondary_reward)/2) * self.config.trainer.belief_state_grading])
                            primary_belief_context['rewards'] = primary_reward
                            secondary_belief_context['rewards'] = secondary_reward

                            new_total_batch_list.extend([[primary_belief_context], [secondary_belief_context]])
                            new_episode_rewards.extend([primary_reward, secondary_reward])
                            new_episode_lengths.extend([1, 1])
                            if secondary_belief_context['is_action_valid']:
                                avg = (primary_belief_context["filtered_belief_generations_len"] + secondary_belief_context["filtered_belief_generations_len"]) / 2
                                if primary_belief_context["filtered_belief_generations_len"] == secondary_belief_context["filtered_belief_generations_len"]:
                                    avg = avg - 20 # we will penalize both a bit if they are equal? this to discourage a deterministic belief generation which just copies the inputs directly.
                                # I only want to apply the penalty to things larger than 100, if the length degenerates to 0, I don't want this to be rewarded. This is very bad.
                                # so if both are larger than 100, then I do this calculation, which will favor the shorter of the two.
                                if self.config.env.env_name == "combolock":
                                    if primary_belief_context["filtered_belief_generations_len"] > 200 and secondary_belief_context["filtered_belief_generations_len"] > 200:
                                        new_episode_penalties.extend([min(max(primary_belief_context["filtered_belief_generations_len"]-avg, -70), 70), 
                                                                      min(max(secondary_belief_context["filtered_belief_generations_len"]-avg, -70), 70)])
                                    else: 
                                        new_episode_penalties.extend([0, 0])
                                else:
                                    if primary_belief_context["filtered_belief_generations_len"] > 400 and secondary_belief_context["filtered_belief_generations_len"] > 400:
                                        new_episode_penalties.extend([primary_belief_context["filtered_belief_generations_len"]-avg, secondary_belief_context["filtered_belief_generations_len"]-avg])
                                    else:
                                        new_episode_penalties.extend([0, 0])

                            else:
                                new_episode_penalties.extend([0,0])

                            new_traj_uid.extend([primary_traj_uid, secondary_traj_uid]) 
                            if og_primary_reward < (-1.4 if self.config.trainer.div_by_const else -1.6): # very heuristic guess. 
                                # we skip the rest of the trajectory. 
                                while i < len(primary_belief_contexts) and primary_belief_contexts[i]['info']["parent_traj_uid"] == primary_belief_context['info']["parent_traj_uid"]:
                                    i += 1
                                continue
                            else:
                                belief_states_graded_in_chain += 1
                            i += 1
                        # print("maxes", self.maxes)
                        for i, new_max_update in enumerate(new_max_updates):
                            self.maxes[i] = new_max_update
                        max_phis_list = self.maxes[self.config.env.rollout.n-1::self.config.env.rollout.n]
                        # print("maxes", self.maxes)
                        success['belief_grade_advantage_magnitude_mean_success_rate'] = np.array([np.array(advantage_magnitude_list).mean().item()] * len(primary_belief_contexts))
                        success['belief_grade_phi_mean_success_rate'] = np.array([np.array(max_phis_list).mean().item()] * len(primary_belief_contexts))
                        success['belief_grade_log_obs_mean_success_rate'] = np.array([np.array(log_obs_list).mean().item()] * len(primary_belief_contexts))
                        # adding different belief grading support for 
                        # (1) reconstruction of true answer with belief state, and 
                        # (2) correct answer generation with belief state. 
                        # Below will be (2), and above will be (1), because its already pretty supported in the existing implementation.
                    else:
                        # need to generate actions, and do the environment rollout, 
                        # but ask the step function to get the scores for the actions
                        # implicitly this should also prompt the model to respond in its action, 
                        # so will probably just set step count to last step value.
                        # mark new contexts as belief grading, give them a new uid,
                        # ohh wait if I add them as new dicts in the trajectory, they will be passed through policy grad because of my poor accounting
                        # this could be fine, but if I start incentivizing shorter beliefs along with this outcome reward, we could get confusing bugs.
                        # because the scope is small, will just add the completion into infos's dict, if I want to see it for any reason.
                        from agent_system.environments.prompts.colabbench import COLABBENCH_ACTION_PROMPT, COLABBENCH_AGENT_FIRST_MESSAGE
                        parent_uid_to_task_desciption = {uid_i: envs.task_desciptions[i] for i, uid_i in enumerate(uid_batch)}
                        action_prompt_chats = []
                        # why don't I just add the belief action to the trajectory reward again? I could even add it independantly of the belief grading? 
                        # Well, this would be pretty far removed from belief grading's impact on the performance, but should be something that someone does if they just wanted higher performance.
                        for belief_context_dict in all_belief_contexts:
                            agent_first_message = COLABBENCH_AGENT_FIRST_MESSAGE.format(max_attempts=self.config.env.max_attempts)
                            hint = "It is your last step."
                            parent_uid = belief_context_dict['info']["parent_uid"]
                            belief = belief_context_dict['filtered_belief_generations'] if belief_context_dict['is_action_valid'] else ""
                            action_prompt_chats.append([{'role':'user', 'content': COLABBENCH_ACTION_PROMPT.format(agent_first_message=agent_first_message, first_user_query=parent_uid_to_task_desciption[parent_uid], belief_state=belief, hint=hint)}])
                        keys_for_generation = ["input_ids", "attention_mask", "position_ids"]
                        input_for_action_gen = DataProto.from_single_dict(data=collate_fn([{k: e[k] for k in keys_for_generation} | {"data_source": "", "raw_prompt": ""} for e in all_belief_contexts]))
                        batch = self.preprocess_batch(gen_batch=input_for_action_gen, obs={'text': ['']*len(action_prompt_chats), 'chat': action_prompt_chats, 'image': None, 'anchor': [""] * len(action_prompt_chats)})
                        batch_input = batch.pop(
                            batch_keys=keys_for_generation,
                        )

                        batch_input.meta_info = input_for_action_gen.meta_info
                        immediate_action_gen_outputs = actor_rollout_wg.generate_sequences(batch_input)
                        
                        
                        # I need to build the inputs for asking for the answer in the next step.
                        # input_for_belief_gen.batch["input_ids"] = input_for_belief_gen.batch["input_ids"][:, :self.config.data.max_prompt_length]
                        # input_for_belief_gen.batch["attention_mask"] = input_for_belief_gen.batch["attention_mask"][:, :self.config.data.max_prompt_length]
                        # input_for_belief_gen.batch['position_ids'] = input_for_belief_gen.batch['position_ids'][:, :self.config.data.max_prompt_length]
                        # need to generate the right input_ids for creating an action after belief. 
                        # This could involve communicating with the environment/context management object (envs) and creating a custom object to get these contexts.
                        # I think it would just be easier to deal with the prompts individually rather than creating a nice general interface for it.
                        
                        new_action_response_strs = self.tokenizer.batch_decode(immediate_action_gen_outputs.batch['responses'], skip_special_tokens=True)
                        # new_belief_action_or_belief_texts, new_belief_valids = envs.get_belief_from_output_text(new_action_response_strs)
                        """
                        all_belief_contexts = [{"info": {"parent_uid": ...}}]
                        then I know they should be ordered as uid_batch was when I enter them into the envs.envs.step function.
                        uid_batch = [uid_1, uid_1, uid_2, uid_2, uid_3, ..., uid_15, uid_16, uid_16] # where uid's are repeated depending on group size

                        """
                        parent_uid_to_action_and_index: dict[str, list[tuple[str, int]]] = defaultdict(list)
                        for belief_context_index, (action, parent_uid) in enumerate(zip(new_action_response_strs, [belief_context_dict['info']['parent_uid'] for belief_context_dict in all_belief_contexts])):
                            parent_uid_to_action_and_index[parent_uid].append((action, belief_context_index))
                        
                        num_environment_steps = math.ceil(max(map(len, parent_uid_to_action_and_index.values())) / self.config.env.rollout.n)
                        belief_context_index = []
                        actives = []
                        rewards = []
                        new_action_response_strs_batch_ordering = []
                        for _ in range(num_environment_steps):
                            new_action_response_strs_batch = []
                            actives_batch = []
                            for uid_i in uid_batch:
                                if len(parent_uid_to_action_and_index[uid_i]) > 0:
                                    parent_uid_to_action_and_index_item = parent_uid_to_action_and_index[uid_i].pop(-1)
                                    new_action_response_strs_batch.append(parent_uid_to_action_and_index_item[0])
                                    belief_context_index.append(parent_uid_to_action_and_index_item[1])
                                    actives_batch.append(True)
                                else:
                                    new_action_response_strs_batch.append("")
                                    actives_batch.append(False)
                                    belief_context_index.append(-1)
                            code_tags, new_action_texts, new_action_valids = envs.projection_f(new_action_response_strs_batch, np.zeros(len(new_action_response_strs)))
                            # might not be all code tags or all valids, just depends on how the model did when generating.
                            # batch = batch.union(batch_output)
                            # text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
                            # next_obs, rewards, dones, infos = envs.step(text_actions, is_done, self.tokenizer)
                            # tag, action, is_belief_generation_step, just_code
                            actions = list(zip(code_tags, new_action_texts, ~np.array(new_action_valids, dtype=bool) | ~np.array(actives_batch, dtype=bool), np.ones_like(new_action_valids, dtype=bool)))
                            text_obs, rewards_batch, dones, infos = envs.envs.step(actions)
                            actives.extend(actives_batch)
                            rewards.extend(rewards_batch)
                            new_action_response_strs_batch_ordering.extend(new_action_response_strs_batch)
                        # todo, check this ordering operation matches the actions that were taken.
                        """
                        actions_ordered = [ r for r, _ in 
                                            sorted([(r, bci) for r, a, bci in zip(new_action_response_strs_batch_ordering, actives, belief_context_index) if a == True],
                                                key=itemgetter(1))
                        ]
                        This should match new_action_response_strs
                        also, make sure np.array(actives) is consistent with np.array(belief_context_index) != -1 it is.
                        """
                        rewards_ordered = [ r for r, _ in 
                                            sorted([(r, bci) for r, a, bci in zip(rewards, actives, belief_context_index) if a == True],
                                                key=itemgetter(1))
                        ]
                        belief_grades = np.array(rewards_ordered)
                        belief_grade_token_mask = np.isnan(belief_grades) # this shouldn't occur.
                        for c, belief_grade, s in zip(all_belief_contexts, belief_grades, new_action_response_strs):
                            c['info']['belief_grade'] = belief_grade
                            c['info']['immediate_action'] = s

                        primary_belief_contexts, secondary_belief_contexts = all_belief_contexts[:len(flattened_valid_belief_contexts)], all_belief_contexts[len(flattened_valid_belief_contexts):]
                        new_total_batch_list = copy(total_batch_list)
                        new_episode_rewards = episode_rewards.tolist()
                        new_episode_lengths = episode_lengths.tolist()
                        # if you have penalties enabled for belief lengths, you shouldn't have them enabled for overall episode length. This encourages very short trajectories.
                        new_episode_penalties = np.zeros_like(episode_penalties).tolist()
                        new_traj_uid = traj_uid.tolist()
                        i = 0
                        belief_states_graded_in_chain = 0

                        while i < len(primary_belief_contexts):
                            primary_belief_context = primary_belief_contexts[i]
                            secondary_belief_context = secondary_belief_contexts[i]
                            primary_reward = primary_belief_context['info']['belief_grade']
                            secondary_reward = 0 if not secondary_belief_context['is_action_valid'] else secondary_belief_context['info']['belief_grade']
                            primary_traj_uid = str(uuid.uuid4())
                            secondary_traj_uid = str(uuid.uuid4())

                            primary_belief_context['traj_uid'] = primary_traj_uid
                            secondary_belief_context['traj_uid'] = secondary_traj_uid
                            primary_belief_context['rewards'] = primary_reward
                            secondary_belief_context['rewards'] = secondary_reward
                            
                            new_total_batch_list.extend([[primary_belief_context], [secondary_belief_context]])
                            new_episode_rewards.extend([primary_reward, secondary_reward])
                            new_episode_lengths.extend([1, 1])
                            # this penalty will likely not be applied unless requested for by reviewers.
                            if secondary_belief_context['is_action_valid']:
                                avg = (primary_belief_context["filtered_belief_generations_len"] + secondary_belief_context["filtered_belief_generations_len"]) / 2
                                if primary_belief_context["filtered_belief_generations_len"] > 400 and secondary_belief_context["filtered_belief_generations_len"] > 400:
                                    new_episode_penalties.extend([primary_belief_context["filtered_belief_generations_len"]-avg, secondary_belief_context["filtered_belief_generations_len"]-avg])
                                else: 
                                    new_episode_penalties.extend([0, 0])
                            else:
                                new_episode_penalties.extend([0,0])

                            new_traj_uid.extend([primary_traj_uid, secondary_traj_uid])
                            # we actually always grade beliefs. There is no notion of stopping midway through because the grading is bad, 
                            # this is more of an advantage estimate similar to Multi-Turn Code Generation Through Single-Step Rewards: https://arxiv.org/pdf/2502.20380v1
                            # we take advantage of the shorter belief generation contexts rather than re forwardpassing large context with full ctx. 
                            # Might not actually be so important, depends on compute/data constraints.
                            belief_states_graded_in_chain += 1
                            i += 1





                    # then just attribute the rewards to the right spot. 
                    # (issue will be that the action generations are so many, 
                    # and won't be easy to grade all at once. volume is 32 * 10?
                    # I can just set up a loop where they are graded one by one, 
                    # the limiting factor in time will be the generations anyway)


                    total_batch_list = new_total_batch_list
                    episode_rewards = np.array(new_episode_rewards)
                    episode_lengths = np.array(new_episode_lengths)
                    episode_penalties = np.array(new_episode_penalties)
                    traj_uid = np.array(new_traj_uid)
                    success['total_avg_belief_grade_success_rate'] = np.array([belief_grades[~belief_grade_token_mask].mean().item()] * len(primary_belief_contexts)) # / len(primary_belief_contexts)
                    success['fraction_parsable_belief_states_success_rate'] = np.array([belief_states_graded_in_chain] * len(primary_belief_contexts)) / len(primary_belief_contexts)
                if len(episode_penalties) != len(episode_rewards):
                    episode_penalties = np.array(episode_penalties.tolist() + [0] * (len(episode_rewards) - len(episode_penalties)))# we need a longer episode penalties to account for the new belief states being graded.

                # all_belief_lens = np.array([ls[0]['filtered_belief_generations_len'] if ls[0]['info'].get('is_belief_grading_context', False) else 0 for ls in total_batch_list])
                # valids = np.array([ls[0]['is_action_valid'] if ls[0]['info'].get('is_belief_grading_context', False) else 0 for ls in total_batch_list])
                # valids = np.logical_and(valids, all_belief_lens > 0)
                # if valids.sum() > 0:
                #     mean_belief_len = all_belief_lens[valids == 1].mean()
                #     episode_penalties_temp = np.zeros_like(episode_rewards)
                #     episode_penalties_temp[valids == 1] = all_belief_lens[valids == 1] # - mean_belief_len
                #     episode_penalties = episode_penalties_temp
                # else:
                #     episode_penalties = np.array(np.zeros_like(episode_penalties).tolist() + [0] * (len(episode_rewards) - len(episode_penalties)))
                # episode_penalties = np.array(episode_penalties.tolist() + [0] * (len(episode_rewards) - len(episode_penalties)))# we need a longer episode penalties to account for the new belief states being graded.
                # for _ in zip([1]): # permutations
                #     # check here if the context is valid of invalid, because we just passed it through.
                #     then from the info, apply it to get the correct beleif state, and if we skip the shit, then go to the end of the parent_traj_uid. Also should do a while True:?

        # attention_mask: Tensor(shape=torch.Size([64, 2560]), device=cpu, dtype=torch.int64, is_shared=False),
        # input_ids: Tensor(shape=torch.Size([64, 2560]), device=cpu, dtype=torch.int64, is_shared=False),
        # position_ids: Tensor(shape=torch.Size([64, 2560]), device=cpu, dtype=torch.int64, is_shared=False),
        # prompts: Tensor(shape=torch.Size([64, 2048]), device=cpu, dtype=torch.int64, is_shared=False),
        # responses: Tensor(shape=torch.Size([64, 512]), device=cpu, dtype=torch.int64, is_shared=False),
        # rollout_log_probs:
        # need to convert
        return total_batch_list, episode_rewards, episode_penalties, episode_lengths, success, traj_uid

    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            batch_list, episode_rewards, episode_lengths, success, traj_uid = filter_group_data(batch_list=batch_list,
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic or vanilla).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        # Initial observations from the environment
        if self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            total_episode_penalties = None
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_penalties, total_episode_lengths, total_success, total_traj_uid = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_penalties)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        

        # Create trajectory data

        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            episode_penalties=total_episode_penalties,
        )
        
        return gen_batch_output
