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
from copy import deepcopy

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
        not_is_belief_grading_context_list = [not info.get('is_belief_grading_context', False) for info in [trajectory_list[0]['info'] for trajectory_list in total_batch_list]]
        batch_size = len(total_batch_list)

        episode_rewards_mean = np.mean(episode_rewards[not_is_belief_grading_context_list])
        episode_rewards_min = np.min(episode_rewards[not_is_belief_grading_context_list])
        episode_rewards_max = np.max(episode_rewards[not_is_belief_grading_context_list])

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
                    if not_is_belief_grading_context_list[bs]:
                        data['episode_rewards'] = episode_rewards[bs]
                    data['episode_rewards_mean'] = episode_rewards_mean
                    data['episode_rewards_min'] = episode_rewards_min
                    data['episode_rewards_max'] = episode_rewards_max
                    # episode_lengths
                    if not_is_belief_grading_context_list[bs]:
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
        episode_lengths = np.zeros(batch_size, dtype=np.int32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)

        # need a completely different loop to handle single context stuff, because else it will be really messy. Almost none will be shared I think.
        if self.config.actor_rollout_ref.actor.single_context:
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
                        if len(input_ids) >= self.config.data.max_prompt_length:
                            prompt_too_long[i] = (True)


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
        if self.config.env.non_terminal_penalty:
            episode_rewards[np.logical_not(is_done) | prompt_too_long] += -self.config.env.non_terminal_penalty

        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        success["non_terminal_trajectories_success_rate"] = np.logical_not(is_done) | prompt_too_long # this should be prompt too long
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
        # breakpoint()
        # generate a pair for each, re label the traj_uid
        # if self.config.trainer.belief_state_grading:
        #   flattened_valid_belief_contexts = [c for trajectory in total_batch_list for c in trajectory if (c['active_masks'] and c['info']["is_action_valid"] and c['info']["action_or_belief"])]
        #   # technically we don't have to filter on the successful beliefs, but lets do this for now.
        #   # we want to create a deepcopy of the full set of valid beleif contexts, then remove unnecessary info, and populate with new traj_uid and uid info and reshuffle into total_batch_list format.
        #   flattened_valid_belief_contexts = deepcopy(flattened_valid_belief_contexts)
        #   # set most of these to none. there are some other things we need to do, like make sure the attention_mask, input_ids, are cut to length, and remove the response, and rollout_log_probs, and change uid to parent_uid and traj_uid to parent_traj_uid (['attention_mask', 'prompts', 'input_ids', 'responses', 'rollout_log_probs', 'position_ids', 'anchor_obs', 'index', 'data_source', 'uid', 'traj_uid', 'raw_prompt', 'is_action_valid', 'rewards', 'active_masks', 'response_ids_str', 'response_ids_token_len', 'input_ids_str', 'input_ids_token_len', 'filtered_belief_generations', 'filtered_action_generations', 'step', 'info'])
        #   # set to nan 'response_ids_str', 'response_ids_token_len', 'input_ids_str', 'input_ids_token_len', 'anchor_obs' => "", ("is_action_valid", 'rewards', 'active_masks', 'filtered_belief_generations', 'filtered_action_generations', 'step') will all have new values, and need to change things in info
        #   num_beliefs_to_compare = len(flattened_valid_belief_contexts)
        #   new_uid = np.array([str(uuid.uuid4()) for _ in range(num_beliefs_to_compare)], dtype=object)
        #   
        #   
        #   new_traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)



        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid
    
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
        else:
            # Vanilla Sampling   
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
        )
        
        return gen_batch_output