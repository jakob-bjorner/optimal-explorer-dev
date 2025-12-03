import ray
import numpy as np
import datasets
import string
import re
# from optimal_explorer.mdps.nqhotpotqa import NQHotpotQA

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score

def compute_score_em(solution_str, ground_truth):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answers = [a.strip() for a in solution_str.split(';')]
    
    score = 0
    for idx, answer in enumerate(answers):
        try:
            score += int(em_check(answer, ground_truth[idx]))
        except Exception as e:
            score += 0
    return score

@ray.remote(num_cpus=0.2)
class NQHotpotQAWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds an independent instance of the specified gym environment.
    """
    def __init__(self, seed, num_envs, split, force_full, num_objectives):
        """Initialize the gym environment in this worker"""
        self.split = split
        self.num_envs = num_envs
        self.ds = datasets.load_dataset("../MEM1/Mem1/train/data/nq_hotpotqa_train_multi_"+str(num_objectives), split=self.split)
        # self.env = NQHotpotQA(combination_length, max_attempts, vocab)
        # self.ds.shuffle(seed)
        self.force_full = force_full
        if num_objectives <= 4:
            self.max_attempts = 6
        else:
            self.max_attempts = 20
        self.attempt = 0
        if self.split == 'test':
            self.index_ordering = (seed + num_envs * np.arange(len(self.ds))) % len(self.ds)
            permutation_inderection = np.random.default_rng(42).permutation(len(self.ds)) # make the order in which we do eval random.
            self.index_ordering = permutation_inderection[self.index_ordering]
        else:
            self.index_ordering = np.random.default_rng(seed).permutation(len(self.ds))
        # self.env.reset(seed)
        self.index = -1
        self.has_won = False
    
    def step(self, action):
        """Execute a step in the environment"""
        tag, action, skip_sampling_step = action
        info = {"attempt": self.attempt+1, "is_last_step": self.attempt+1==self.max_attempts, "steps_remaining": self.max_attempts - self.attempt - 1, "target": self.data['reward_model']['ground_truth']['target']}
        if skip_sampling_step:
            # the belief generation step is checked only on the top level env manager. 
            # This is just a noop so other environments can step if they need to.
            return "", 0, False, info |{"won": self.has_won}
        if self.attempt+1 >= self.max_attempts:
            return "", 0, True, info | {"won": self.has_won}
        # obs, _, done, info = self.env.step(action)
        # str_response_in_tool_call = ""
        # for i, (g, f) in enumerate(zip(action, info['feedback'])):
        #     position = i + 1
        #     if f == 0: 
        #         str_response_in_tool_call += f"\n{g} is not in the lock"
        #     elif f == 1: 
        #         str_response_in_tool_call += f"\n{g} is not in Position {position}, but is in the lock"
        #     else: # f == 2
        #         str_response_in_tool_call += f"\n{g} is in Position {position}!"
        # str_response_in_tool_call = str_response_in_tool_call.strip()

        # will only do something unique here for answer tag, because the search tag seems best handled in batched form.
        

        # reward should only be given from the self.env.get_trajectory_score function
        info = info | {"won": False}
        
        obs_res = ""
        done = False
        if tag == "answer":
            if self.force_full and self.attempt+1 == self.max_attempts:
                reward = 0
            else:
                reward = compute_score_em(action, self.data['reward_model']['ground_truth']['target'])
                if reward > 0:
                    info = info | {"won": True}
                    self.has_won = True
            done = True
        else:
            reward = 0
        if tag == "search":
            num_turns_left = self.max_attempts - self.attempt - 1
            if num_turns_left > 1:
                hint = f"<hint>You have {num_turns_left} turns left.</hint>"
            elif num_turns_left == 1:
                hint = f"<hint>You have 1 turn left. You must answer the question in the next turn.</hint>"
            else:
                hint = ""
            obs_res = hint

        self.attempt += 1
        return obs_res, reward, done, info
    
    def reset(self, seed_for_reset=None):
        """Reset the environment with optional seed"""
        if not self.split == 'test' and seed_for_reset is not None:
            self.index = int(np.random.default_rng(seed_for_reset).choice(self.index_ordering))
        else:
            self.index = (self.index + 1) % len(self.ds)
        # for this environment, it actually does matter what we return,
        # because the first observation needs to contain the question information
        self.data = self.ds[int(self.index_ordering[self.index])]
        prompt_str: str = self.data['prompt'][0]['content']
        question_str = prompt_str.split("\n\nAnswer the following questions:")[1].strip()
        self.has_won = False
        self.attempt = 0
        return question_str, {}
    def get_ds_len(self):
        return len(self.ds)

    


class NQHotpotQAEnvs:
    """
    Ray-based parallel environment wrapper for gym cards environments.
    - env_id: combo lock environment ID
    - env_num: Number of distinct environments
    - group_n: Number of replicas within each group (commonly used for multiple copies with the same seed)
    - env_kwargs: Parameters needed to create a single gym.make(env_id)
    """

    def __init__(self,
                 split,
                 num_objectives,
                 force_full,
                 seed=0,
                 env_num=1,
                 group_n=1,
                 is_train=True):
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
        self.split = split
        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        assert self.num_processes <= 512, "we probably can't load too much data 20 MB for train, and we load the Dataset for every worker."

        np.random.seed(seed)
        
        # Create Ray remote actors instead of processes
        self.workers = []
        self.reset_count = 0
        seeds = np.arange(env_num).repeat(group_n)
        for i in range(self.num_processes):
            seed_i = seeds[i]
            worker = NQHotpotQAWorker.remote(
                seed_i,
                env_num,
                split,
                force_full,
                num_objectives
            )
            self.workers.append(worker)

    def step(self, actions):
        """
        Perform step in parallel.
        :param actions: list or numpy array, length must equal self.num_processes.
        :return: obs_list, reward_list, done_list, info_list
        """
        assert len(actions) == self.num_processes

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self.workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
        
        if isinstance(obs_list[0], np.ndarray):
            obs_list = np.array(obs_list)
        return obs_list, reward_list, done_list, info_list

    def get_ds_len(self):
        return ray.get(self.workers[0].get_ds_len.remote())
    def reset(self):
        """
        Perform reset in parallel.
        Different seeds will be assigned to each environment (or the same seed within a group).
        :return: (obs_list, info_list)
        """
        if self.split == 'test':
            # we want to do one epoch, so we should increment the seed so it steps through the dataset.
            seeds = np.arange(self.env_num) + self.reset_count
        else:
            if self.is_train:
                seeds = np.random.randint(0, 2**16 - 1, size=self.env_num)
            else:
                seeds = np.random.randint(2**16, 2**32 - 1, size=self.env_num)
        self.reset_count += 1
        # Repeat seed for environments in the same group
        seeds = np.repeat(seeds, self.group_n)
        seeds = seeds.tolist()

        # Send reset commands to all workers
        futures = []
        for i, worker in enumerate(self.workers):
            future = worker.reset.remote(seeds[i])
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        if isinstance(obs_list[0], np.ndarray):
            obs_list = np.array(obs_list)
        return obs_list, info_list
    def get_epochs(self):
        # we do one reset and then we rollout
        return self.reset_count * self.env_num / self.get_ds_len()

    def close(self):
        """
        Close all Ray actors.
        """
        # Kill all Ray actors
        for worker in self.workers:
            ray.kill(worker)

    def __del__(self):
        self.close()


def build_nqhotpotqa_envs(split,
                        num_objectives,
                        force_full,
                        seed,
                        env_num,
                        group_n,
                        is_train=True):
    """
    Externally exposed constructor function to create parallel Combolock environments.
    - max_attempts: for combo lock
    - vocab: for combo lock
    - seed: For reproducible randomness
    - env_num: Number of distinct environments
    - group_n: Number of environment replicas under the same seed
    - is_train: Determines the seed range used (train/test)
    """
    return NQHotpotQAEnvs(
        split,
        num_objectives,
        force_full,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
    )







# def process_guess_msg(msg_str, vocab, combination_length):
#     remove_list = ["**", "</Answer>", "</answer>", "<Answer>", "<answer>", "</Ans>", "</ans>", "<Ans>", "<ans>","<Action>","</Action>","<action>","</action>",'[action]','[/action]','[answer]','[/answer]']
#     def rem_list_from_str(s: str):
#         if s.endswith("**"):
#             s = s[:-2]
#         for rm_str in remove_list:
#             s = s.replace(rm_str, "")
#         return s
#     guess = ''.join(c for c in rem_list_from_str(msg_str) if c in vocab)[-combination_length:].lower()
#     return guess


# class ComboLockInteraction(BaseInteraction):
#     """ Jakob rewrite of gsm8k interaction for general combo lock setting. (using this over tool call because default support for interactions was added just recently.)
#     - `start_interaction`: start a interaction instance for a trajectory.
#     - `generate_response`: generate the response of the user.
#     - `calculate_score`: calculate the score of the interaction.
#     - `finalize_interaction`: finalize the interaction instance.
#     """
#     def __init__(self, config: dict):
#         super().__init__(config)
#         self._instance_dict = {}

#     async def start_interaction(self, instance_id: Optional[str], combination_length: int, max_attempts: int, vocab: str, ground_truth: Tuple[str], format: str, format_penalty_coef: float, lax_format: bool, **kwargs) -> str:
#         if instance_id is None:
#             instance_id = str(uuid4())
#         self.lax_format = lax_format
#         env = CombinationLock(combination_length, max_attempts, vocab)
#         env.reset()
#         env.target_combination = "".join(map(str, ground_truth))
#         self._instance_dict[instance_id] = {"env": env, "format": format, "invalid_format_errors": 0}
#         self.format_penalty_coef = format_penalty_coef
#         return instance_id

#     async def generate_response(self, instance_id: str, messages: List[Dict[str, Any]], **kwargs) -> Tuple[bool, str, float, dict]:
#         mdp = self._instance_dict[instance_id]['env']
#         content = messages[-1]['content'] # I assume the last message given will be the assistant waiting for a user response?
#         contents = [content]
#         # jakob: temp comment out to try generous version of parsing.
#         if not self.lax_format:
#             contents, valid, error_msg = process_msg_content(content, tag_list=['action'])
#             if not valid:
#                 self._instance_dict[instance_id]["invalid_format_errors"] += 1
#                 return False, error_msg, 0.0, {}
        
#         guess = process_guess_msg(contents[0], mdp.vocab, mdp.combination_length)
#         if not mdp._is_valid_guess(guess):
#             # mdp.current_attempt += 1 # we don't increment the current attempt just so we don't confound our attempts when successful number. Only penalize incorrect attempts through length.
#             # if mdp.current_attempt == mdp.max_attempts:
#             #     # we are done.
#             #     return True, "DONE", -1.0, {} # this should only happen when you run out on your last guess because it is unclear.
#             content_summary = contents[0] if len(contents[0]) < 20 else f"...{contents[0][-20:]}"
#             self._instance_dict[instance_id]["invalid_format_errors"] += 1
#             return False, f"Could not parse valid guess from: '{content_summary}'. Please ensure the guess is contained in the final characters of your response, and using only use the characters from the vocab in your guess characters. Do not repeat characters in your guess.", 0.0, {}
#         obs, reward, done, info = mdp.step(guess)
#         str_response_in_tool_call = ""
#         for i, (g, f) in enumerate(zip(guess, info['feedback'])):
#             position = i + 1
#             if f == 0: 
#                 str_response_in_tool_call += f"\n{g} is not in the lock"
#             elif f == 1: 
#                 str_response_in_tool_call += f"\n{g} is not in Position {position}, but is in the lock"
#             else: # f == 2
#                 str_response_in_tool_call += f"\n{g} is in Position {position}!"
#         str_response_in_tool_call = str_response_in_tool_call.strip()
#         if self._instance_dict[instance_id]['format'] == "interaction_belief":
#             str_response_in_tool_call += ""
#             # str_response_in_tool_call += ("\nNow update your beliefs and make your next query to the lock."
#             #                             " Knowledge in your beliefs must only be updated but can never be discarded,"
#             #                             " forgotten, or removed. Do not say anything about which information is new"
#             #                             " and updated or old and remains the same.\n"
#             #                             "Please format your response as: <Update>Any step-by-step"
#             #                             " thinking to update your latest beliefs about the code with the latest"
#             #                             " feedback.</Update><Beliefs>Your new beliefs</Beliefs><Think>Any step-by-step"
#             #                             " thinking to determine what the next query should be based"
#             #                             f" on your beliefs</Think><Action>Your query to the lock ({mdp.combination_length} characters, all different)</Action>")
#         elif self._instance_dict[instance_id]['format'] == "interaction_think":
#             str_response_in_tool_call += ""
#             # str_response_in_tool_call += ("\nNow make your next query to the lock. Please format your"
#             #                              " response as: <think> Any step-by-step thinking"
#             #                              " to determine what the next query should be </think> <answer> Your query"
#             #                              f" to the lock ({mdp.combination_length} characters, all different) </answer>")
#         if done: 
#             reward = mdp.get_trajectory_score()
#         return done, str_response_in_tool_call, reward, {}
#     def get_attempts(self, instance_id: str) -> int:
#         return self._instance_dict[instance_id]['env'].current_attempt
#     def get_trajectory_info(self, instance_id: str) -> dict:
#         return self._instance_dict[instance_id]['env'].get_trajectory_info() | {"invalid_format_errors": self._instance_dict[instance_id]["invalid_format_errors"]}
    
#     def get_mdp(self, instance_id: str):
#         return self._instance_dict[instance_id]['env']
#     def get_format_penalty_coefficient(self, instance_id: str):
#         return self._instance_dict[instance_id]['env']
#     async def calculate_score(self, instance_id: str, **kwargs) -> float:
#         # this is used in  sglang_rollout.py, and we ignore the step level reward to account for early terminating sequences.
#         return self._instance_dict[instance_id]['env'].get_trajectory_score()
#         # the user per interaction score is used instead.
#         # return 0.0
#     def get_format_penalty_coef(self, instance_id: str):
#         return self.format_penalty_coef / self._instance_dict[instance_id]['env'].max_attempts

#     async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
#         del self._instance_dict[instance_id]
