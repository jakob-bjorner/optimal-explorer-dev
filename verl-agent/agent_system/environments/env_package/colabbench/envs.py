import ray
import numpy as np
import json
from openai import OpenAI
from sweet_rl.environments import (HumanDesignInteractionEnv,
                                   HumanInteractionEnv)
from sweet_rl.utils import check_correctness
from ...prompts import COLABBENCH_HUMAN_SIMULATOR_CODE_PROMPT
@ray.remote(num_cpus=0.2)

class ColabBenchWorker:
    """
    Ray remote actor that replaces the worker function.
    Each actor holds an independent instance of the specified gym environment.
    """
    """
    The implementation in sweet rl batches the environment and agent 
    calls, much like the verl-agent environment. 
    They however don't structure the calls as a single conversation 
    it seems. They structure the calls to the langauge model calls 
    as a single user question followed by the assistant for both the 
    agent and environment.

    What should I do for action and environment? 
    I can simply copy what they do. It will just work. 
    might be slower, but probably the right idea. 
    What should I do in terms of the agent tho? 
    I should restructure their calls? 
    how do I structure the calls for vanilla versions of combo lock? 
    I was able to make it in a nice implementation, one tensor per trajectory.
    So, I can do this for the vanilla version of colabbench.
    Conclusion is I should restructure the action calls, 
    but leave the environment calls the same.
    """
    def __init__(self, seed, num_envs, split, hostname, port, model_id, task_type, max_steps):
        """Initialize the gym environment in this worker"""
        self.client = OpenAI(base_url=f"http://{hostname}:{port}/v1", api_key="EMPTY")
        # user_prompt_path = "../sweet_rl/prompts/human_simulator_code_prompt.txt"
        # with open(user_prompt_path, "r") as fb:
        #     self.human_prompt = fb.read() # problem_description, hidden_information, dialogue_history
        self.human_prompt = COLABBENCH_HUMAN_SIMULATOR_CODE_PROMPT
        self.max_steps = max_steps
        self.env = HumanInteractionEnv(self.client, self.human_prompt, model_id, max_steps=max_steps)
        self.has_won = False
        self.split = split
        input_path = f"../sweet_rl/data/backend_tasks/{split}.jsonl" 
        self.index = -1
        with open(input_path, "r") as fb:
            self.tasks = [json.loads(line) for line in fb]
            # task["problem_description"], task["ground_truth"]
        if self.split == 'test':
            self.index_ordering = (seed + num_envs * np.arange(len(self.tasks))) % len(self.tasks)
        else:
            self.index_ordering = np.random.default_rng(seed).permutation(len(self.tasks))

    def reset(self, seed_for_reset=None):
        """Reset the environment with optional seed"""
        if not self.split == 'test' and seed_for_reset is not None:
            self.index = int(np.random.default_rng(seed_for_reset).choice(self.index_ordering))
        else:
            self.index = (self.index + 1) % len(self.tasks)
        self.task = self.tasks[int(self.index_ordering[self.index])]
        self.env.reset(self.task["problem_description"], self.task["ground_truth"])
        self.has_won = False
        return self.task["problem_description"], {}
    
    def step(self, action):
        """Execute a step in the environment"""
        # breakpoint()
        tag, action, is_belief_generation_step, just_code = action
        if just_code:
            if is_belief_generation_step: 
                # if this is true, then the model failed to generate a valid parsable code segment.
                # the tag should be code, but may not be, but this should just result in poor reward, and not an exception.
                reward = 0
            else:
                reward = check_correctness(self.task['ground_truth'], action, self.task['test_cases'])
            return "Nothing", reward, True, {} # don't plan for info to be used in this just_code case.
        info = {"attempt": self.env.steps+2, "is_last_step": self.env.steps+2==self.max_steps, "steps_remaining": self.max_steps - self.env.steps - 2, "problem_description": self.task["problem_description"], "ground_truth": self.task["ground_truth"]}
        if is_belief_generation_step:
            info = {"attempt": self.env.steps+1, "is_last_step": self.env.steps+1==self.max_steps, "steps_remaining": self.max_steps - self.env.steps - 1, "problem_description": self.task["problem_description"], "ground_truth": self.task["ground_truth"]}
            # the belief generation step is checked only on the top level env manager. 
            # This is just a noop so other environments can step if they need to.
            return "", 0, False, info | {"won":self.has_won}
        if self.env.done:
            return "", 0, True, info | {"won":self.has_won}
        # obs, _, done, info = self.env.step(action)
        if tag == "code":
            action = "I WANT TO ANSWER:" + action
        dialog_history, _, done = self.env.step(action)
        if dialog_history is not None:
            env_response = dialog_history[-1]['content']
        else:
            env_response = "NOTHING!!!"
        # reward should only be given from the self.env.get_trajectory_score function
        info = info | {"won": False}
        if done:
            test_function = self.env.answer # this gets populated by environment when done is True for the first time.
            reward = check_correctness(self.task['ground_truth'], test_function, self.task['test_cases'])
            if reward != -1:
                info = info | {"won": True}
                self.has_won = True
        else:
            reward = 0
        return env_response, reward, done, info
    
    def get_ds_len(self):
        return len(self.tasks)
    # def get_target(self):
    #     return self.env.target_combination
    


class ColabBenchEnvs:
    """
    Ray-based parallel environment wrapper for gym cards environments.
    - env_id: combo lock environment ID
    - env_num: Number of distinct environments
    - group_n: Number of replicas within each group (commonly used for multiple copies with the same seed)
    - env_kwargs: Parameters needed to create a single gym.make(env_id)
    """

    def __init__(self,
                 split, 
                 hostname, 
                 port, 
                 model_id,
                 task_type,
                 max_steps,
                #  max_attempts,
                 seed=0,
                 env_num=1,
                 group_n=1,
                 is_train=True):
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()
        self.is_train = is_train
        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n

        np.random.seed(seed)
        self.reset_count = 0
        # Create Ray remote actors instead of processes
        self.workers = []
        seeds = np.arange(env_num).repeat(group_n)
        for i in range(self.num_processes):
            seed_i = seeds[i]
            worker = ColabBenchWorker.remote(
                seed_i,
                env_num, 
                split, 
                hostname, 
                port, 
                model_id,
                task_type,
                max_steps,
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

    def reset(self):
        """
        Perform reset in parallel.
        Different seeds will be assigned to each environment (or the same seed within a group).
        :return: (obs_list, info_list)
        """
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
    
    def get_ds_len(self):
        return ray.get(self.workers[0].get_ds_len.remote())
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


def build_colabbench_envs(split, 
                        hostname, 
                        port, 
                        model_id,
                        task_type,
                        max_steps,
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
    return ColabBenchEnvs(
        split, 
        hostname, 
        port, 
        model_id,
        task_type,
        max_steps,
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
#         env = ColabBench(combination_length, max_attempts, vocab)
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
