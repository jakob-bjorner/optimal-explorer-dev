import numpy as np
import random
from typing import List, Tuple, Optional, Any, Dict

def play_bandit_best_arm_selection_interactive():
    env = sample_bandit_best_arm_selection_env()
    print(f"Welcome to Bandit Best Arm Selection! You have {env.max_steps} steps to explore the arms.", flush=True)
    print(f"Arms: {', '.join(env.arm_names)}", flush=True)
    print("Enter 'q' to quit at any time.\n", flush=True)
    
    while not env.done:
        if env.current_step < env.max_steps:
            guess = input(f"Step {env.current_step+1}/{env.max_steps} - Choose an arm {env.arm_names}: ").strip().lower()
            if guess == 'q':
                print("Game over!", flush=True)
                break
            if not env._is_valid_arm(guess):
                print(f"'{guess}' is not a valid arm. Please try again.", flush=True)
                continue
            obs, reward, done, info = env.step(guess)
            print(f"You chose '{guess}' and received reward {info['reward']}", flush=True)
        else:
            print("You have exhausted your budget for trying out different choices.", flush=True)
            final_guess = input(f"Which arm do you think is best? {env.arm_names}: ").strip().lower()
            obs, reward, done, info = env.step(final_guess)
            print(f"Your final answer: '{final_guess}'", flush=True)
            if info['correct']:
                print("Correct! 🎉 You identified the best arm.", flush=True)
            else:
                print(f"Incorrect. The best arm was '{env.best_arm_name}'.", flush=True)
            break

class BanditBestArmSelection:
    def __init__(self, arm_names: Optional[List[str]] = None, max_steps: int = 20, seed: Optional[int] = None):
        """
        Initialize the Bandit Best Arm Selection environment.
        Args:
            arm_names: List of arm names (default: 5 colored arms)
            max_steps: Number of exploration steps before final answer
            seed: Random seed for reproducibility
        """
        if arm_names is None:
            arm_names = ['blue', 'green', 'red', 'yellow', 'purple']
        self.arm_names = arm_names
        self.k = len(arm_names)
        self.max_steps = max_steps
        self.current_step = 0
        self.done = False
        self.arm_means = None
        self.best_arm = None
        self.best_arm_name = None
        self.history = []  # List of (arm, reward)
        self.final_answer = None
        self._rng = np.random.RandomState(seed)
        self.reset(seed)

    def _sample_arm_means(self):
        """Sample the mean rewards for each arm as described in the task."""
        eps = self._rng.uniform(0.1, 0.2)
        best_arm = self._rng.randint(self.k)
        means = np.zeros(self.k)
        means[best_arm] = 0.5 + eps
        for i in range(self.k):
            if i != best_arm:
                means[i] = self._rng.uniform(0, 0.5 - eps)
        return means, best_arm

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Reset the environment for a new episode.
        Args:
            seed: Optional random seed
        Returns:
            Initial observation
        """
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self.arm_means, self.best_arm = self._sample_arm_means()
        self.best_arm_name = self.arm_names[self.best_arm]
        self.current_step = 0
        self.done = False
        self.history = []
        self.final_answer = None
        return self._get_observation()

    def _is_valid_arm(self, arm: str) -> bool:
        return arm in self.arm_names

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        Args:
            action: The chosen arm (for exploration) or final answer (after max_steps)
        Returns:
            observation, reward, done, info
        """
        if self.done:
            raise Exception("Episode already terminated.")
        if self.current_step < self.max_steps:
            if not self._is_valid_arm(action):
                return self._get_observation(), -1.0, True, {'error': 'Invalid arm'}
            arm_idx = self.arm_names.index(action)
            reward = int(self._rng.rand() < self.arm_means[arm_idx])
            self.history.append((action, reward))
            self.current_step += 1
            obs = self._get_observation()
            done = False
            info = {'reward': reward}
            if self.current_step == self.max_steps:
                obs['final_answer_required'] = True
            return obs, 0.0, done, info
        else:
            # Final answer step
            self.final_answer = action
            correct = (action == self.best_arm_name)
            reward = 1.0 if correct else 0.0
            self.done = True
            obs = self._get_observation()
            info = {'correct': correct, 'best_arm': self.best_arm_name}
            return obs, reward, True, info

    def _get_observation(self) -> Dict[str, Any]:
        return {
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'history': list(self.history),
            'arm_names': list(self.arm_names),
            'final_answer': self.final_answer,
        }

    def render(self) -> None:
        print(f"\nArms: {', '.join(self.arm_names)}", flush=True)
        print(f"Current step: {self.current_step}/{self.max_steps}", flush=True)
        for i, (arm, reward) in enumerate(self.history):
            print(f"Step {i+1}: {arm} -> reward {reward}", flush=True)
        if self.final_answer is not None:
            print(f"Final answer: {self.final_answer}", flush=True)
            print(f"Best arm: {self.best_arm_name}", flush=True)

    def get_trajectory_score(self) -> float:
        # Score is 1 if final answer is correct, else 0
        if self.final_answer == self.best_arm_name:
            return 1.0
        else:
            return 0.0

    def get_trajectory_info(self):
        # Return number of times each arm was pulled
        arm_counts = {arm: 0 for arm in self.arm_names}
        for arm, _ in self.history:
            arm_counts[arm] += 1
        return {
            'arm_counts': arm_counts,
            'final_answer': self.final_answer,
            'best_arm': self.best_arm_name
        }

def sample_bandit_best_arm_selection_env(seed: Optional[int] = None) -> BanditBestArmSelection:
    """
    Sample a Bandit Best Arm Selection environment instance with random arm means.
    Args:
        seed: Random seed for reproducibility
    Returns:
        BanditBestArmSelection environment instance
    """
    env = BanditBestArmSelection(seed=seed)
    env.reset(seed)
    return env

class MultiArmedBandits:
    def __init__(
        self,
        arm_names: Optional[List[str]] = None,
        num_arms: int = 5,
        max_steps: int = 50,
        reward_type: str = 'gaussian',  # 'gaussian' or 'bernoulli'
        noise_level: str = 'low',       # 'low', 'medium', 'high'
        scenario_description: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the Multi-Armed Bandit environment.
        Args:
            arm_names: List of arm names (default: numeric labels)
            num_arms: Number of arms (default: 5)
            max_steps: Number of interaction steps per episode (default: 50)
            reward_type: 'gaussian' or 'bernoulli'
            noise_level: 'low', 'medium', or 'high' (affects Gaussian sigma)
            scenario_description: Optional textual description of the scenario
            seed: Random seed for reproducibility
        """
        if arm_names is None:
            arm_names = [str(i) for i in range(num_arms)]
        self.arm_names = arm_names
        self.k = len(arm_names)
        self.max_steps = max_steps
        self.reward_type = reward_type.lower()
        self.noise_level = noise_level.lower()
        self.scenario_description = scenario_description or self._default_description()
        self.current_step = 0
        self.done = False
        self.arm_means = None
        self.arm_sigmas = None
        self.history = []  # List of (arm, reward)
        self._rng = np.random.RandomState(seed)

        self.reset(seed)

    def _default_description(self):
        return (
            f"You are a bandit algorithm and interact with {self.k} arms labeled {', '.join(self.arm_names)}. "
            f"Each arm is associated with a {self.reward_type.capitalize()} distribution with a fixed but unknown mean; "
            f"the means for the arms could be different. For either arm, when you use it, you will get a reward that is "
            f"sampled from the arm's associated distribution. You have {self.max_steps} time steps and, on each time step, "
            f"you MUST choose one of the arms and receive the reward. Your goal is to maximize the total reward."
        )

    def _get_sigma(self):
        if self.reward_type == 'gaussian':
            if self.noise_level == 'low':
                return 0.1
            elif self.noise_level == 'medium':
                return 1.0
            elif self.noise_level == 'high':
                return 3.0
            else:
                raise ValueError(f"Unknown noise level: {self.noise_level}")
        return None

    def _sample_arm_means(self):
        if self.reward_type == 'gaussian':
            means = self._rng.uniform(0, 1, self.k)
            sigma = self._get_sigma()
            sigmas = np.full(self.k, sigma)
            return means, sigmas
        elif self.reward_type == 'bernoulli':
            means = self._rng.uniform(0, 1, self.k)
            return means, None
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self.arm_means, self.arm_sigmas = self._sample_arm_means()
        self.current_step = 0
        self.done = False
        self.history = []
        return self._get_observation()

    def _is_valid_arm(self, arm: str) -> bool:
        return arm in self.arm_names

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.done:
            raise Exception("Episode already terminated.")
        if not self._is_valid_arm(action):
            return self._get_observation(), -1.0, True, {'error': 'Invalid arm'}
        arm_idx = self.arm_names.index(action)
        if self.reward_type == 'gaussian':
            mu = self.arm_means[arm_idx]
            sigma = self.arm_sigmas[arm_idx]
            reward = self._rng.normal(mu, sigma)
        elif self.reward_type == 'bernoulli':
            mu = self.arm_means[arm_idx]
            reward = int(self._rng.rand() < mu)
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
        self.history.append((action, reward))
        self.current_step += 1
        obs = self._get_observation()
        done = self.current_step >= self.max_steps
        self.done = done
        info = {'reward': reward}
        return obs, reward, done, info

    def _get_observation(self) -> Dict[str, Any]:
        return {
            'current_step': self.current_step,
            'max_steps': self.max_steps,
            'history': list(self.history),
            'arm_names': list(self.arm_names),
        }

    def render(self) -> None:
        print(f"\nArms: {', '.join(self.arm_names)}", flush=True)
        print(f"Current step: {self.current_step}/{self.max_steps}", flush=True)
        for i, (arm, reward) in enumerate(self.history):
            print(f"Step {i+1}: {arm} -> reward {reward}", flush=True)

    def get_trajectory_score(self) -> float:
        # Total reward accumulated
        return float(sum(r for _, r in self.history))

    def get_trajectory_info(self):
        arm_counts = {arm: 0 for arm in self.arm_names}
        for arm, _ in self.history:
            arm_counts[arm] += 1
        return {
            'arm_counts': arm_counts,
            'total_reward': self.get_trajectory_score(),
        }
