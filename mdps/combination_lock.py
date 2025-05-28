import numpy as np
import random
from typing import List, Tuple, Optional
import itertools

def play_combination_lock_interactive():
    mdp = sample_combination_lock_mdp()
    
    print("Welcome to Combination Lock! You have 8 attempts to find the 3-digit combination.", flush=True)
    print("Enter 'q' to quit at any time.\n", flush=True)
    
    while True:
        guess = input("Enter your 3-digit guess (e.g. 123): ").lower()
        if guess == 'q':
            print(f"Game over! The combination was: {''.join(map(str, mdp.target_combination))}", flush=True)
            break
            
        if not mdp._is_valid_guess(guess):
            print(f"'{guess}' is not a valid 3-digit combination. Please try again.", flush=True)
            continue
            
        obs, reward, done, info = mdp.step(guess)
        feedback = info['feedback']
        feedback_str = ''.join(['⬜' if f == 0 else '🟨' if f == 1 else '🟩' for f in feedback])
        print(f"Attempt #{mdp.current_attempt}: {guess} -> {feedback_str}", flush=True)
        if done:
            if reward == 1.0:
                print("\nCongratulations! You unlocked it! 🎉", flush=True)
            else:
                print(f"\nGame over! The combination was: {''.join(map(str, mdp.target_combination))}", flush=True)
            break
        if mdp.current_attempt >= mdp.max_attempts:
            print(f"\nGame over! The combination was: {''.join(map(str, mdp.target_combination))}", flush=True)
            break

class CombinationLock:
    def __init__(self, combination_length: int = 3, max_attempts: int = 8):
        """
        Initialize the Combination Lock MDP.
        
        Args:
            combination_length: Length of the combination (default: 3)
            max_attempts: Maximum number of allowed attempts (default: 8)
        """
        self.combination_length = combination_length
        self.max_attempts = max_attempts
        self.current_attempt = 0
        self.target_combination = None
        self.possible_combinations = self._generate_combinations()
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()
        self.guess_history = []
        
    def _generate_combinations(self) -> List[str]:
        """Generate all possible 3-digit combinations with distinct digits."""
        digits = list(range(10))  # 0-9
        combinations = []
        for combo in itertools.permutations(digits, self.combination_length):
            combinations.append(''.join(map(str, combo)))
        return combinations
    
    def _create_observation_space(self) -> dict:
        """Create the observation space for the MDP."""
        return {
            'current_attempt': (0, self.max_attempts),
            'feedback_history': [(0, 3) for _ in range(self.max_attempts)],  # 0: Not in code, 1: Wrong position, 2: Correct position
            'guess_history': ['' for _ in range(self.max_attempts)]
        }
    
    def _create_action_space(self) -> List[str]:
        """Create the action space (all possible valid combinations)."""
        return self.possible_combinations
    
    def reset(self, seed: Optional[int] = None) -> dict:
        """
        Reset the environment to start a new game.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Initial observation
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.current_attempt = 0
        self.guess_history = []
        self.target_combination = random.choice(self.possible_combinations)
        return self._get_observation()
    
    def step(self, action: str) -> Tuple[dict, float, bool, dict]:
        """
        Take a step in the environment by making a guess.
        
        Args:
            action: The combination being guessed
            
        Returns:
            observation: Current state observation
            reward: Reward for the action
            done: Whether the episode is finished
            info: Additional information
        """
        if not self._is_valid_guess(action):
            return self._get_observation(), -1.0, True, {'error': 'Invalid guess'}
        
        feedback = self._get_feedback(action)
        self.current_attempt += 1
        self.guess_history.append(action)
        
        # Calculate reward
        if action == self.target_combination:
            reward = 1.0
            done = True
        elif self.current_attempt >= self.max_attempts:
            reward = 0.0
            done = True
        else:
            reward = 0.0
            done = False
            
        return self._get_observation(), reward, done, {'feedback': feedback}
    
    def _is_valid_guess(self, guess: str) -> bool:
        """Check if a guess is valid."""
        return (len(guess) == self.combination_length and 
                guess.isdigit() and 
                len(set(guess)) == self.combination_length)  # All digits must be distinct
    
    def _get_feedback(self, guess: str) -> List[int]:
        """
        Get feedback for a guess.
        Returns a list of integers: 0 (Not in code), 1 (Wrong position), 2 (Correct position)
        """
        feedback = []
        target_digits = list(self.target_combination)
        guess_digits = list(guess)
        
        # First pass: mark correct positions
        for i in range(len(guess_digits)):
            if guess_digits[i] == target_digits[i]:
                feedback.append(2)  # Correct position
                target_digits[i] = None  # Mark as used
            else:
                feedback.append(0)  # Temporary "Not in code"
        
        # Second pass: mark correct digits in wrong positions
        for i in range(len(guess_digits)):
            if feedback[i] != 2:  # If not already marked as correct position
                if guess_digits[i] in target_digits:
                    feedback[i] = 1  # Wrong position
                    # Remove the first occurrence of this digit
                    target_digits[target_digits.index(guess_digits[i])] = None
        
        return feedback
    
    def _get_observation(self) -> dict:
        """Get the current observation."""
        return {
            'current_attempt': self.current_attempt,
            'feedback_history': [self._get_feedback(guess) if guess else [0] * self.combination_length 
                               for guess in self.guess_history],
            'guess_history': self.guess_history + [''] * (self.max_attempts - len(self.guess_history))
        }
    
    def render(self) -> None:
        """Render the current state of the game."""
        print(f"\nTarget combination: {self.target_combination}", flush=True)
        print(f"Current attempt: {self.current_attempt}/{self.max_attempts}", flush=True)
        for i, guess in enumerate(self.guess_history):
            if guess:
                feedback = self._get_feedback(guess)
                feedback_str = ''.join(['⬜' if f == 0 else '🟨' if f == 1 else '🟩' for f in feedback])
                print(f"Attempt {i+1}: {guess} -> {feedback_str}", flush=True)

def sample_combination_lock_mdp(seed: Optional[int] = None) -> CombinationLock:
    """
    Sample a Combination Lock MDP instance with a random target combination.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        CombinationLock MDP instance
    """
    mdp = CombinationLock()
    mdp.reset(seed)
    return mdp
