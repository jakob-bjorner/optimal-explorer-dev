import numpy as np
import random
from typing import List, Tuple, Optional
import os
import sys
import termios
import tty

def get_single_char():
    """Get a single character from stdin without requiring Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def play_wordle_interactive():
    mdp = sample_wordle_mdp()
    
    print("Welcome to Wordle! You have 6 guesses to find the word.", flush=True)
    print("Enter 'q' to quit at any time.\n", flush=True)
    
    while True:
        guess = input("Enter your guess: ").lower()
        if guess == 'q':  # Simplified quit condition
            print(f"Game over! The word was: {mdp.target_word}", flush=True)
            break
            
        if not mdp._is_valid_guess(guess):
            print(f"'{guess}' is not in the word list. Please try again.", flush=True)
            continue
            
        obs, reward, done, info = mdp.step(guess)
        feedback = info['feedback']
        feedback_str = ''.join(['⬜' if f == 0 else '🟨' if f == 1 else '🟩' for f in feedback])
        print(f"Guess #{mdp.current_guess}: {guess} -> {feedback_str}", flush=True)
        if done:
            if reward == 1.0:
                print("\nCongratulations! You won! 🎉", flush=True)
            else:
                print(f"\nGame over! The word was: {mdp.target_word}", flush=True)
            break
        if mdp.current_guess >= mdp.max_guesses:
            print(f"\nGame over! The word was: {mdp.target_word}", flush=True)
            break

class Wordle:
    def __init__(self, word_length: int = 5, max_guesses: int = 6):
        """
        Initialize the Wordle MDP.
        
        Args:
            word_length: Length of words in the game (default: 5)
            max_guesses: Maximum number of allowed guesses (default: 6)
        """
        self.word_length = word_length
        self.max_guesses = max_guesses
        self.current_guess = 0
        self.target_word = None
        self.possible_words = self._load_words()
        self.observation_space = self._create_observation_space()
        self.action_space = self._create_action_space()
        
    def _load_words(self) -> List[str]:
        """Load words from the words file."""
        words_file = os.path.join(os.path.dirname(__file__), 'wordle_data', 'words.txt')
        if not os.path.exists(words_file):
            # Create a default word list if file doesn't exist
            default_words = ['apple', 'beach', 'cloud', 'dance', 'eagle', 
                           'flame', 'ghost', 'heart', 'ivory', 'joker']
            with open(words_file, 'w') as f:
                f.write('\n'.join(default_words))
            return default_words
        
        with open(words_file, 'r') as f:
            return [word.strip().lower() for word in f.readlines()]
    
    def _create_observation_space(self) -> dict:
        """Create the observation space for the MDP."""
        return {
            'current_guess': (0, self.max_guesses),
            'feedback_history': [(0, 3) for _ in range(self.max_guesses)],  # 0: ⬜, 1: 🟨, 2: 🟩
            'guess_history': ['' for _ in range(self.max_guesses)]
        }
    
    def _create_action_space(self) -> List[str]:
        """Create the action space (all possible valid words)."""
        return self.possible_words
    
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
            
        self.current_guess = 0
        self.target_word = random.choice(self.possible_words)
        return self._get_observation()
    
    def step(self, action: str) -> Tuple[dict, float, bool, dict]:
        """
        Take a step in the environment by making a guess.
        
        Args:
            action: The word being guessed
            
        Returns:
            observation: Current state observation
            reward: Reward for the action
            done: Whether the episode is finished
            info: Additional information
        """
        if not self._is_valid_guess(action):
            return self._get_observation(), -1.0, True, {'error': 'Invalid guess'}
        
        feedback = self._get_feedback(action)
        self.current_guess += 1
        
        # Calculate reward
        if action == self.target_word:
            reward = 1.0
            done = True
        elif self.current_guess >= self.max_guesses:
            reward = 0.0
            done = True
        else:
            reward = 0.0
            done = False
            
        return self._get_observation(), reward, done, {'feedback': feedback}
    
    def _is_valid_guess(self, guess: str) -> bool:
        """Check if a guess is valid."""
        return guess.lower() in self.possible_words
    
    def _get_feedback(self, guess: str) -> List[int]:
        """
        Get feedback for a guess.
        Returns a list of integers: 0 (⬜), 1 (🟨), 2 (🟩)
        """
        feedback = []
        target_chars = list(self.target_word)
        guess_chars = list(guess.lower())
        
        # First pass: mark correct positions (🟩)
        for i in range(len(guess_chars)):
            if guess_chars[i] == target_chars[i]:
                feedback.append(2)  # 🟩
                target_chars[i] = None  # Mark as used
            else:
                feedback.append(0)  # Temporary ⬜
        
        # Second pass: mark correct letters in wrong positions (🟨)
        for i in range(len(guess_chars)):
            if feedback[i] != 2:  # If not already marked as correct
                if guess_chars[i] in target_chars:
                    feedback[i] = 1  # 🟨
                    # Remove the first occurrence of this letter
                    target_chars[target_chars.index(guess_chars[i])] = None
        
        return feedback
    
    def _get_observation(self) -> dict:
        """Get the current observation."""
        return {
            'current_guess': self.current_guess,
            'feedback_history': [self._get_feedback(guess) if guess else [0] * self.word_length 
                               for guess in self.guess_history],
            'guess_history': self.guess_history
        }
    
    @property
    def guess_history(self) -> List[str]:
        """Get the history of guesses made so far."""
        return ['' for _ in range(self.max_guesses)]  # Placeholder - implement actual history tracking
    
    def render(self) -> None:
        """Render the current state of the game."""
        print(f"\nTarget word: {self.target_word}", flush=True)
        print(f"Current guess: {self.current_guess}/{self.max_guesses}", flush=True)
        for i, guess in enumerate(self.guess_history):
            if guess:
                feedback = self._get_feedback(guess)
                feedback_str = ''.join(['⬜' if f == 0 else '🟨' if f == 1 else '🟩' for f in feedback])
                print(f"Guess {i+1}: {guess} -> {feedback_str}", flush=True)

def sample_wordle_mdp(seed: Optional[int] = None) -> Wordle:
    """
    Sample a Wordle MDP instance with a random target word.
    
    Args:
        seed: Random seed for reproducibility
        
    Returns:
        Wordle MDP instance
    """
    mdp = Wordle()
    mdp.reset(seed)
    return mdp


