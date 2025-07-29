#!/usr/bin/env python3
"""
Configuration helper for paprika interactions.
This makes it easy to switch between different games by changing a single variable.
"""

import os
import sys
from typing import Dict, Any, List, Tuple

# Add the paprika and verl paths to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'paprika'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'verl_submodule'))

from omegaconf import OmegaConf, DictConfig


class PaprikaConfigHelper:
    """
    Helper class to create configurations for different paprika games.
    """
    
    # Available game environments and their default configurations
    GAME_CONFIGS = {
        "twenty_questions": {
            "agent_model": "gpt-4o-mini",
            "env_model": "gpt-4o-mini", 
            "judge_model": "gpt-4o-mini",
            "belief_style": "basic"
        },
        "guess_my_city": {
            "agent_model": "gpt-4o-mini",
            "env_model": "gpt-4o-mini",
            "judge_model": "gpt-4o-mini", 
            "belief_style": "basic"
        },
        "murder_mystery": {
            "agent_model": "gpt-4o-mini",
            "env_model": "gpt-4o-mini",
            "judge_model": "gpt-4o-mini",
            "belief_style": "basic"
        },
        "customer_service": {
            "agent_model": "gpt-4o-mini", 
            "env_model": "gpt-4o-mini",
            "judge_model": "gpt-4o-mini",
            "belief_style": "basic"
        },
        "wordle": {
            "agent_model": "gpt-4o-mini",
            "env_model": "wordle",
            "judge_model": "wordle",
            "belief_style": "basic"
        },
        "cellular_automata": {
            "agent_model": "gpt-4o-mini",
            "env_model": "cellular_automata", 
            "judge_model": "cellular_automata",
            "belief_style": "basic"
        },
        "mastermind": {
            "agent_model": "gpt-4o-mini",
            "env_model": "mastermind",
            "judge_model": "mastermind", 
            "belief_style": "basic"
        },
        "battleship": {
            "agent_model": "gpt-4o-mini",
            "env_model": "battleship",
            "judge_model": "battleship",
            "belief_style": "basic"
        },
        "minesweeper": {
            "agent_model": "gpt-4o-mini",
            "env_model": "minesweeper",
            "judge_model": "minesweeper",
            "belief_style": "basic"
        },
        "bandit_bai_fixed_budget": {
            "agent_model": "gpt-4o-mini",
            "env_model": "bandit_bai_fixed_budget",
            "judge_model": "bandit_bai_fixed_budget",
            "belief_style": "basic"
        }
    }
    
    @classmethod
    def get_available_games(cls) -> List[str]:
        """Get list of available game environments"""
        return list(cls.GAME_CONFIGS.keys())
    
    @classmethod
    def create_config(cls, game_env_name: str, **overrides) -> Dict[str, Any]:
        """
        Create a configuration for a specific game.
        
        Args:
            game_env_name: The paprika game environment to use
            **overrides: Any configuration overrides
            
        Returns:
            Configuration dictionary
        """
        if game_env_name not in cls.GAME_CONFIGS:
            available = cls.get_available_games()
            raise ValueError(f"Game {game_env_name} not supported. Available: {available}")
        
        # Get default config for this game
        default_config = cls.GAME_CONFIGS[game_env_name].copy()
        
        # Apply overrides
        default_config.update(overrides)
        
        # Create the full configuration
        agent_config = {
            "model_type": "openai_api_models",
            "model_name": default_config["agent_model"],
            "model_max_length": 20000
        }
        
        env_config = {
            "model_type": "openai_api_models",
            "model_name": default_config["env_model"],
            "model_max_length": 20000
        }
        
        judge_config = {
            "model_type": "openai_api_models",
            "model_name": default_config["judge_model"],
            "model_max_length": 1000
        }
        
        belief_config = {
            "style": default_config["belief_style"]
        }
        
        return {
            "game_env_name": game_env_name,
            "agent_config": agent_config,
            "env_config": env_config,
            "judge_config": judge_config,
            "belief_config": belief_config
        }
    
    @classmethod
    def create_config_from_paprika_style(cls, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create paprika interaction config from the style used in the notebook.
        This mimics the create_config function from the notebook.
        
        Args:
            config_dict: Dictionary with paprika-style configuration
            
        Returns:
            Configuration dictionary for paprika interaction
        """
        game_env_name = config_dict.get("game_env", {}).get("environment_name", "wordle")
        agent_config = config_dict.get("agent", {})
        env_config = config_dict.get("env", {})
        judge_config = config_dict.get("judge", {})
        belief_config = config_dict.get("belief_config", {"style": "basic"})
        
        return {
            "game_env_name": game_env_name,
            "agent_config": agent_config,
            "env_config": env_config,
            "judge_config": judge_config,
            "belief_config": belief_config
        }


# Example usage functions
def example_simple_usage():
    """Example of simple usage - just change the game name"""
    
    # Simply change this variable to switch games
    GAME_NAME = "wordle"  # Change this to any available game
    
    config = PaprikaConfigHelper.create_config(GAME_NAME)
    print(f"Configuration for {GAME_NAME}:")
    print(config)
    return config


def example_with_overrides():
    """Example with configuration overrides"""
    
    GAME_NAME = "twenty_questions"
    
    # Override some settings
    config = PaprikaConfigHelper.create_config(
        GAME_NAME,
        agent_model="gpt-4",
        belief_style="belief_no_inst"
    )
    
    print(f"Configuration for {GAME_NAME} with overrides:")
    print(config)
    return config


def example_from_paprika_style():
    """Example converting from paprika-style config"""
    
    # This mimics the config from the notebook
    paprika_style_config = {
        "game_env": {
            "environment_name": "wordle"
        },
        "agent": {
            "model_type": "openai_api_models",
            "model_name": "gpt-4o-mini"
        },
        "env": {
            "model_type": "openai_api_models", 
            "model_name": "wordle"
        },
        "judge": {
            "model_type": "openai_api_models",
            "model_name": "wordle"
        },
        "belief_config": {
            "style": "basic"
        }
    }
    
    config = PaprikaConfigHelper.create_config_from_paprika_style(paprika_style_config)
    print("Configuration converted from paprika style:")
    print(config)
    return config


if __name__ == "__main__":
    print("Paprika Config Helper Examples")
    print("=" * 40)
    
    print("\n1. Simple usage:")
    example_simple_usage()
    
    print("\n2. With overrides:")
    example_with_overrides()
    
    print("\n3. From paprika style:")
    example_from_paprika_style()
    
    print(f"\nAvailable games: {PaprikaConfigHelper.get_available_games()}") 