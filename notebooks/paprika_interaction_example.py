#!/usr/bin/env python3
"""
Example script showing how to use the PaprikaInteraction class with different game configurations.
This demonstrates how to switch between different paprika games by changing config variables.
"""

import asyncio
import os
import sys
from typing import Dict, Any

# Add the paprika and verl paths to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'paprika'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'verl_submodule'))

from omegaconf import OmegaConf, DictConfig
from verl.interactions.paprika_interaction import PaprikaInteraction


def create_paprika_config(
    game_env_name: str = "wordle",
    agent_model: str = "gpt-4o-mini",
    env_model: str = "wordle",
    judge_model: str = "wordle",
    belief_style: str = "basic"
) -> Dict[str, Any]:
    """
    Create a configuration for paprika interaction.
    
    Args:
        game_env_name: The paprika game environment to use
        agent_model: Model for the agent
        env_model: Model for the environment
        judge_model: Model for the judge
        belief_style: Belief update style
    
    Returns:
        Configuration dictionary
    """
    
    # Available game environments
    available_games = [
        "twenty_questions",
        "guess_my_city", 
        "murder_mystery",
        "customer_service",
        "wordle",
        "cellular_automata",
        "mastermind",
        "battleship",
        "minesweeper",
        "bandit_bai_fixed_budget"
    ]
    
    if game_env_name not in available_games:
        raise ValueError(f"Game {game_env_name} not supported. Available: {available_games}")
    
    # Agent configuration
    agent_config = {
        "model_type": "openai_api_models",
        "model_name": agent_model,
        "model_max_length": 20000
    }
    
    # Environment configuration
    env_config = {
        "model_type": "openai_api_models", 
        "model_name": env_model,
        "model_max_length": 20000
    }
    
    # Judge configuration
    judge_config = {
        "model_type": "openai_api_models",
        "model_name": judge_model,
        "model_max_length": 1000
    }
    
    # Belief configuration
    belief_config = {
        "style": belief_style  # "basic", "belief_no_inst", or "none"
    }
    
    return {
        "game_env_name": game_env_name,
        "agent_config": agent_config,
        "env_config": env_config,
        "judge_config": judge_config,
        "belief_config": belief_config
    }


async def run_paprika_interaction_example():
    """Example of running a paprika interaction"""
    
    # Create interaction
    interaction = PaprikaInteraction(config={"name": "paprika_interaction"})
    
    # Example configurations for different games
    game_configs = [
        ("wordle", "gpt-4o-mini", "wordle", "wordle"),
        ("twenty_questions", "gpt-4o-mini", "gpt-4o-mini", "gpt-4o-mini"),
        ("mastermind", "gpt-4o-mini", "mastermind", "mastermind"),
    ]
    
    for game_env, agent_model, env_model, judge_model in game_configs:
        print(f"\n=== Running {game_env} game ===")
        
        # Create config for this game
        config = create_paprika_config(
            game_env_name=game_env,
            agent_model=agent_model,
            env_model=env_model,
            judge_model=judge_model,
            belief_style="basic"
        )
        
        # Start interaction
        instance_id = await interaction.start_interaction(
            instance_id=None,
            **config
        )
        
        print(f"Started interaction with instance_id: {instance_id}")
        
        # Example messages (in practice, these would come from the LLM)
        messages = [
            {"role": "user", "content": "Let me start the game"},
            {"role": "assistant", "content": "<action>I will make my first move in the game</action>"}
        ]
        
        # Generate response
        should_terminate, response, score, additional_data = await interaction.generate_response(
            instance_id=instance_id,
            messages=messages
        )
        
        print(f"Response: {response}")
        print(f"Score: {score}")
        print(f"Should terminate: {should_terminate}")
        
        # Calculate final score
        final_score = await interaction.calculate_score(instance_id=instance_id)
        print(f"Final score: {final_score}")
        
        # Get trajectory info
        trajectory_info = interaction.get_trajectory_info(instance_id=instance_id)
        print(f"Trajectory info: {trajectory_info}")
        
        # Finalize interaction
        await interaction.finalize_interaction(instance_id=instance_id)
        print(f"Finalized interaction for {game_env}")


def main():
    """Main function to run the example"""
    print("Paprika Interaction Example")
    print("=" * 50)
    
    # Run the async example
    asyncio.run(run_paprika_interaction_example())
    
    print("\nExample completed!")


if __name__ == "__main__":
    main() 