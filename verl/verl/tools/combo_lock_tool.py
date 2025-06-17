# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import logging
import os
from typing import Any, Optional, Tuple
from uuid import uuid4


# special import requires installing the optimal_explorer package for the environments.
from optimal_explorer.mdps.combination_lock import CombinationLock



from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ComboLockTool(BaseTool):
    """A demo tool for calculating the reward of gsm8k.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function":{
                "name": "test_combination",
                "description": "see how close the combination is close to the true one",
                "strict": true,
                "parameters": {
                    "type": "object",
                    "required": [
                        "digit1",
                        "digit2",
                        "digit3"
                    ],
                    "properties": {
                        "digit1": {
                            "type": "number",
                            "minimum": 0,
                            "exclusiveMaximum": 10,
                            "description": "The first digit of the combination lock"
                        },
                        "digit2": {
                            "type": "number",
                            "minimum": 0,
                            "exclusiveMaximum": 10,
                            "description": "The second digit of the combination lock"
                        },
                        "digit3": {
                            "type": "number",
                            "minimum": 0,
                            "exclusiveMaximum": 10,
                            "description": "The third digit of the combination lock"
                        }
                    },
                    "additionalProperties": false
                }
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        env = CombinationLock() # new instance for each create function call.
        env.reset()
        env.target_combination = "".join(map(str, ground_truth))
        self._instance_dict[instance_id] = env
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        guess = str(parameters.get("digit1", "")) + str(parameters.get("digit2", "")) + str(parameters.get("digit3", ""))
        if not self._instance_dict[instance_id]._is_valid_guess(guess):
            return "invalid guess", 0.0, {"done": True}
        obs, reward, done, info = self._instance_dict[instance_id].step(guess)
        str_response_in_tool_call = "Feedback:"
        for i, (g, f) in enumerate(zip(guess, info['feedback'])):
            position = i + 1
            if f == 0: 
                str_response_in_tool_call += f"\n\t{g} is not in digit{position}, and is not in the lock"
            elif f == 1: 
                str_response_in_tool_call += f"\n\t{g} is not in digit{position}, but is in the lock"
            else: # f == 2 
                str_response_in_tool_call += f"\n\t{g} is in digit{position}!"
        return str_response_in_tool_call, 0.0, {"done": done}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return self._instance_dict[instance_id].get_trajectory_score()

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
