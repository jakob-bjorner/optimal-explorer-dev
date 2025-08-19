from typing import List

from typing import List
import re
from copy import deepcopy

def combolock_projection(actions: List[str], generate_belief: List[bool]):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """

    valids = [0] * len(actions)
    actions = deepcopy(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        # actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        if generate_belief[i]:
            start_tag = '<belief>'
            end_tag = '</belief>'
        else:
            start_tag = "<action>"
            end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = actions[i][-30:]  
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
            # then we check if it is a valid action according to the mdp:
            if not generate_belief[i]:
                vocab = "0123456789"
                guess = ''.join(c for c in extracted_action if c in vocab)
                if not (len(guess) == 3 and 
                    all(c in vocab for c in guess) and 
                    len(set(guess)) == 3): 
                    # this is invalid
                    actions[i] = extracted_action
                    valids[i] = 0
                else:
                    # this is valid
                    actions[i] = guess
                    valids[i] = 1
            else:
                actions[i] = extracted_action
                valids[i] = 1

        except:
            actions[i] = actions[i][-30:]

        # jakob removing these below checks for consistency with other env.

        # # check <think>...</think>
        # think_start_idx = original_str.find("<think>")
        # think_end_idx = original_str.find("</think>")
        # if think_start_idx == -1 or think_end_idx == -1:
        #     valids[i] = 0

        # # check if contains any Chinese characters
        # if re.search(r'[\u4e00-\u9fff]', original_str):
        #     valids[i] = 0

    return actions, valids
