from typing import List

from typing import List
import re
from copy import deepcopy

def colabbench_projection(actions: List[str], generate_belief: List[bool]):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """
        # for prediction in predictions:
        #     if isinstance(prediction, str):
        #         pattern = r'<(answer|search)>(.*?)</\1>'
        #         match = re.search(pattern, prediction, re.DOTALL)
        #         if match:
        #             content = match.group(2).strip()
        #             action = match.group(1).strip()

    tags = [""] * len(actions)
    valids = [0] * len(actions)
    actions = deepcopy(actions)
    actions_og = deepcopy(actions)
    def extract_first_in_tag(tag, str_to_extract):
        pattern = '<('+tag+r')>(.*?)</\1>'
        match = re.search(pattern, str_to_extract, re.DOTALL)
        if match:
            tag = match.group(1).strip()
            extracted_content = match.group(2).strip()
            return tag, extracted_content
        else:
            return False

    for i in range(len(actions)):
        if generate_belief[i]:
            match = extract_first_in_tag("belief", actions_og[i])
            if match:
                tags[i] = match[0]
                actions[i] = match[1]
                valids[i] = 1
            else:
                actions[i] = actions_og[i][-30:]
                valids[i] = 0
                continue
        else:
            match = extract_first_in_tag('ask', actions_og[i])
            if match:
                tags[i] = match[0]
                actions[i] = match[1]
                valids[i] = 1
            else:
                match = extract_first_in_tag('code', actions_og[i])
                if match:
                    tags[i] = match[0]
                    actions[i] = match[1]
                    valids[i] = 1
                else:
                    actions[i] = actions_og[i][-30:]
                    valids[i] = 0
                    continue
    return tags, actions, valids
