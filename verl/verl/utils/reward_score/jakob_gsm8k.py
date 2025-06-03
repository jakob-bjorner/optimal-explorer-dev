# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import re


def extract_solution(solution_str, method="strict"):
    assert method in ["strict", "flexible"]

    if method == "strict":
        # this also tests the formatting of the model
        solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
        if solution is None:
            final_answer = None
        else:
            final_answer = solution.group(0)
            final_answer = final_answer.split("#### ")[1].replace(",", "").replace("$", "")
    elif method == "flexible":
        answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            # no reward is there is no answer
            pass
        else:
            invalid_str = ["", "."]
            # find the last number that is not '.'
            for final_answer in reversed(answer):
                if final_answer not in invalid_str:
                    break
    return final_answer

def compute_score(*args, **kwargs):
# def compute_score(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    
    method = kwargs.get("method", "strict")
    solution_str = kwargs.get('solution_str')
    ground_truth = kwargs.get('ground_truth')
    format_score = kwargs.get('format_score', 0.0)
    score = kwargs.get('score', 1.0)
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    '''() {'data_source': 'openai/gsm8k', 'solution_str': "To determine how much money Tom has left to finish the month, we can follow these steps:\n\n1. Calculate how much money Tom spends in the first week.\n2. Subtract the amount spent in the first week from his initial allowance to find out how much he has left.\n3. Calculate how much money Tom spends in the second week.\n4. Subtract the amount spent in the second week from the amount left after the first week to find out how much he has left to finish the month.\n\nLet's go through these steps one by one.\n\n1. Tom initially receives $12 per month. We can calculate how much he has left after the first week by subtracting the amount spent in the first week from his initial allowance:\n   \\[\n   12 - \\frac{1}{3} \\times 12 = 12 - 4 = \\$8\n   \\]\n   So, after the first week, Tom has $8 left.\n\n2. In the second week, Tom spends a quarter of what he has left. Let's find out how much that means in terms of his allowance:\n   \\[\n   \\frac{1}{4} \\times 8 = 2\n   \\]\n   So, Tom spends $2 in", 'ground_truth': '6', 'extra_info': {'answer': 'The first week he spends: $12 * 1/3 = $<<12*1/3=4>>4.\nSo he has $12 - $4 = $<<12-4=8>>8 left.\nIn the second week, he spends $8 x 1/4 = $<<8*1/4=2>>2.\nSo Tom has $8 - $2 = $<<8-2=6>>6 left to finish the month.\n#### 6', 'index': 4570, 'question': 'Tom receives a $12 allowance per month. In the first week, he spends a third of it; in the second week, he spends a quarter of what he has left. How much money does he have left to finish the month?', 'split': 'train'}}'''
    answer = extract_solution(solution_str=solution_str, method=method)
    if answer is None:
        return 0
    else:
        if answer == ground_truth:
            return -score # jakob edit. make negative.
        else:
            return format_score
