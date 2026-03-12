# COMBO_BELIEF_GENERATION_FAILURE_MSG="Belief generation failed to parse. The belief must be contained within <belief> ... </belief> tags. Try again."
NQHOTPOTQA_BELIEF_GENERATION_FAILURE_MSG="Belief generation failed to parse. The belief must be contained within <belief> ... </belief> tags. Try again."
# COMBO_NO_PRIOR_BELIEF_MESSAGE="No prior belief."
NQHOTPOTQA_NO_PRIOR_BELIEF_MESSAGE="No prior belief."
# COMBO_INVALID_ACTION_MESSAGE="invalid action"
NQHOTPOTQA_INVALID_ACTION_MESSAGE="invalid action"

# COMBO_AGENT_FIRST_MESSAGE="""You will determine the correct combination of characters at [Position 1, Position 2, Position 3] in a 3-character combination lock through iterative reasoning and queries.
# All 3 characters are unique.
# The set of valid characters are as follows: {vocab_list}
# Each action is a query of the form ['char 1', 'char 2', 'char 3'].
# Each time you query a combination, you will get feedback from the user about each character: either not in the combination, in the combination but in a different position, or in the combination and in the right position.
# You can make up to {max_attempts} queries.
# Your goal is to find the correct combination in the least number of queries."""
NQHOTPOTQA_AGENT_FIRST_MESSAGE="""You will answer multiple complex questions using iterative reasoning, and web search.
When taking an action, choose from one of the following actions:
   - If any question remains unanswered, issue a single query for one question inside <search> ... </search>. The query should consist of keywords or a short phrase. Only search one question at a time.
   - If all questions are answered, provide the final answers—separated by semicolons—within <answer> answer1; answer2; ... </answer>. The answers must be concise, contain only essential words, and avoid any explanations.

Important:
- Do not search multiple queries or questions simultaneously.

Answer the following questions: {question}"""

NQHOTPOTQA_FULL_AGENT_FIRST_MESSAGE="""You will answer multiple complex questions using iterative reasoning, and web search.
When taking an action, choose from one of the following actions:
   - If any question remains unanswered or if you have steps remaining, issue a single query for one question inside <search> ... </search>. The query should consist of keywords or a short phrase. Only search one question at a time.
   - If it is your last step, provide the final answers—separated by semicolons—within <answer> answer1; answer2; ... </answer>. The answers must be concise, contain only essential words, and avoid any explanations.

Important:
- Do not search multiple queries or questions simultaneously.

Answer the following questions: {question}"""


def get_NQHOTPOTQA_AGENT_FIRST_MESSAGE_MEM1(questions: str, instruct: bool):
   NQHOTPOTQA_AGENT_FIRST_MESSAGE_MEM1="""You will answer multiple complex questions using iterative reasoning, summarization, and web search.

At each step, you will see the questions, a cumulative summary of relevant information, the current search query, and search results (except in the first step, where only the questions are provided). Your task is to:

1. Perform reasoning and update a cumulative, concise summary within <think> ... </think>. This acts as persistent memory and must include all essential information from previous <think> and <information> tags.

2. Then choose one of the following actions:
   - If any question remains unanswered, issue a single query for one question inside <search> ... </search>. The query should consist of keywords or a short phrase. Only search one question at a time.
   - If all questions are answered, provide the final answers—separated by semicolons—within <answer> answer1; answer2; ... </answer>. The answers must be concise, contain only essential words, and avoid any explanations.

Important:
- Always follow this structure after <information> or the initial questions: <think> ... </think><search> ... </search> or <think> ... </think><answer> ... </answer>.
- Do not search multiple queries or questions simultaneously.

Answer the following questions: {questions}\n"""

   return (("" if instruct else "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n") 
            + NQHOTPOTQA_AGENT_FIRST_MESSAGE_MEM1.format_map({"questions": questions})
            + ("" if instruct else "<|im_end|>\n<|im_start|>assistant\n"))

def get_NQHOTPOTQA_FULL_AGENT_FIRST_MESSAGE_MEM1(questions: str, instruct: bool):
   NQHOTPOTQA_FULL_AGENT_FIRST_MESSAGE_MEM1="""You will answer multiple complex questions using iterative reasoning, summarization, and web search.

At each step, you will see the questions, a cumulative summary of relevant information, the current search query, and search results (except in the first step, where only the questions are provided). Your task is to:

1. Perform reasoning and update a cumulative, concise summary within <think> ... </think>. This acts as persistent memory and must include all essential information from previous <think> and <information> tags.

2. Then choose one of the following actions:
   - If any question remains unanswered or if you have steps remaining, issue a single query for one question inside <search> ... </search>. The query should consist of keywords or a short phrase. Only search one question at a time.
   - If it is your last step, provide the final answers—separated by semicolons—within <answer> answer1; answer2; ... </answer>. The answers must be concise, contain only essential words, and avoid any explanations.

Important:
- Always follow this structure after <information> or the initial questions: <think> ... </think><search> ... </search> or <think> ... </think><answer> ... </answer>.
- Do not search multiple queries or questions simultaneously.

Answer the following questions: {questions}\n"""

   return (("" if instruct else "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n") 
            + NQHOTPOTQA_FULL_AGENT_FIRST_MESSAGE_MEM1.format_map({"questions": questions})
            + ("" if instruct else "<|im_end|>\n<|im_start|>assistant\n"))

NQHOTPOTQA_ENV_RESPONSE_SEARCH="{hint}\n\n{search_result}"

NQHOTPOTQA_ENV_RESPONSE_SEARCH_MEM1='\n\n<information>{hint}\n\n{search_result}</information>\n\n'

# """At each step, you will see the questions, a cumulative belief state of relevant information, the current search query, and search results (except in the first step, where only the questions are provided). Your task is to:

# 1. Perform reasoning and update a cumulative, concise belief state within <belief> ... </belief>. This acts as persistent memory and must include all essential information from previous <belief> and <information> tags.
# """


# COMBO_FIRST_USER_MESSAGE="""Give me your first query formatted as a list of 3 characters inside <action> ... </action> after thinking inside <think> ... </think>, e.g., <think> Let's think step by step before giving the query [your extensive thinking] </think> <action>['char 1', 'char 2', 'char 3']</action>.
# """
# # interesting to note that the new line is here and was on the two below, before the most recent runs. This is just a sanity check
NQHOTPOTQA_FIRST_USER_MESSAGE="""Give me your first action. Remember to think before you act, as in <think> ... </think><search> ... </search> or <think> ... </think><answer> ... </answer>.""" 
NQHOTPOTQA_FULL_FIRST_USER_MESSAGE="""Give me your first action. Remember to think before you act, as in <think> ... </think><search> ... </search>.""" 

# COMBO_BELIEF_PROMPT="""{agent_first_message}
# Your current belief state: <belief>{belief_state}</belief>
# Your last action:
# <action>{agent_action}</action>
# Environment feedback:
# {env_response}
# Now update your belief state to include all important new information you have gathered.
# Do not say anything about future actions. Think step by step and then output your new belief state inside <belief> ... </belief>, e.g., <think>Any thinking</think><belief>your new beliefs</belief>."""

# the only valid action is search, if you attempt to answer, the game must be over, so we wouldn't desply the belief prompt any more.
NQHOTPOTQA_BELIEF_PROMPT="""Global Instruction: <instruction>{agent_first_message}</instruction>
Your current belief state: <belief>{belief_state}</belief>
Your last action: <search>{agent_action}</search>
Environment feedback: <environment>{env_response}</environment>
Now update your belief state to be a concise summary of all essential information you have gathered.
Do not say anything about future actions. Think step by step and then output your new belief state inside <belief> ... </belief>, e.g., <think>Any thinking</think><belief>your new belief</belief>."""


# COMBO_BELIEF_PROMPT_SINGLE_CONTEXT = """Now update your belief state to include all important new information you have gathered.
# Do not say anything about future actions. Think step by step and then output your new belief state inside <belief> ... </belief>, e.g., <think>Any thinking</think><belief>your new beliefs</belief>."""
# COMBO_ACTION_PROMPT_SINGLE_CONTEXT = """Now think step by step and then output your next action formatted as a list of 3 characters inside <action> ... </action>, e.g.,<think>Any step by step, short and concise thinking to determine your next action</think><action>['char 1', 'char 2', 'char 3']</action>."""

NQHOTPOTQA_BELIEF_PROMPT_SINGLE_CONTEXT="""Now update your belief state to be a concise summary of all essential information you have gathered.
Do not say anything about future actions. Think step by step and then output your new belief state inside <belief> ... </belief>, e.g., <think>Any thinking</think><belief>your new belief</belief>."""
NQHOTPOTQA_ACTION_PROMPT_SINGLE_CONTEXT="""Now think step by step and then output your next action formatted as <think> ... </think><search> ... </search> or <think> ... </think><answer> ... </answer>."""

# COMBO_ACTION_PROMPT="""Global Instruction: {agent_first_message}
# Current belief: <belief>{belief_state}</belief>
# Now think step by step and then output your next action formatted as a list of 3 characters inside <action> ... </action>, e.g.,<think>Any step by step, short and concise thinking to determine your next action</think><action>['char 1', 'char 2', 'char 3']</action>."""


NQHOTPOTQA_ACTION_PROMPT="""Global Instruction: <instruction>{agent_first_message}</instruction>
Current belief: <belief>{belief_state}</belief>
Now think step by step and then output your next action formatted as <think> ... </think><search> ... </search> or <think> ... </think><answer> ... </answer>. Remember if it is your last step you must answer. {hint}"""
NQHOTPOTQA_ACTION_PROMPT_TYPE_1="""Global Instruction: <instruction>{agent_first_message}</instruction>
Past action: <search>{prior_action}</search>
Current belief: <belief>{belief_state}</belief>
Now think step by step and then output your next action formatted as <think> ... </think><search> ... </search> or <think> ... </think><answer> ... </answer>. Remember if it is your last step you must answer. {hint}"""
NQHOTPOTQA_ACTION_PROMPT_TYPE_2="""Global Instruction: <instruction>{agent_first_message}</instruction>
Past action: <search>{prior_action}</search>
Past environment feedback: <environment>{prior_env_response}</environment>
Current belief: <belief>{belief_state}</belief>
Now think step by step and then output your next action formatted as <think> ... </think><search> ... </search> or <think> ... </think><answer> ... </answer>. Remember if it is your last step you must answer. {hint}"""
NQHOTPOTQA_FULL_ACTION_PROMPT = """Global Instruction: <instruction>{agent_first_message}</instruction>
Current belief: <belief>{belief_state}</belief>
Now think step by step and then output your next action formatted as <think> ... </think><search> ... </search> or <think> ... </think><answer> ... </answer>. Remember only put an answer on your last step. If you have steps remaining, you must search. {hint}"""

NQHOTPOTQA_ENV_RESPONSE = 'Could not parse response. Please ensure your response is formatted as <think> ... </think><search> ... </search> or <think> ... </think><answer> ... </answer>.' 