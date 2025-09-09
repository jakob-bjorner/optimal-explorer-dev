COMBO_BELIEF_GENERATION_FAILURE_MSG="Belief generation failed to parse. The belief must be contained within <belief> ... </belief> tags. Try again."
    
COMBO_NO_PRIOR_BELIEF_MESSAGE="No prior belief."

COMBO_INVALID_ACTION_MESSAGE="invalid action"

COMBO_AGENT_FIRST_MESSAGE="""You will determine the correct combination of characters at [Position 1, Position 2, Position 3] in a 3-character combination lock through iterative reasoning and queries.
All 3 characters are unique.
The set of valid characters are as follows: {vocab_list}
Each action is a query of the form ['char 1', 'char 2', 'char 3'].
Each time you query a combination, you will get feedback from the user about each character: either not in the combination, in the combination but in a different position, or in the combination and in the right position.
You can make up to {max_attempts} queries.
Your goal is to find the correct combination in the least number of queries."""

COMBO_FIRST_USER_MESSAGE="""Give me your first query formatted as a list of 3 characters inside <action> ... </action> after thinking inside <think> ... </think>, e.g., <think> Let's think step by step before giving the query [your extensive thinking] </think> <action>['char 1', 'char 2', 'char 3']</action>.
""" 
# interesting to note that the new line is here and was on the two below, before the most recent runs. This is just a sanity check

COMBO_BELIEF_PROMPT="""{agent_first_message}
Your current belief state: <belief>{belief_state}</belief>
Your last action:
<action>{agent_action}</action>
Environment feedback:
{env_response}
Now update your belief state to include all important new information you have gathered.
Do not say anything about future actions. Think step by step and then output your new belief state inside <belief> ... </belief>, e.g., <think>Any thinking</think><belief>your new beliefs</belief>."""

COMBO_BELIEF_PROMPT_SINGLE_CONTEXT = """Now update your belief state to include all important new information you have gathered.
Do not say anything about future actions. Think step by step and then output your new belief state inside <belief> ... </belief>, e.g., <think>Any thinking</think><belief>your new beliefs</belief>."""
COMBO_ACTION_PROMPT_SINGLE_CONTEXT = """Now think step by step and then output your next action formatted as a list of 3 characters inside <action> ... </action>, e.g.,<think>Any step by step, short and concise thinking to determine your next action</think><action>['char 1', 'char 2', 'char 3']</action>."""
COMBO_ACTION_PROMPT="""Global Instruction: {agent_first_message}
Current belief: <belief>{belief_state}</belief>
Now think step by step and then output your next action formatted as a list of 3 characters inside <action> ... </action>, e.g.,<think>Any step by step, short and concise thinking to determine your next action</think><action>['char 1', 'char 2', 'char 3']</action>."""

