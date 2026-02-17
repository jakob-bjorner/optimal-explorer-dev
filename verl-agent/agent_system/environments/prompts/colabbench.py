COLABBENCH_BELIEF_GENERATION_FAILURE_MSG="Belief generation failed to parse. The belief must be contained within <belief> ... </belief> tags. Try again."
    
COLABBENCH_NO_PRIOR_BELIEF_MESSAGE="No prior belief."

COLABBENCH_INVALID_ACTION_MESSAGE="invalid action"

COLABBENCH_AGENT_FIRST_MESSAGE="""You are a helpful LLM agent. 
Your task is to help a human user to resolve their problem, in particular python programming.
1) Note that the problem is highly personalized so you need to explicitly gather information 
by asking questions to the human user about some hidden information and implicit constraints.
YOU SHOULD TRY TO ASK CLARIFICATION QUESTIONS. Put your questions within <ask> ... </ask>.
2) Note that you should not ask human users complicated questions as they will only answer questions briefly in two sentences.
3) When you have gathered enough information to answer, put your final python code within <code> ... </code>.
4) Note that you can only interact with the human users WITHIN {max_attempts} back-and-forth rounds and you have to provide your final answer before the conversation ends.
5) You should be as concise as possible in your response to human.
6) Think step-by-step in think tags before you act. i.e <think> ... </think> <ask> ... </ask> or <think> ... </think> <code> ... </code>."""

# COLABBENCH_FIRST_USER_MESSAGE="""Give me your first query formatted as <action> ... </action> after thinking inside <think> ... </think>, e.g., <think> Let's think step by step before giving the query [your extensive thinking] </think> <action>[your code]</action>.
# """ 
# interesting to note that the new line is here and was on the two below, before the most recent runs. This is just a sanity check

COLABBENCH_BELIEF_PROMPT="""Global Instruction: <instruction>{agent_first_message}</instruction>
First user query: <query>{first_user_query}</query>
Your current belief state: <belief>{belief_state}</belief>
Your last action: <action>{agent_action}</action>
Environment feedback: <environment>{env_response}</environment>
Now update your belief state to include all important new information you have gathered.
Do not say anything about future actions. Think step by step and then output your new belief state inside <belief> ... </belief>, e.g., <think>Any thinking</think><belief>your new beliefs</belief>."""





COLABBENCH_BELIEF_PROMPT_SINGLE_CONTEXT="""Now update your belief state to be a concise summary of all essential information you have gathered.
Do not say anything about future actions. Think step by step and then output your new belief state inside <belief> ... </belief>, e.g., <think>Any thinking</think><belief>your new belief</belief>."""
COLABBENCH_ACTION_PROMPT_SINGLE_CONTEXT="""Now think step by step and then output your next action formatted as <think> ... </think><ask> ... </ask> or <think> ... </think><code> ... </code>."""

# COMBO_ACTION_PROMPT="""Global Instruction: {agent_first_message}
# Current belief: <belief>{belief_state}</belief>
# Now think step by step and then output your next action formatted as a list of 3 characters inside <action> ... </action>, e.g.,<think>Any step by step, short and concise thinking to determine your next action</think><action>['char 1', 'char 2', 'char 3']</action>."""


COLABBENCH_ACTION_PROMPT="""Global Instruction: <instruction>{agent_first_message}</instruction>
First user query: <query>{first_user_query}</query>
Current belief: <belief>{belief_state}</belief>
Now think step by step and then output your next action formatted as <think> ... </think><ask> ... </ask> or <think> ... </think><code> ... </code>. Remember if it is your last step you must code. {hint}"""

COLABBENCH_ENV_RESPONSE = 'Could not parse response. Please ensure your response is formatted as <think> ... </think><ask> ... </ask> or <think> ... </think><code> ... </code>.' 

COLABBENCH_TURNS_REMAINING_HINT = " Remember if it is your last step you must code. {hint}"

COLABBENCH_HUMAN_SIMULATOR_CODE_PROMPT = """Your task is to simulate a human user that interacts with an LLM agent in a dialogue.
You would like the LLM agent to help you with the following problem:
{problem_description}

Your goal is to engage in the conversation with the LLM agent so that it can get to a personalized answer.
You should make use of the following hidden information to answer the LLM agent.
YOU SHOULD BEHAVE LIKE A HUMAN THAT NEEDS THE HELP FROM AN AGENT.
You SHOULD ONLY ANSWER QUESTIONS WITH INFORMATION PROVIDED IN THE HIDDEN INFORMATION, AND SAY YOU DON"T KNOW IF THE ANSWER CAN NOT BE FOUND IN THE HIDDEN INFORMATION.
DO NOT GIVE THE LLM AGENT THE CODE DIRECTLY.

HIDDEN INFORMATION:
{hidden_information}
END OF HIDDEN INFORMATION

Here is the dialogue so far:
{dialogue_history}


Now directly output your answer to the LLM agent IN TWO SENTENCES. DO NOT SAY ANYTHING ELSE."""

COLABBENCH_BELIEF_GRADING_0_NO_LOSS = """Global Instruction: <instruction>You are a helpful LLM agent. 
Your task is to help a human user to resolve their problem, in particular python programming.
1) Note that the problem is highly personalized so you need to explicitly gather information 
by asking questions to the human user about some hidden information and implicit constraints.
YOU SHOULD TRY TO ASK CLARIFICATION QUESTIONS. Put your questions within <ask> ... </ask>.
2) Note that you should not ask human users complicated questions as they will only answer questions briefly in two sentences.
3) When you have gathered enough information to answer, put your final python code within <code> ... </code>.
4) Note that you can only interact with the human users WITHIN 10 back-and-forth rounds and you have to provide your final answer before the conversation ends.
5) You should be as concise as possible in your response to human.
6) Think step-by-step in think tags before you act. i.e <think> ... </think> <ask> ... </ask> or <think> ... </think> <code> ... </code>.</instruction>
First user query: <query>{first_user_query}</query>
Your new belief stateis : <belief>{future_belief}</belief>

Your past belief state was: <belief>"""


COLABBENCH_BELIEF_GRADING_1_LOSS = """{prior_belief}</belief>

"""

COLABBENCH_BELIEF_GRADING_2_NO_LOSS = """Your past action: <action>"""
COLABBENCH_BELIEF_GRADING_3_LOSS = """{prior_action}</action>

"""
COLABBENCH_BELIEF_GRADING_4_NO_LOSS = """

Your past environment feedback: <environment>"""
COLABBENCH_BELIEF_GRADING_5_LOSS = """{prior_obs}</environment>

"""