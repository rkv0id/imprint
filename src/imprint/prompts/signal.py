"""Prompt for LLM-based signal detection."""

SYSTEM = """\
You analyze a user's response to an AI agent and decide whether it carries \
a signal the agent should learn from.

Signal types:
- CORRECTION: the user is correcting something the agent said or did
- DIRECTION: the user is explicitly telling the agent how to behave
- PREFERENCE: the user is expressing a stable preference
- FACT: the user is stating a piece of information about themselves or their context
- REINFORCEMENT: the user is confirming the agent did something well

Most user responses are continuation - follow-up questions, acknowledgments, \
small talk - and contain NO signal. Default to NONE unless the response \
clearly matches one of the categories above.

Output exactly one of these tokens, with no other text and no punctuation:
CORRECTION, DIRECTION, PREFERENCE, FACT, REINFORCEMENT, NONE
"""


def build_user_prompt(*, agent_output: str, user_response: str) -> str:
    return f"AGENT'S OUTPUT:\n{agent_output}\n\nUSER'S RESPONSE:\n{user_response}\n\nSignal type:"
