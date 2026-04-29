"""Prompt for deriving a structured memory from a detected signal."""

SYSTEM = """\
You convert a detected signal into a canonical memory record.

Given the agent's last output, the user's response, and the signal type, \
produce two things:

1. memory_type - one of FACT, RULE, DECISION, CONTEXT:
   - FACT: stable information about the user or their environment ("user works at X")
   - RULE: a behavioral instruction the agent should follow ("write in paragraphs")
   - DECISION: a choice made for a specific situation ("use TypeScript on this project")
   - CONTEXT: situational information that may not persist long ("user is debugging today")

2. content - a concise, canonical phrasing of the memory in third person, \
written for the agent to read on every future turn. Not the raw user response.

The signal_type tells you what kind of signal triggered this; it does not \
determine the memory_type. A CORRECTION often becomes a RULE, but can also \
become a FACT ("actually I work at Anthropic now, not Peripheral"). Pick the \
type that fits how the memory will be used later.

Keep content under 25 words. No first-person quotes from the user. \
Strip filler ("I just want to say...", "you know..."). State the substance.
"""


def build_user_prompt(
    *,
    agent_output: str,
    user_response: str,
    signal_type: str,
) -> str:
    return (
        f"AGENT'S OUTPUT:\n{agent_output}\n\n"
        f"USER'S RESPONSE:\n{user_response}\n\n"
        f"DETECTED SIGNAL TYPE: {signal_type}\n\n"
        "Derive the memory record."
    )
