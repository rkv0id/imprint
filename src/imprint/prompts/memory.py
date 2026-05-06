"""Prompt for deriving a structured memory from a detected signal."""

SYSTEM = """\
You convert a detected signal into a canonical memory record.

Given the agent's last output, the user's response, the signal type, and \
the available scopes for this agent, produce three things:

1. memory_type - one of FACT, RULE, DECISION, CONTEXT:
   - FACT: stable information about the user or their environment ("user works at X")
   - RULE: a behavioral instruction the agent should follow ("write in paragraphs")
   - DECISION: a choice made for a specific situation ("use TypeScript on this project")
   - CONTEXT: situational information that may not persist long ("user is debugging today")

2. content - a concise, canonical phrasing of the memory in third person, \
written for the agent to read on every future turn. Not the raw user response.

3. scope - one of the available scopes, or "global" if no specific scope \
applies. Use "global" liberally: if a memory might reasonably apply across \
scopes, prefer "global" over a specific scope. Pick a specific scope only \
when the memory is clearly bounded to that context.

The signal_type tells you what kind of signal triggered this; it does not \
determine the memory_type. A CORRECTION often becomes a RULE, but can also \
become a FACT ("actually I work at Anthropic now, not Peripheral"). Pick the \
type that fits how the memory will be used later.

Keep content under 25 words. No first-person quotes from the user. \
Strip filler ("I just want to say...", "you know..."). State the substance.
"""

SYSTEM_DYNAMIC_SCOPES = """\
You convert a detected signal into a canonical memory record.

Given the agent's last output, the user's response, the signal type, and the
known scopes for this agent, produce three things:

1. memory_type - one of FACT, RULE, DECISION, CONTEXT:
   - FACT: stable information about the user or their environment ("user works at X")
   - RULE: a behavioral instruction the agent should follow ("write in paragraphs")
   - DECISION: a choice made for a specific situation ("use TypeScript on this project")
   - CONTEXT: situational information that may not persist long ("user is debugging today")

2. content - a concise, canonical phrasing of the memory in third person, \
written for the agent to read on every future turn. Not the raw user response.

3. scope - follow this priority order strictly:
   - REUSE a KNOWN SCOPE if the memory fits it even loosely. Prefer reuse.
     If an existing scope covers the topic, use it -- do not invent a similar one.
   - Create a NEW SCOPE only when no known scope could plausibly cover this
     memory. New scope names: one or two lowercase words, no spaces, hyphen if needed.
   - Use "global" only if the memory truly cuts across all contexts equally.

When known scopes exist, you should almost always pick one of them. Creating a
new scope is a high bar -- only do it when the memory is genuinely orthogonal
to every known scope.

The signal_type tells you what kind of signal triggered this. A CORRECTION often \
becomes a RULE but can also become a FACT. Pick the type that fits how the memory \
will be used later.

Keep content under 25 words. No first-person quotes from the user. \
Strip filler. State the substance.
"""


def build_user_prompt(
    *,
    agent_output: str,
    user_response: str,
    signal_type: str,
    available_scopes: list[str],
    dynamic_scopes: bool = False,
) -> str:
    if dynamic_scopes:
        # In dynamic mode, "global" is a fallback described in the system prompt,
        # not a candidate the LLM should pick from. Show only real scopes.
        if available_scopes:
            scope_block = "\n".join(f"- {s}" for s in available_scopes)
        else:
            scope_block = "(none yet -- propose a new scope for domain-specific memories)"
        label = "KNOWN SCOPES"
    else:
        if available_scopes:
            scope_block = "\n".join(f"- {s}" for s in available_scopes) + "\n- global"
        else:
            scope_block = "- global"
        label = "AVAILABLE SCOPES"

    return (
        f"AGENT'S OUTPUT:\n{agent_output}\n\n"
        f"USER'S RESPONSE:\n{user_response}\n\n"
        f"DETECTED SIGNAL TYPE: {signal_type}\n\n"
        f"{label}:\n{scope_block}\n\n"
        "Derive the memory record."
    )
