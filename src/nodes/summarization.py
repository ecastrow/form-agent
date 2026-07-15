"""Summarize node: produces a concise summary of one or more forms.

Handles 1..N forms through the same code path -- the router
simplification documented in technical_considerations.md, Sec. 11.
When multiple forms are in scope, the prompt explicitly asks for
synthesis across them, not separate summaries stitched together.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.nodes.shared import (
    conversation_history_text,
    form_context,
    latest_user_text,
    resolve_forms_in_scope,
)
from src.schemas import AgentState

# Generating an actual summary is a harder reasoning task than
# routing -- back on the stronger model tier, unlike router.py's
# gpt-4o-mini (Sec. 7's cost levers).
SUMMARIZE_MODEL = "gpt-4o"

SUMMARIZE_SYSTEM_PROMPT = """
You are summarizing one or more health insurance prior
authorization forms for someone who needs the gist quickly, not the
full form.

Rules:
- Highlight the most important details: patient identity, what is
  being requested and why (the diagnosis), review urgency, and
  anything flagged or unclear.
- If a "Conversation so far" section is included, use it only to
  understand what the user is actually asking for now (e.g. what
  "the other one" or "that patient" refers to) -- do not repeat or
  summarize earlier answers themselves.
- If the user's message signals a specific interest, focus on that
  rather than giving a generic overview of every field.
- If more than one form is provided, synthesize across them --
  shared patterns, differences, anything notable together -- rather
  than summarizing each one separately and stopping there.
- If a form has a FLAGGED note in its context below, mention that
  plainly rather than presenting flagged data as certain.
- Be concise. This is a summary, not a restatement of the form.
""".strip()


def summarize_node(
    state: AgentState,
    llm: Optional[ChatOpenAI] = None,
) -> dict:
    """Generates a summary for whatever forms are in scope.

    Args:
        state: current agent state -- reads messages, forms, and
            intent_scope.
        llm: optional pre-built chat model, same dependency-injection
            pattern used throughout -- lets tests inject a fake
            instead of making a real API call.

    Returns both `response` (the answer text on its own, convenient
    for callers/tests) and an `AIMessage` appended to `messages` --
    the latter is what the checkpointer will persist as
    conversation history for future turns once graph.py wires this
    up with memory.
    """
    forms = resolve_forms_in_scope(state)
    if not forms:
        answer = "There are no forms available to summarize yet."
        return {
            "response": answer,
            "messages": [AIMessage(content=answer)],
        }

    llm = llm or ChatOpenAI(model=SUMMARIZE_MODEL, temperature=0.3)

    all_messages = state.get("messages", [])
    user_text = latest_user_text(all_messages)
    request_line = user_text or "(no specific request; give a general summary)"

    # all_messages[-1] is assumed to be the current turn's own
    # message (the one turn_input.py just added) -- see
    # conversation_history_text()'s docstring.
    history_text = conversation_history_text(all_messages[:-1])
    history_block = (
        f"Conversation so far:\n{history_text}\n\n" if history_text else ""
    )

    context = "\n\n".join(form_context(f) for f in forms)

    messages = [
        SystemMessage(content=SUMMARIZE_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"{history_block}User request: {request_line}\n\n{context}"
            )
        ),
    ]

    result = llm.invoke(messages)
    answer = str(result.content)

    return {
        "response": answer,
        "messages": [AIMessage(content=answer)],
    }
