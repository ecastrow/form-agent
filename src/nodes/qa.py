"""QA node: answers a question using one or more forms.

Handles 1..N forms through the same code path as summarize_node --
the router simplification documented in technical_considerations.md,
Sec. 11. When multiple forms are in scope, the prompt explicitly
asks for a holistic answer, not per-form answers stitched together.
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

QA_MODEL = "gpt-4o"

QA_SYSTEM_PROMPT = """
You are answering a question about one or more health insurance
prior authorization forms, using only the form data provided below
-- do not use outside knowledge about specific people or claims.

Rules:
- Answer directly and specifically; do not restate the whole form.
- If a "Conversation so far" section is included, use it only to
  understand what the user is actually asking now (e.g. what "the
  other one" or "that patient" refers to) -- do not repeat earlier
  answers.
- If more than one form is provided, give a holistic answer that
  draws connections across them where the question calls for it
  (comparisons, totals, shared patterns), not several answers
  concatenated -- unless the question is clearly scoped to one form.
- If a fact relevant to your answer comes from a form with a
  FLAGGED note in its context below, mention that as a caveat
  rather than stating it as settled fact.
- If the available forms do not contain enough information to
  answer, say so directly rather than guessing.
""".strip()


def qa_node(
    state: AgentState,
    llm: Optional[ChatOpenAI] = None,
) -> dict:
    """Answers the latest user question using forms in scope.

    Args:
        state: current agent state -- reads messages, forms, and
            intent_scope.
        llm: optional pre-built chat model, same dependency-injection
            pattern used throughout -- lets tests inject a fake
            instead of making a real API call.

    Returns both `response` and an `AIMessage` appended to
    `messages`, same convention as summarize_node -- see that
    module's docstring for why the messages update matters.
    """
    forms = resolve_forms_in_scope(state)
    if not forms:
        answer = (
            "There are no forms available to answer that question "
            "from yet."
        )
        return {
            "response": answer,
            "messages": [AIMessage(content=answer)],
        }

    llm = llm or ChatOpenAI(model=QA_MODEL, temperature=0)

    all_messages = state.get("messages", [])
    user_text = latest_user_text(all_messages)

    # all_messages[-1] is assumed to be the current turn's own
    # message -- see conversation_history_text()'s docstring.
    history_text = conversation_history_text(all_messages[:-1])
    history_block = (
        f"Conversation so far:\n{history_text}\n\n" if history_text else ""
    )

    context = "\n\n".join(form_context(f) for f in forms)

    messages = [
        SystemMessage(content=QA_SYSTEM_PROMPT),
        HumanMessage(
            content=f"{history_block}Question: {user_text}\n\n{context}"
        ),
    ]

    result = llm.invoke(messages)
    answer = str(result.content)

    return {
        "response": answer,
        "messages": [AIMessage(content=answer)],
    }
