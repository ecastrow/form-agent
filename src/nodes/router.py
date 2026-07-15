"""Router node: classifies what to do with a user's query.

Given the latest user message and the forms already ingested,
decides whether the user wants a summary or an answer to a
question, and which forms (if not all of them) the query concerns.
This is the entry point of the query pipeline.
"""

from __future__ import annotations

from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.nodes.shared import form_reference_list, latest_user_text
from src.schemas import AgentState

# Classification is a simple, cheap task relative to extraction or
# answer generation -- a smaller/cheaper model tier is appropriate
# here, per the cost levers in technical_considerations.md, Sec. 7.
ROUTER_MODEL = "gpt-4o-mini"

ROUTER_SYSTEM_PROMPT = """
You are routing a user's question about one or more insurance
prior authorization forms to the right handler.

Decide:
- action: "summarize" if the user wants an overview/summary of one
  or more forms. "qa" if they are asking a specific question that
  needs an answer -- including comparisons or holistic questions
  across forms.
- scope: which forms (by source_file) the question is about. Use
  the exact source_file values from the list provided. Leave scope
  empty if the user is referring to all available forms, or did
  not specify a subset -- do not guess a subset when "all" or
  nothing specific was said.
""".strip()


class RouterDecision(BaseModel):
    action: Literal["summarize", "qa"]
    scope: list[str] = Field(
        default_factory=list,
        description=(
            "source_file values in scope; empty means all forms "
            "currently in state"
        ),
    )


def router_node(
    state: AgentState,
    llm: Optional[ChatOpenAI] = None,
) -> dict:
    """Classifies the latest user message into an action and scope.

    Args:
        state: current agent state -- reads messages and forms.
        llm: optional pre-built chat model, same dependency-injection
            pattern as extract_form() -- lets tests inject a fake
            instead of making a real API call.
    """
    llm = llm or ChatOpenAI(model=ROUTER_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(RouterDecision)

    user_text = latest_user_text(state.get("messages", []))
    available_forms = form_reference_list(state)

    messages = [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Available forms:\n{available_forms}\n\n"
                f"User message: {user_text}"
            )
        ),
    ]

    decision = structured_llm.invoke(messages)

    return {
        "intent_action": decision.action,
        # Empty list -> None, matching AgentState's documented
        # convention: empty/None means "all forms in scope".
        "intent_scope": decision.scope or None,
    }
