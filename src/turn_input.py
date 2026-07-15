"""Builds the partial state update for a single turn.

This is intentionally NOT a graph node. Detecting "a file was
uploaded" or "the user sent a message" is an interface/IO concern
-- a CLI loop, a notebook cell, or eventually a UI -- not agent
reasoning. Whatever surfaces user input is responsible for calling
this before invoking the graph; the graph itself only ever sees the
resulting state update, never the raw upload event.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.messages import HumanMessage


def build_turn_update(
    user_text: Optional[str] = None,
    uploaded_files: Optional[list[str]] = None,
) -> dict:
    """Builds the partial state update LangGraph expects for one turn.

    A turn may carry a message, uploaded files, or both at once --
    e.g. "here's a new claim, how does it compare to the others?"
    uploads a file and asks a question in the same breath. Whichever
    fields are present here are what the (not-yet-built) graph will
    route on: pending_uploads triggers extraction_node, a new
    message triggers the router.

    Raises:
        ValueError: if neither a message nor a file is given -- an
            empty turn isn't a valid thing to send the graph.
    """
    if not user_text and not uploaded_files:
        raise ValueError(
            "A turn needs at least a message or an uploaded file."
        )

    # Always explicitly set, both True and False -- never omitted.
    # `messages` accumulates forever via LangGraph's add_messages
    # reducer, so state["messages"] being non-empty does NOT mean
    # "there's a new question this turn" from turn 2 onward. This
    # flag is graph.py's only reliable signal for that, precisely
    # because it's always present (partial-update semantics leave
    # omitted keys unchanged, which would make this stale if it
    # were only set on the True case).
    update: dict = {"has_new_query": bool(user_text)}
    if uploaded_files:
        update["pending_uploads"] = list(uploaded_files)
    if user_text:
        update["messages"] = [HumanMessage(content=user_text)]
    return update
