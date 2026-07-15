"""Unit tests for src/turn_input.py.

Run with: pytest -v
Pure Python -- no LLM, no network, no LangGraph runtime needed.
"""

import pytest
from langchain_core.messages import HumanMessage

from src.turn_input import build_turn_update


def test_upload_only_sets_pending_uploads():
    """Uploading files with no message only sets pending_uploads."""
    update = build_turn_update(uploaded_files=["form1.png", "form2.png"])
    assert update == {
        "pending_uploads": ["form1.png", "form2.png"],
        "has_new_query": False,
    }


def test_message_only_sets_messages():
    """A message with no upload only sets messages."""
    update = build_turn_update(user_text="Summarize this form")
    assert "pending_uploads" not in update
    assert update["has_new_query"] is True
    assert len(update["messages"]) == 1
    assert isinstance(update["messages"][0], HumanMessage)
    assert update["messages"][0].content == "Summarize this form"


def test_upload_and_message_together():
    """A turn can carry both an upload and a question at once."""
    update = build_turn_update(
        user_text="How does this compare to the others?",
        uploaded_files=["new_claim.png"],
    )
    assert update["pending_uploads"] == ["new_claim.png"]
    assert update["has_new_query"] is True
    assert update["messages"][0].content == (
        "How does this compare to the others?"
    )


def test_neither_argument_raises():
    """An empty turn (no message, no upload) is not valid."""
    with pytest.raises(ValueError):
        build_turn_update()
