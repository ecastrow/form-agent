"""Unit tests for the routing functions in src/graph.py.

Run with: pytest -v
Tests only the pure routing functions (_route_from_entry,
_route_after_extraction, _route_by_action) directly -- these are
plain functions over a dict, fully testable without building or
running the actual compiled LangGraph graph. build_graph() itself
needs a real langgraph runtime and isn't exercised here; verify it
manually (e.g. via the CLI, once built) instead.
"""

from src.graph import (
    _route_after_extraction,
    _route_by_action,
    _route_from_entry,
)


def test_route_from_entry_prefers_extraction_when_uploads_pending():
    """Uploads route to extraction first, even with a query present."""
    state = {"pending_uploads": ["a.png"], "has_new_query": True}
    assert _route_from_entry(state) == "extraction"


def test_route_from_entry_routes_to_router_with_no_uploads():
    """No uploads but a new query goes straight to the router."""
    state = {"pending_uploads": [], "has_new_query": True}
    assert _route_from_entry(state) == "router"


def test_route_from_entry_ends_with_neither():
    """No uploads, no query -- shouldn't happen given

    build_turn_update()'s validation, but routes to end rather
    than erroring if it somehow does.
    """
    state = {"pending_uploads": [], "has_new_query": False}
    assert _route_from_entry(state) == "end"


def test_route_from_entry_ignores_stale_messages_history():
    """has_new_query, not messages, decides routing.

    This is the exact bug has_new_query exists to prevent: a
    non-empty messages list from prior turns must not trigger the
    router on a turn that's actually upload-only.
    """
    state = {
        "pending_uploads": ["b.png"],
        "has_new_query": False,
        "messages": ["some", "old", "accumulated", "history"],
    }
    assert _route_from_entry(state) == "extraction"


def test_route_after_extraction_continues_when_query_present():
    """Upload+question together: after extraction, go to router."""
    state = {"has_new_query": True}
    assert _route_after_extraction(state) == "router"


def test_route_after_extraction_ends_on_upload_only_turn():
    """Upload with no question: after extraction, end the turn."""
    state = {"has_new_query": False}
    assert _route_after_extraction(state) == "end"


def test_route_by_action_maps_summarize():
    """Router's summarize decision maps to the summarize node."""
    assert _route_by_action({"intent_action": "summarize"}) == "summarize"


def test_route_by_action_maps_qa():
    """Router's qa decision maps to the qa node."""
    assert _route_by_action({"intent_action": "qa"}) == "qa"
