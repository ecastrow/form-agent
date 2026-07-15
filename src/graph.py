"""Assembles the full agent graph: ingestion + query pipelines,

wired into one graph with a checkpointer for multi-turn memory.

Sequential by design, not parallel: extraction always runs before
the router when a turn has both a new upload and a new question,
so the router sees up-to-date forms rather than racing against
extraction. See technical_considerations.md for the reasoning.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from src.nodes.extraction import extraction_node
from src.nodes.qa import qa_node
from src.nodes.router import router_node
from src.nodes.summarization import summarize_node
from src.schemas import AgentState


def _route_from_entry(state: AgentState) -> str:
    """Decides the first node to run this turn.

    Extraction runs first whenever there's something to ingest;
    the query pipeline (router onward) only runs if this turn
    actually included a new question -- see has_new_query's
    docstring on AgentState for why that's a distinct check from
    "messages is non-empty".
    """
    if state.get("pending_uploads"):
        return "extraction"
    if state.get("has_new_query"):
        return "router"
    return "end"  # defensive; build_turn_update() shouldn't allow this


def _route_after_extraction(state: AgentState) -> str:
    """After ingesting, decide if there's also a question to answer

    this turn (upload+question together) or if this was an
    upload-only turn.
    """
    if state.get("has_new_query"):
        return "router"
    return "end"


def _route_by_action(state: AgentState) -> str:
    """Router already decided the action; just map it to a node."""
    return state["intent_action"]


def build_graph():
    """Builds and compiles the full agent graph.

    Returns a compiled graph with an in-memory checkpointer
    (MemorySaver) -- state persists for the life of the process,
    keyed by whatever thread_id the caller passes in config. Not
    durable across restarts; fine for this project's scope (a CLI
    session), and swappable for a durable backend (e.g. a sqlite
    checkpointer) later without touching any node.
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("extraction", extraction_node)
    workflow.add_node("router", router_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("qa", qa_node)

    workflow.add_conditional_edges(
        START,
        _route_from_entry,
        {"extraction": "extraction", "router": "router", "end": END},
    )
    workflow.add_conditional_edges(
        "extraction",
        _route_after_extraction,
        {"router": "router", "end": END},
    )
    workflow.add_conditional_edges(
        "router",
        _route_by_action,
        {"summarize": "summarize", "qa": "qa"},
    )
    workflow.add_edge("summarize", END)
    workflow.add_edge("qa", END)

    # PriorAuthForm lives inside state["forms"] and gets checkpointed
    # on every turn. LangGraph's default serializer warns on any
    # custom type it doesn't explicitly recognize (recent security
    # hardening -- checkpoint deserialization can reconstruct
    # arbitrary Python objects, so unrecognized types are flagged,
    # and a future version blocks them outright rather than just
    # warning). Allowlisting it here is the actual fix, not just a
    # way to silence the warning.
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("src.schemas", "PriorAuthForm")],
    )
    checkpointer = MemorySaver(serde=serde)
    return workflow.compile(checkpointer=checkpointer)
