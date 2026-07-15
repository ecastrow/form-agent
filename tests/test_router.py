"""Unit tests for src/nodes/router.py.

Run with: pytest -v
No network or API key needed -- a fake LLM stands in for the real
OpenAI call, same pattern as tests/test_extraction.py. Tests for
the shared helpers router.py uses (latest_user_text,
form_reference_list) live in tests/test_shared.py instead, since
those functions live in src/nodes/shared.py, not here.
"""

from langchain_core.messages import HumanMessage

from src.nodes.router import RouterDecision, router_node
from src.schemas import (
    PatientInfo,
    PriorAuthForm,
    ProviderInfo,
    RequestingProviderDetail,
    ServiceLine,
    ServiceProviderDetail,
    ServicesRequested,
)


class _FakeStructuredLLM:
    """Stands in for llm.with_structured_output(...)."""

    def __init__(self, result):
        self._result = result

    def invoke(self, messages):
        return self._result


class _FakeLLM:
    """Stands in for a ChatOpenAI instance."""

    def __init__(self, result):
        self._result = result

    def with_structured_output(self, schema):
        return _FakeStructuredLLM(self._result)


def _minimal_form(source_file: str, patient_name: str) -> PriorAuthForm:
    """Builds a minimal valid form for router-context tests."""
    return PriorAuthForm(
        source_file=source_file,
        patient=PatientInfo(name=patient_name),
        providers=ProviderInfo(
            requesting_provider=RequestingProviderDetail(name="Dr. A"),
            service_provider=ServiceProviderDetail(name="Dr. B"),
        ),
        services_requested=ServicesRequested(
            service_lines=[
                ServiceLine(planned_service_or_procedure="X"),
            ],
        ),
        extraction_confidence=0.95,
    )


def test_router_returns_summarize_action():
    """A summarize decision maps directly onto intent_action."""
    decision = RouterDecision(action="summarize", scope=[])
    state = {
        "messages": [HumanMessage(content="Summarize this form")],
        "forms": [_minimal_form("a.png", "Daniel Jarvis")],
    }
    update = router_node(state, llm=_FakeLLM(decision))
    assert update["intent_action"] == "summarize"
    assert update["intent_scope"] is None


def test_router_returns_scoped_qa():
    """A non-empty scope from the decision passes through as-is."""
    decision = RouterDecision(action="qa", scope=["a.png"])
    state = {
        "messages": [HumanMessage(content="What is the diagnosis?")],
        "forms": [_minimal_form("a.png", "Daniel Jarvis")],
    }
    update = router_node(state, llm=_FakeLLM(decision))
    assert update["intent_action"] == "qa"
    assert update["intent_scope"] == ["a.png"]
