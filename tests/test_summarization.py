"""Unit tests for src/nodes/summarization.py.

Run with: pytest -v
No network or API key needed -- a fake LLM stands in. Unlike
extraction/router, this node calls llm.invoke() directly for a
plain-text response, not with_structured_output(), since the
output IS the final answer text, not structured data for
downstream code.
"""

from langchain_core.messages import AIMessage, HumanMessage

from src.nodes.summarization import summarize_node
from src.schemas import (
    PatientInfo,
    PriorAuthForm,
    ProviderInfo,
    RequestingProviderDetail,
    ServiceLine,
    ServiceProviderDetail,
    ServicesRequested,
)


class _FakeResponse:
    """Stands in for the AIMessage llm.invoke() normally returns."""

    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    """Stands in for a ChatOpenAI instance."""

    def __init__(self, content: str):
        self._content = content
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return _FakeResponse(self._content)


def _minimal_form(source_file: str, patient_name: str) -> PriorAuthForm:
    """Builds a minimal valid form for summarize tests."""
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


def test_summarize_returns_response_text():
    """The node's response is exactly the LLM's returned text."""
    fake_llm = _FakeLLM("This form is a routine PT request.")
    state = {
        "messages": [HumanMessage(content="Summarize this")],
        "forms": [_minimal_form("a.png", "Daniel Jarvis")],
        "intent_scope": None,
    }
    update = summarize_node(state, llm=fake_llm)
    assert update["response"] == "This form is a routine PT request."


def test_summarize_appends_ai_message_for_conversation_memory():
    """The answer also lands in messages, not just response.

    This is the fix for the gap found in the earlier (discarded)
    version: without this, future turns' checkpointed context
    would be missing the assistant's own prior replies.
    """
    fake_llm = _FakeLLM("Summary text here.")
    state = {
        "messages": [HumanMessage(content="Summarize this")],
        "forms": [_minimal_form("a.png", "Daniel Jarvis")],
        "intent_scope": None,
    }
    update = summarize_node(state, llm=fake_llm)
    assert len(update["messages"]) == 1
    assert isinstance(update["messages"][0], AIMessage)
    assert update["messages"][0].content == "Summary text here."


def test_summarize_with_no_forms_skips_the_llm_call():
    """No forms in scope means a canned response, no LLM call made."""
    fake_llm = _FakeLLM("should never be returned")
    state = {"messages": [], "forms": [], "intent_scope": None}
    update = summarize_node(state, llm=fake_llm)
    assert "No forms" in update["response"] or (
        "no forms" in update["response"].lower()
    )
    assert fake_llm.last_messages is None


def test_summarize_only_includes_scoped_forms_in_prompt():
    """Only forms in intent_scope reach the LLM's prompt context."""
    fake_llm = _FakeLLM("summary text")
    state = {
        "messages": [HumanMessage(content="Summarize Daniel's form")],
        "forms": [
            _minimal_form("a.png", "Daniel Jarvis"),
            _minimal_form("b.png", "Danielle Pratt"),
        ],
        "intent_scope": ["a.png"],
    }
    summarize_node(state, llm=fake_llm)
    prompt_text = str(fake_llm.last_messages[-1].content)
    assert "a.png" in prompt_text
    assert "b.png" not in prompt_text


def test_summarize_flags_escalated_forms_in_prompt():
    """A form's FLAGGED caveat reaches the prompt so it can hedge."""
    fake_llm = _FakeLLM("summary with caveat")
    escalated = _minimal_form("a.png", "Daniel Jarvis").model_copy(
        update={"extraction_confidence": 0.3}
    )
    state = {
        "messages": [HumanMessage(content="Summarize this")],
        "forms": [escalated],
        "intent_scope": None,
    }
    summarize_node(state, llm=fake_llm)
    prompt_text = str(fake_llm.last_messages[-1].content)
    assert "FLAGGED" in prompt_text


def test_summarize_includes_prior_conversation_in_prompt():
    """Earlier turns reach the prompt, so follow-ups have context.

    The quick-and-dirty fix: without this, "summarize the other
    one" on turn 2 would have no idea what "the other one" means.
    """
    fake_llm = _FakeLLM("summary")
    state = {
        "messages": [
            HumanMessage(content="Summarize Daniel's form"),
            AIMessage(content="Daniel's form is a routine PT request."),
            HumanMessage(content="Now summarize the other one"),
        ],
        "forms": [_minimal_form("a.png", "Daniel Jarvis")],
        "intent_scope": None,
    }
    summarize_node(state, llm=fake_llm)
    prompt_text = str(fake_llm.last_messages[-1].content)
    assert "Conversation so far" in prompt_text
    assert "Summarize Daniel's form" in prompt_text
    assert "routine PT request" in prompt_text
    # The current turn's own message should appear once, as the
    # request line -- not duplicated inside the history block too.
    assert prompt_text.count("Now summarize the other one") == 1
