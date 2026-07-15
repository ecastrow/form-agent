"""Unit tests for src/nodes/qa.py.

Run with: pytest -v
No network or API key needed -- same fake-LLM pattern as
test_summarization.py.
"""

from langchain_core.messages import AIMessage, HumanMessage

from src.nodes.qa import qa_node
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
    """Builds a minimal valid form for QA tests."""
    return PriorAuthForm(
        source_file=source_file,
        patient=PatientInfo(name=patient_name, dob="03/20/1992"),
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


def test_qa_returns_response_text():
    """The node's response is exactly the LLM's returned text."""
    fake_llm = _FakeLLM("Daniel Jarvis's DOB is 03/20/1992.")
    state = {
        "messages": [HumanMessage(content="What is the DOB?")],
        "forms": [_minimal_form("a.png", "Daniel Jarvis")],
        "intent_scope": None,
    }
    update = qa_node(state, llm=fake_llm)
    assert update["response"] == "Daniel Jarvis's DOB is 03/20/1992."


def test_qa_appends_ai_message_for_conversation_memory():
    """The answer also lands in messages, not just response."""
    fake_llm = _FakeLLM("An answer.")
    state = {
        "messages": [HumanMessage(content="What is the DOB?")],
        "forms": [_minimal_form("a.png", "Daniel Jarvis")],
        "intent_scope": None,
    }
    update = qa_node(state, llm=fake_llm)
    assert len(update["messages"]) == 1
    assert isinstance(update["messages"][0], AIMessage)
    assert update["messages"][0].content == "An answer."


def test_qa_with_no_forms_skips_the_llm_call():
    """No forms in scope means a canned response, no LLM call made."""
    fake_llm = _FakeLLM("should never be returned")
    state = {"messages": [], "forms": [], "intent_scope": None}
    update = qa_node(state, llm=fake_llm)
    assert "no forms" in update["response"].lower()
    assert fake_llm.last_messages is None


def test_qa_includes_all_forms_when_scope_is_none():
    """intent_scope of None means every form reaches the prompt."""
    fake_llm = _FakeLLM("holistic answer")
    state = {
        "messages": [HumanMessage(content="Compare these two forms")],
        "forms": [
            _minimal_form("a.png", "Daniel Jarvis"),
            _minimal_form("b.png", "Danielle Pratt"),
        ],
        "intent_scope": None,
    }
    qa_node(state, llm=fake_llm)
    prompt_text = str(fake_llm.last_messages[-1].content)
    assert "a.png" in prompt_text
    assert "b.png" in prompt_text


def test_qa_flags_escalated_forms_in_prompt():
    """A form's FLAGGED caveat reaches the prompt so it can hedge."""
    fake_llm = _FakeLLM("answer with caveat")
    escalated = _minimal_form("a.png", "Daniel Jarvis").model_copy(
        update={"extraction_confidence": 0.3}
    )
    state = {
        "messages": [HumanMessage(content="What's the confidence?")],
        "forms": [escalated],
        "intent_scope": None,
    }
    qa_node(state, llm=fake_llm)
    prompt_text = str(fake_llm.last_messages[-1].content)
    assert "FLAGGED" in prompt_text


def test_qa_includes_prior_conversation_in_prompt():
    """Earlier turns reach the prompt, so follow-up questions resolve.

    The quick-and-dirty fix: without this, "what about her DOB?"
    on turn 2 would have no idea who "her" refers to.
    """
    fake_llm = _FakeLLM("answer")
    state = {
        "messages": [
            HumanMessage(content="Who is the patient on form a?"),
            AIMessage(content="The patient is Daniel Jarvis."),
            HumanMessage(content="What about his DOB?"),
        ],
        "forms": [_minimal_form("a.png", "Daniel Jarvis")],
        "intent_scope": None,
    }
    qa_node(state, llm=fake_llm)
    prompt_text = str(fake_llm.last_messages[-1].content)
    assert "Conversation so far" in prompt_text
    assert "Who is the patient on form a?" in prompt_text
    assert "The patient is Daniel Jarvis" in prompt_text
    assert prompt_text.count("What about his DOB?") == 1
