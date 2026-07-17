"""Unit tests for src/nodes/shared.py.

Run with: pytest -v
Pure Python -- no LLM, no network needed for any of these.
"""

from langchain_core.messages import AIMessage, HumanMessage

from src.nodes.shared import (
    conversation_history_text,
    form_context,
    form_reference_list,
    latest_user_text,
    resolve_forms_in_scope,
)
from src.schemas import (
    DMEDetails,
    HomeHealthDetails,
    PatientInfo,
    PriorAuthForm,
    ProviderInfo,
    RequestingProviderDetail,
    ServiceLine,
    ServiceProviderDetail,
    ServicesRequested,
    TherapyDetails,
)


def _minimal_form(source_file: str, patient_name: str) -> PriorAuthForm:
    """Builds a minimal valid form for shared-helper tests."""
    return PriorAuthForm(
        source_file=source_file,
        patient=PatientInfo(name=patient_name, dob="01/01/2000"),
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


def test_latest_user_text_finds_most_recent_human_message():
    """Only the latest HumanMessage is used, not earlier ones."""
    messages = [
        HumanMessage(content="first question"),
        AIMessage(content="an answer"),
        HumanMessage(content="second question"),
    ]
    assert latest_user_text(messages) == "second question"


def test_latest_user_text_handles_no_human_message():
    """No human message yet returns an empty string, not an error."""
    assert latest_user_text([]) == ""


def test_conversation_history_text_formats_turns():
    """Human/AI turns render as labeled, alternating plain text."""
    messages = [
        HumanMessage(content="Summarize this"),
        AIMessage(content="It's a routine PT request."),
    ]
    result = conversation_history_text(messages)
    assert result == (
        "User: Summarize this\nAssistant: It's a routine PT request."
    )


def test_conversation_history_text_handles_empty_list():
    """No prior turns yields an empty string, not an error."""
    assert conversation_history_text([]) == ""


def test_resolve_forms_in_scope_returns_all_when_scope_empty():
    """No intent_scope set means every form in state is in scope."""
    forms = [
        _minimal_form("a.png", "Daniel Jarvis"),
        _minimal_form("b.png", "Danielle Pratt"),
    ]
    state = {"forms": forms, "intent_scope": None}
    assert resolve_forms_in_scope(state) == forms


def test_resolve_forms_in_scope_filters_to_named_forms():
    """A non-empty intent_scope filters down to those forms only."""
    form_a = _minimal_form("a.png", "Daniel Jarvis")
    form_b = _minimal_form("b.png", "Danielle Pratt")
    state = {"forms": [form_a, form_b], "intent_scope": ["b.png"]}
    assert resolve_forms_in_scope(state) == [form_b]


def test_form_reference_list_handles_empty_state():
    """No forms yet produces a clear placeholder, not an error."""
    result = form_reference_list({"forms": []})
    assert result == "(no forms ingested yet)"


def test_form_reference_list_lists_each_form_by_patient():
    """Each form appears as filename plus patient name."""
    state = {
        "forms": [
            _minimal_form("a.png", "Daniel Jarvis"),
            _minimal_form("b.png", "Danielle Pratt"),
        ]
    }
    result = form_reference_list(state)
    assert "a.png" in result and "Daniel Jarvis" in result
    assert "b.png" in result and "Danielle Pratt" in result


def test_form_context_includes_caveats_when_escalating():
    """A form needing escalation gets a visible FLAGGED marker."""
    form = _minimal_form("a.png", "Daniel Jarvis").model_copy(
        update={"extraction_confidence": 0.3}
    )
    context = form_context(form)
    assert "FLAGGED" in context
    assert "low overall confidence" in context


def test_form_context_omits_caveats_when_clean():
    """A form with no escalation reasons has no FLAGGED marker."""
    form = _minimal_form("a.png", "Daniel Jarvis")
    context = form_context(form)
    assert "FLAGGED" not in context


def test_form_context_includes_service_line_details():
    """Service line procedure and ICD code appear in the context."""
    form = _minimal_form("a.png", "Daniel Jarvis")
    context = form_context(form)
    assert "X" in context  # the planned_service_or_procedure value


def test_form_context_includes_therapy_when_present():
    """Real bug reproduced: therapy data was silently omitted.

    This is the exact shape of data that was extracted correctly
    for Daniel's real form but never reached the LLM's prompt,
    causing the agent to incorrectly say no therapy was found.
    """
    form = _minimal_form("a.png", "Daniel Jarvis")
    form.services_requested.therapy = TherapyDetails(
        types=["Mental Health/Substance Abuse"],
        number_of_sessions="4",
        duration="2 weeks",
    )
    context = form_context(form)
    assert "Mental Health/Substance Abuse" in context
    assert "4 sessions" in context
    assert "2 weeks" in context


def test_form_context_omits_therapy_when_not_present():
    """No therapy types means no Therapy line -- not a noisy 'None'."""
    form = _minimal_form("a.png", "Daniel Jarvis")
    context = form_context(form)
    assert "Therapy" not in context


def test_form_context_includes_settings_when_present():
    """Setting checkboxes (e.g. Outpatient) reach the context."""
    form = _minimal_form("a.png", "Daniel Jarvis")
    form.services_requested.settings = ["Outpatient"]
    context = form_context(form)
    assert "Setting: Outpatient" in context


def test_form_context_includes_home_health_when_requested():
    """Home health details reach the context when populated."""
    form = _minimal_form("a.png", "Daniel Jarvis")
    form.services_requested.home_health = HomeHealthDetails(
        requested=True, number_of_visits="3"
    )
    context = form_context(form)
    assert "Home health: requested" in context
    assert "3 visits" in context


def test_form_context_includes_dme_when_requested():
    """DME details reach the context when populated."""
    form = _minimal_form("a.png", "Daniel Jarvis")
    form.services_requested.dme = DMEDetails(
        requested=True, equipment_or_supplies="Wheelchair"
    )
    context = form_context(form)
    assert "DME: requested" in context
    assert "Wheelchair" in context


def test_form_context_enriches_known_icd_code():
    """A known ICD code gets its plain-language description added."""
    form = _minimal_form("a.png", "Daniel Jarvis")
    form.services_requested.service_lines[0].icd_code = "Z47.1"
    context = form_context(form)
    assert "Z47.1" in context
    assert "Aftercare following joint replacement surgery" in context


def test_form_context_falls_back_for_unknown_icd_code():
    """An unrecognized ICD code still shows the raw code, not nothing."""
    form = _minimal_form("a.png", "Daniel Jarvis")
    form.services_requested.service_lines[0].icd_code = "Z99.99"
    context = form_context(form)
    assert "Z99.99" in context
