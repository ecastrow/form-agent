"""Shared helpers for the query-pipeline nodes (router, summarize, qa).

Centralized here rather than duplicated in, or privately
cross-imported between, individual node modules. Also the intended
seam for future ICD-code RAG enrichment (technical_considerations.md,
Section 17): form_context() is where a form's data becomes prompt
text, so adding looked-up diagnosis descriptions later means
extending this one function, not every node that uses it.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.icd_lookup import lookup_icd_description
from src.schemas import AgentState, PriorAuthForm


def latest_user_text(messages: list[BaseMessage]) -> str:
    """Finds the most recent human message's text content."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def conversation_history_text(messages: list[BaseMessage]) -> str:
    """Renders prior turns as simple "User: .../Assistant: ..." text.

    Quick-and-dirty by design: no windowing or summarization, just
    the full history rendered as plain text. Callers pass everything
    EXCEPT the current turn's message (typically messages[:-1]) --
    this function doesn't identify "current" vs "prior" itself.
    Good enough for a form-review assistant where conversations stay
    short; would need revisiting if that assumption ever changes.
    """
    lines = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"User: {message.content}")
        elif isinstance(message, AIMessage):
            lines.append(f"Assistant: {message.content}")
    return "\n".join(lines)


def resolve_forms_in_scope(state: AgentState) -> list[PriorAuthForm]:
    """Returns the forms a query applies to.

    intent_scope holds source_file values to filter to; None or
    empty means all forms currently in state -- the convention
    documented on AgentState itself.
    """
    forms = state.get("forms", [])
    scope = state.get("intent_scope")
    if not scope:
        return forms
    return [form for form in forms if form.source_file in scope]


def form_reference_list(state: AgentState) -> str:
    """Short filename + patient-name list, for the router's prompt."""
    lines = [
        f"- {form.source_file}: patient {form.patient.name}"
        for form in state.get("forms", [])
    ]
    return "\n".join(lines) if lines else "(no forms ingested yet)"


def form_context(form: PriorAuthForm) -> str:
    """One form's key data plus its escalation status, as text.

    Used identically by summarize_node and qa_node so they see the
    same data for the same form -- see the module docstring for
    why this is the intended RAG-enrichment seam. ICD codes are
    enriched with a plain-language description when available
    (lookup_icd_description), falling back to the raw code alone
    when not -- never silently dropped either way.
    """
    reasons = form.escalation_reasons()
    flag = f" [FLAGGED: {'; '.join(reasons)}]" if reasons else ""
    lines = [
        f"Form: {form.source_file}{flag}",
        f"Patient: {form.patient.name} (DOB {form.patient.dob})",
        (
            "Requesting provider: "
            f"{form.providers.requesting_provider.name}"
        ),
        f"Service provider: {form.providers.service_provider.name}",
        f"Review type: {form.general.review_type}",
    ]
    for line in form.services_requested.service_lines:
        icd_plain = lookup_icd_description(line.icd_code)
        icd_display = (
            f"{line.icd_code} ({icd_plain})" if icd_plain else line.icd_code
        )
        lines.append(
            f"  - {line.planned_service_or_procedure} "
            f"({line.code}): {line.diagnosis_description} "
            f"[{icd_display}], {line.start_date}-{line.end_date}"
        )
    return "\n".join(lines)
