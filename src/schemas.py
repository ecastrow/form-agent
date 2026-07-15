"""Pydantic schemas for the Intelligent Form Agent.

These models define the structured representation extracted from
ingested forms, plus the shared LangGraph state that flows between
nodes.

Currently modeled against the Texas Standard Prior Authorization
Request Form for Health Care Services (the sample forms provided for
this project). The extraction node prompts an LLM to fill this
schema from a form image/PDF; Pydantic validates the result before
it enters shared state.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, computed_field

# Deterministic escalation gate -- see technical_considerations.md,
# Section 11. A single knob, not a per-node magic number.
ESCALATION_CONFIDENCE_THRESHOLD: float = 0.75

# Fields whose absence or unreliability makes a form genuinely
# un-actionable for an insurer -- i.e. worth a human's time. A
# missing provider or missing requested service means there is
# nothing to authorize or nobody to authorize it for. Everything
# else (fax numbers, issuer name, secondary contacts, addresses) is
# still recorded in low_confidence_fields for transparency, but does
# NOT on its own trigger escalation. This keeps escalation rare,
# matching the project's goal of minimizing manual review.
CRITICAL_FIELD_PREFIXES: tuple[str, ...] = (
    "patient.name",
    "providers.requesting_provider.name",
    "providers.service_provider.name",
    "services_requested.service_lines",
    "services_requested.therapy",
)


# -----------------------------------------------------------------
# Section I -- Submission
# -----------------------------------------------------------------

class SubmissionInfo(BaseModel):
    issuer_name: Optional[str] = Field(
        default=None,
        description="Null if illegible, redacted, or not present.",
    )
    date: Optional[str] = Field(
        default=None,
        description="Submission date, as printed.",
    )


# -----------------------------------------------------------------
# Section II -- General Information
# -----------------------------------------------------------------

class GeneralInfo(BaseModel):
    review_type: Optional[Literal["Non-Urgent", "Urgent"]] = None
    clinical_reason_for_urgency: Optional[str] = None
    request_type: Optional[
        Literal["Initial Request", "Extension/Renewal/Amendment"]
    ] = None
    prev_auth_number: Optional[str] = None


# -----------------------------------------------------------------
# Section III -- Patient Information
# -----------------------------------------------------------------

class PatientInfo(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    dob: Optional[str] = None
    sex: Optional[Literal["Male", "Female", "Other", "Unknown"]] = None
    subscriber_name: Optional[str] = None
    member_or_medicaid_id: Optional[str] = None
    group_number: Optional[str] = None


# -----------------------------------------------------------------
# Section IV -- Provider Information
# -----------------------------------------------------------------

class ProviderDetail(BaseModel):
    name: Optional[str] = None
    npi: Optional[str] = None
    specialty: Optional[str] = None
    phone: Optional[str] = None
    fax: Optional[str] = None


class RequestingProviderDetail(ProviderDetail):
    """Left column of Section IV -- includes Contact Name/Phone."""

    contact_name: Optional[str] = None
    contact_phone: Optional[str] = None


class ServiceProviderDetail(ProviderDetail):
    """Right column of Section IV -- includes Primary Care Provider."""

    primary_care_provider_name: Optional[str] = None
    primary_care_provider_phone: Optional[str] = None
    primary_care_provider_fax: Optional[str] = None


class ProviderInfo(BaseModel):
    requesting_provider: RequestingProviderDetail = Field(
        default_factory=RequestingProviderDetail
    )
    service_provider: ServiceProviderDetail = Field(
        default_factory=ServiceProviderDetail
    )


# -----------------------------------------------------------------
# Section V -- Services Requested
# -----------------------------------------------------------------

class ServiceLine(BaseModel):
    planned_service_or_procedure: Optional[str] = None
    code: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    diagnosis_description: Optional[str] = None
    icd_code: Optional[str] = None


ServiceSetting = Literal[
    "Inpatient", "Outpatient", "Provider Office", "Observation",
    "Home", "Day Surgery", "Other",
]

TherapyType = Literal[
    "Physical Therapy", "Occupational Therapy", "Speech Therapy",
    "Cardiac Rehab", "Mental Health/Substance Abuse",
]


class TherapyDetails(BaseModel):
    types: list[TherapyType] = Field(default_factory=list)
    number_of_sessions: Optional[str] = None
    duration: Optional[str] = None
    frequency: Optional[str] = None


class HomeHealthDetails(BaseModel):
    requested: bool = False
    md_signed_order_attached: Optional[bool] = None
    nursing_assessment_attached: Optional[bool] = None
    number_of_visits: Optional[str] = None
    duration: Optional[str] = None
    frequency: Optional[str] = None


class DMEDetails(BaseModel):
    requested: bool = False
    md_signed_order_attached: Optional[bool] = None
    title_19_certification_attached: Optional[bool] = None
    equipment_or_supplies: Optional[str] = None
    duration: Optional[str] = None


class ServicesRequested(BaseModel):
    service_lines: list[ServiceLine] = Field(default_factory=list)
    settings: list[ServiceSetting] = Field(default_factory=list)
    therapy: Optional[TherapyDetails] = None
    home_health: Optional[HomeHealthDetails] = None
    dme: Optional[DMEDetails] = None


# -----------------------------------------------------------------
# Section VI -- Clinical Documentation
# -----------------------------------------------------------------

class ClinicalDocumentation(BaseModel):
    service_provider_address: Optional[str] = None


# -----------------------------------------------------------------
# Top-level form + extraction metadata
# -----------------------------------------------------------------

class FieldConfidence(BaseModel):
    """A single field flagged as untrustworthy during extraction."""

    field: str = Field(
        description="Dotted path, e.g. 'patient.dob'.",
    )
    reason: str = Field(
        description="Why flagged, e.g. 'illegible' or 'redacted'.",
    )


class PriorAuthForm(BaseModel):
    """A single Texas Standard Prior Auth Request Form, extracted."""

    source_file: str = Field(
        description="Filename or identifier of the source form.",
    )
    submission: SubmissionInfo = Field(default_factory=SubmissionInfo)
    general: GeneralInfo = Field(default_factory=GeneralInfo)
    patient: PatientInfo
    providers: ProviderInfo = Field(default_factory=ProviderInfo)
    services_requested: ServicesRequested = Field(
        default_factory=ServicesRequested
    )
    clinical_documentation: ClinicalDocumentation = Field(
        default_factory=ClinicalDocumentation
    )

    extraction_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall extraction confidence, 0-1.",
    )
    low_confidence_fields: list[FieldConfidence] = Field(
        default_factory=list
    )

    def _service_lines_missing(self) -> bool:
        return not self.services_requested.service_lines

    def _missing_provider_name(self) -> bool:
        """True if either provider's name is missing (not just both).

        Both the requesting provider (who's asking) and the service
        provider (who's delivering the service) matter for an
        insurer's decision -- one being present doesn't make the
        other's absence acceptable.
        """
        req = self.providers.requesting_provider.name
        svc = self.providers.service_provider.name
        return not req or not svc

    def _patient_name_missing(self) -> bool:
        return not self.patient.name

    def escalation_reasons(self) -> list[str]:
        """Specific, human-readable reasons this form needs review.

        An empty list means it does not need escalation. Kept
        separate from needs_escalation (rather than that property
        just returning True/False) so callers -- and the human
        reviewer -- can see WHY, not just that something was
        flagged.
        """
        reasons: list[str] = []

        if self.extraction_confidence < ESCALATION_CONFIDENCE_THRESHOLD:
            conf = self.extraction_confidence
            reasons.append(f"low overall confidence ({conf:.2f})")

        if self._patient_name_missing():
            reasons.append("no patient name found")

        if self._service_lines_missing():
            reasons.append("no requested service/procedure found")

        if self._missing_provider_name():
            reasons.append("missing requesting or service provider name")

        for fc in self.low_confidence_fields:
            if fc.field.startswith(CRITICAL_FIELD_PREFIXES):
                reasons.append(f"{fc.field} ({fc.reason})")

        return reasons

    @computed_field
    @property
    def needs_escalation(self) -> bool:
        """Deterministic gate -- a rule check, not a free LLM call.

        True whenever escalation_reasons() is non-empty. See that
        method for what actually triggers it, and
        technical_considerations.md Section 13 for the policy
        rationale (critical vs. non-critical fields).
        """
        return bool(self.escalation_reasons())



# -----------------------------------------------------------------
# Shared LangGraph state
# -----------------------------------------------------------------

class AgentState(TypedDict):
    # Conversation history; add_messages is the reducer that makes
    # this accumulate across turns instead of being overwritten --
    # what the LangGraph checkpointer persists for multi-turn
    # context.
    messages: Annotated[list, add_messages]

    # All forms ingested so far this session. Re-entrant ingestion:
    # appended to whenever a new file is uploaded, at any point in
    # the conversation (see technical_considerations.md, Sec. 11).
    forms: list[PriorAuthForm]

    # File paths uploaded but not yet run through extraction. The
    # extraction node consumes and clears this on each run -- how
    # re-entrant ingestion is actually triggered.
    pending_uploads: list[str]

    # True only for turns where turn_input.py's build_turn_update()
    # was given a new user message -- NOT the same as "messages is
    # non-empty", since messages accumulates forever via the
    # add_messages reducer above. This is graph.py's signal for
    # "should the query pipeline (router onward) run this turn",
    # distinct from "does conversation history exist at all".
    # Always explicitly set (both True/False) by build_turn_update,
    # never left stale by omission -- see that function's docstring.
    has_new_query: bool

    # Router output: what to do, and which forms it applies to.
    intent_action: Optional[Literal["summarize", "qa"]]
    # source_file values in scope; empty/None means all forms.
    intent_scope: Optional[list[str]]

    # Escalation state for the current turn.
    needs_escalation: bool
    escalation_reason: Optional[str]

    # Final answer for the current turn.
    response: Optional[str]
