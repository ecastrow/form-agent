"""Unit tests for src/schemas.py.

Run with: pytest -v
No network or LLM calls needed -- these test the schema itself, not
extraction.
"""

import pytest
from pydantic import ValidationError

from src.schemas import (
    FieldConfidence,
    PatientInfo,
    PriorAuthForm,
    ProviderInfo,
    RequestingProviderDetail,
    ServiceProviderDetail,
    ServiceLine,
    ServicesRequested,
)


def _sample_form(**overrides) -> PriorAuthForm:
    """Builds a form resembling the Daniel Jarvis sample form."""
    defaults = dict(
        source_file="0a01f77b_TX_page_1.png",
        patient=PatientInfo(
            name="Daniel Jarvis",
            phone="(378)-041-2101",
            dob="03/20/1992",
            sex="Unknown",
            member_or_medicaid_id="62106",
        ),
        providers=ProviderInfo(
            requesting_provider=RequestingProviderDetail(
                name="Elizabeth Foley"
            ),
            service_provider=ServiceProviderDetail(
                name="Leslie Johnson"
            ),
        ),
        services_requested=ServicesRequested(
            service_lines=[
                ServiceLine(
                    planned_service_or_procedure="Open placement",
                    code="44300",
                    start_date="11/20/2022",
                    end_date="11/26/2022",
                    diagnosis_description="Aftercare following",
                    icd_code="Z47.1",
                ),
            ],
            settings=["Outpatient"],
        ),
        extraction_confidence=0.95,
    )
    defaults.update(overrides)
    return PriorAuthForm(**defaults)


def test_valid_form_parses():
    """A fully valid form builds and needs no escalation."""
    form = _sample_form()
    assert form.patient.name == "Daniel Jarvis"
    line = form.services_requested.service_lines[0]
    assert line.icd_code == "Z47.1"
    assert form.needs_escalation is False
    assert form.escalation_reasons() == []


def test_missing_patient_name_triggers_escalation():
    """A missing patient name escalates instead of failing to parse.

    Previously patient.name was a required str, which forced the
    model to fabricate something rather than express absence (see
    technical_considerations.md, Sec. 21 -- the real bug this
    replaced). Now it's Optional, and absence is caught by our own
    structural check instead of Pydantic's required-field
    validation.
    """
    form = _sample_form(
        patient=PatientInfo(name=None, member_or_medicaid_id="62106")
    )
    assert form.needs_escalation is True
    assert "no patient name found" in form.escalation_reasons()


def test_low_aggregate_confidence_triggers_escalation():
    """Overall confidence below the threshold alone escalates."""
    form = _sample_form(extraction_confidence=0.4)
    assert form.needs_escalation is True
    assert "confidence" in form.escalation_reasons()[0]


def test_no_service_lines_triggers_escalation():
    """Nothing requested at all -- an insurer has nothing to act on."""
    form = _sample_form(
        services_requested=ServicesRequested(settings=["Outpatient"])
    )
    assert form.needs_escalation is True
    assert "no requested service" in form.escalation_reasons()[0]


def test_both_provider_names_missing_triggers_escalation():
    """Neither provider name present -- who is even asking?"""
    form = _sample_form(
        providers=ProviderInfo(
            requesting_provider=RequestingProviderDetail(name=None),
            service_provider=ServiceProviderDetail(name=None),
        )
    )
    assert form.needs_escalation is True
    reason = "missing requesting or service provider name"
    assert reason in form.escalation_reasons()


def test_missing_service_provider_name_triggers_escalation():
    """Requesting provider alone is no longer enough.

    Real case found in testing: service_provider.name was null
    (correctly, not flagged as low-confidence -- it was genuinely
    blank, not illegible) but escalation didn't fire under the old
    policy. Both names are now required.
    """
    form = _sample_form(
        providers=ProviderInfo(
            requesting_provider=RequestingProviderDetail(
                name="Elizabeth Foley"
            ),
            service_provider=ServiceProviderDetail(name=None),
        )
    )
    assert form.needs_escalation is True


def test_missing_requesting_provider_name_triggers_escalation():
    """Same rule the other way -- service provider alone isn't enough."""
    form = _sample_form(
        providers=ProviderInfo(
            requesting_provider=RequestingProviderDetail(name=None),
            service_provider=ServiceProviderDetail(
                name="Leslie Johnson"
            ),
        )
    )
    assert form.needs_escalation is True


def test_both_providers_present_does_not_escalate():
    """Both provider names present -- no escalation on this basis."""
    form = _sample_form()  # default fixture has both providers set
    assert form.needs_escalation is False


def test_non_critical_low_confidence_field_does_not_escalate():
    """A flagged issuer name (non-critical) should not escalate.

    Mirrors the real redacted Issuer Name on sample form 2: a
    routing/payer detail, not something that blocks clinical review.
    """
    form = _sample_form(
        low_confidence_fields=[
            FieldConfidence(
                field="submission.issuer_name",
                reason="redacted in source image",
            ),
        ],
    )
    assert form.needs_escalation is False
    assert form.escalation_reasons() == []


def test_critical_low_confidence_field_triggers_escalation():
    """An illegible provider name IS critical, unlike issuer name."""
    form = _sample_form(
        low_confidence_fields=[
            FieldConfidence(
                field="providers.requesting_provider.name",
                reason="illegible handwriting over name field",
            ),
        ],
    )
    assert form.needs_escalation is True
    reason = form.escalation_reasons()[0]
    assert "providers.requesting_provider.name" in reason
    assert "illegible handwriting" in reason


def test_flagged_therapy_field_triggers_escalation():
    """Therapy details are critical too, per user-stated priority."""
    form = _sample_form(
        low_confidence_fields=[
            FieldConfidence(
                field="services_requested.therapy.types",
                reason="ambiguous checkbox marking",
            ),
        ],
    )
    assert form.needs_escalation is True


def test_confidence_out_of_range_rejected():
    """extraction_confidence must be within 0.0-1.0."""
    with pytest.raises(ValidationError):
        _sample_form(extraction_confidence=1.5)


def test_service_setting_rejects_unknown_values():
    """Settings must come from the fixed checkbox list on the form."""
    with pytest.raises(ValidationError):
        _sample_form(
            services_requested=ServicesRequested(
                settings=["Not A Real Setting"]  # type: ignore
            )
        )
