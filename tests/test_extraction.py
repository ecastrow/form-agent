"""Unit tests for src/nodes/extraction.py.

Run with: pytest -v
No network or API key needed -- a fake LLM stands in for the real
OpenAI call, so these test extract_form()'s own logic (file
handling, source_file stamping, error routing), not the model's
extraction quality itself. PDF tests use a real (but tiny,
synthetically generated) PDF via PyMuPDF for the rendering step,
with extract_form itself still faked out.
"""

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from src.nodes.extraction import (
    extract_form,
    extract_form_from_pdf,
    extraction_node,
)
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


def _good_form(source_file: str = "whatever-the-model-guessed.png"):
    """Builds a minimal, valid, non-escalating PriorAuthForm."""
    return PriorAuthForm(
        source_file=source_file,
        patient=PatientInfo(name="Daniel Jarvis"),
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
                ),
            ],
        ),
        extraction_confidence=0.95,
    )


def test_extract_form_stamps_real_filename(tmp_path):
    """source_file always uses the real filename, not the model's guess."""
    image = tmp_path / "sample_form.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")  # minimal stand-in bytes

    fake_llm = _FakeLLM(_good_form())
    result = extract_form(image, llm=fake_llm)

    assert result.source_file == "sample_form.png"
    assert result.patient.name == "Daniel Jarvis"


def test_extract_form_display_name_overrides_filename(tmp_path):
    """display_name lets PDF pages name the source themselves."""
    image = tmp_path / "temp_render.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    fake_llm = _FakeLLM(_good_form())
    result = extract_form(
        image, llm=fake_llm, display_name="claim.pdf#page1"
    )

    assert result.source_file == "claim.pdf#page1"


def test_extract_form_missing_file_raises(tmp_path):
    """A path that doesn't exist should raise, not be swallowed."""
    missing = tmp_path / "does_not_exist.png"
    with pytest.raises(FileNotFoundError):
        extract_form(missing, llm=_FakeLLM(_good_form()))


def test_extract_form_from_pdf_calls_extract_form_per_page(
    tmp_path, monkeypatch
):
    """A 2-page PDF should be rendered and extracted page by page."""
    fitz = pytest.importorskip("fitz")

    pdf_path = tmp_path / "two_page.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()

    calls = []

    def fake_extract_form(path, llm=None, display_name=None):
        calls.append(display_name)
        return _good_form(source_file=display_name)

    monkeypatch.setattr(
        "src.nodes.extraction.extract_form", fake_extract_form
    )

    results = extract_form_from_pdf(pdf_path)

    assert len(results) == 2
    assert calls == ["two_page.pdf#page1", "two_page.pdf#page2"]


def test_extraction_node_appends_to_existing_forms(tmp_path, monkeypatch):
    """New extractions are appended to forms already in state."""
    image = tmp_path / "second_form.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    def fake_extract_form(path, llm=None, display_name=None):
        return _good_form(source_file=Path(path).name)

    monkeypatch.setattr(
        "src.nodes.extraction.extract_form", fake_extract_form
    )

    existing_form = _good_form(source_file="first_form.png")
    state = {
        "forms": [existing_form],
        "pending_uploads": [str(image)],
    }

    update = extraction_node(state)

    assert len(update["forms"]) == 2
    assert update["forms"][0].source_file == "first_form.png"
    assert update["forms"][1].source_file == "second_form.png"
    assert update["pending_uploads"] == []
    assert update["needs_escalation"] is False
    assert "second_form.png" in update["response"]
    assert isinstance(update["messages"][0], AIMessage)
    assert update["messages"][0].content == update["response"]


def test_extraction_node_routes_pdf_to_pdf_extractor(monkeypatch):
    """A .pdf upload routes through extract_form_from_pdf (multi-page)."""

    def fake_extract_form_from_pdf(path, llm=None):
        name = Path(path).name
        return [
            _good_form(source_file=f"{name}#page1"),
            _good_form(source_file=f"{name}#page2"),
        ]

    monkeypatch.setattr(
        "src.nodes.extraction.extract_form_from_pdf",
        fake_extract_form_from_pdf,
    )

    state = {"forms": [], "pending_uploads": ["claim.pdf"]}
    update = extraction_node(state)

    assert len(update["forms"]) == 2
    assert update["forms"][0].source_file == "claim.pdf#page1"
    assert update["needs_escalation"] is False


def test_extraction_node_flags_low_confidence_form(monkeypatch):
    """Low-confidence forms are kept, but flagged with a specific reason."""

    def fake_extract_form(path, llm=None, display_name=None):
        form = _good_form(source_file=Path(path).name)
        return form.model_copy(update={"extraction_confidence": 0.2})

    monkeypatch.setattr(
        "src.nodes.extraction.extract_form", fake_extract_form
    )

    state = {"forms": [], "pending_uploads": ["low_conf_form.png"]}
    update = extraction_node(state)

    assert len(update["forms"]) == 1
    assert update["needs_escalation"] is True
    assert "low_conf_form.png" in update["escalation_reason"]
    assert "confidence" in update["escalation_reason"]


def test_extraction_node_handles_validation_error(monkeypatch):
    """A schema-invalid extraction is dropped and flags escalation."""

    def fake_extract_form(path, llm=None, display_name=None):
        raise ValidationError.from_exception_data("PriorAuthForm", [])

    monkeypatch.setattr(
        "src.nodes.extraction.extract_form", fake_extract_form
    )

    state = {"forms": [], "pending_uploads": ["broken_form.png"]}
    update = extraction_node(state)

    assert update["forms"] == []
    assert update["needs_escalation"] is True
    assert "broken_form.png" in update["escalation_reason"]
    assert "broken_form.png" in update["response"]
    assert "No forms were successfully ingested" in update["response"]
