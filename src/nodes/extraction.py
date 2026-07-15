"""Extraction node: turns a form file into validated form(s).

Uses OpenAI's vision-capable structured outputs (gpt-4o) to read a
form image directly and fill the Pydantic schema in one call, rather
than a separate OCR step -- see technical_considerations.md,
Section 1, for why.

PDFs are supported by rendering each page to an image (via PyMuPDF)
and reusing the exact same image extraction path per page -- there
is no separate "PDF extraction" logic, just a page-rendering step
in front of the same function.
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from src.schemas import AgentState, PriorAuthForm

EXTRACTION_MODEL = "gpt-4o"
PDF_RENDER_DPI = 200

EXTRACTION_SYSTEM_PROMPT = """
You are extracting structured data from a health insurance prior
authorization request form. Read the form image carefully and fill
in the given schema.

General rules:
- Only report what is actually printed, typed, or checked on the
  form. Never guess or fabricate a value.
- If a field is blank, not applicable, or the section was not
  filled in, leave it as None / empty -- do not invent a value.
- If the space for a value is blocked, blurred, crossed out, or
  otherwise covered -- even if the field's own printed label is
  still visible -- do not report the label text itself as if it
  were the value (e.g. do not report "Name:" as somebody's name).
  Leave the field as None and add it to low_confidence_fields with
  a reason such as "blocked" or "no value visible near label".
- If a field is present but illegible, redacted, cut off, or
  otherwise unreadable, leave the field as None AND add an entry
  to low_confidence_fields naming the field's dotted path and the
  reason (e.g. "illegible", "redacted", "cut off").

Reading checkboxes carefully:
- The form has SEVERAL separate checkbox groups (sex, review type,
  request type, service setting, therapy type, among others).
  Before reporting any option as checked, confirm which printed
  group label it belongs to, and match its own printed text
  exactly -- do not let a mark near one group get attributed to a
  different, nearby group.
- A genuine mark is a clear, deliberate checkmark, X, filled box,
  or dot covering a meaningful portion of the box. Faint smudges,
  stray partial pixels, print artifacts, or bleed-through from the
  other side of the page are NOT marks -- do not report these as
  checked.
- Sex, review type, and request type allow only ONE selection
  each. If more than one option in such a group looks marked, or a
  mark is faint/ambiguous rather than clearly one option, do NOT
  confidently pick one -- add that field to low_confidence_fields
  with reason "ambiguous checkbox marking" and lower
  extraction_confidence to reflect the real uncertainty.
- Therapy type and service setting allow multiple selections, but
  are separate groups from each other -- verify each checked item
  against its own group's printed label, not an adjacent group.
- For a horizontal row of checkbox options, count positions left
  to right and match each mark to its exact position and printed
  label. Do not attribute a mark to the nearest-looking or most
  contextually plausible option -- match by position, not guess.

Confidence calibration:
- extraction_confidence must reflect genuine, field-by-field
  uncertainty -- it is not a general impression of overall image
  quality. A form can be fully legible and still contain one
  ambiguous checkbox; that alone should lower
  extraction_confidence, not be outweighed by how clear the rest
  of the page is.
- Before finalizing, re-check every checkbox group you marked
  against its own printed label one more time. If filled-in
  details (e.g. number of sessions, duration) suggest a group
  should have something checked but you found nothing clearly
  marked, or vice versa, treat this as a sign you may have
  misread the group -- lower confidence and/or flag the field.
""".strip()


def _encode_image(image_path: Path) -> tuple[str, str]:
    """Base64-encodes an image and returns (data, mime_type)."""
    data = image_path.read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return encoded, mime


def extract_form(
    image_path: str | Path,
    llm: Optional[ChatOpenAI] = None,
    display_name: Optional[str] = None,
) -> PriorAuthForm:
    """Extracts one form image into a validated PriorAuthForm.

    Args:
        image_path: path to a single-page form image (the file
            actually read from disk -- may be a temp render of a
            PDF page).
        llm: optional pre-built chat model. Exists so tests can
            inject a fake and verify this function's own logic
            without a real API call -- see tests/test_extraction.py.
        display_name: what to stamp as source_file. Defaults to
            image_path's own filename; PDF extraction overrides
            this to something like "claim.pdf#page1" so the origin
            is traceable even though the actual bytes came from a
            temporary render.

    Raises:
        pydantic.ValidationError if the model's output doesn't
        satisfy the schema. Callers (extraction_node) should catch
        this and route to escalation rather than let it propagate.
    """
    image_path = Path(image_path)
    encoded, mime = _encode_image(image_path)

    llm = llm or ChatOpenAI(model=EXTRACTION_MODEL, temperature=0)
    structured_llm = llm.with_structured_output(PriorAuthForm)

    messages = [
        SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Extract this prior authorization form.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}"},
                },
            ]
        ),
    ]

    result = structured_llm.invoke(messages)
    # The model may fabricate something for source_file since it's
    # a required field it has no real way to know -- we always
    # overwrite it ourselves.
    stamped_name = display_name or image_path.name
    return result.model_copy(update={"source_file": stamped_name})


def extract_form_from_pdf(
    pdf_path: str | Path,
    llm: Optional[ChatOpenAI] = None,
) -> list[PriorAuthForm]:
    """Extracts every page of a PDF, one PriorAuthForm per page.

    Renders each page to a PNG at PDF_RENDER_DPI and calls
    extract_form() on each render in turn -- no separate extraction
    logic for PDFs, just a rendering step in front of the same
    image pipeline. Each form's source_file is stamped as
    "{filename}#page{n}" so a multi-page submission stays
    traceable back to its origin and page number.
    """
    pdf_path = Path(pdf_path)
    forms: list[PriorAuthForm] = []

    doc = fitz.open(pdf_path)
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            for page_index, page in enumerate(doc, start=1):
                pixmap = page.get_pixmap(dpi=PDF_RENDER_DPI)
                temp_path = Path(tmp_dir) / f"page_{page_index}.png"
                pixmap.save(temp_path)

                display_name = f"{pdf_path.name}#page{page_index}"
                forms.append(
                    extract_form(
                        temp_path,
                        llm=llm,
                        display_name=display_name,
                    )
                )
    finally:
        doc.close()

    return forms


def extraction_node(state: AgentState) -> dict:
    """Ingestion graph node: extracts every pending upload.

    Re-entrant -- safe to call any time new files appear in
    pending_uploads, not just once at session start (see
    technical_considerations.md, Section 11).

    Routes .pdf files through extract_form_from_pdf (one file can
    yield multiple forms, one per page); everything else through
    extract_form directly.

    Note: this also absorbs the "confidence check" step from the
    original ingestion diagram -- each form's own needs_escalation
    (a computed field on PriorAuthForm) already IS that check, so a
    separate node would just duplicate it.
    """
    new_forms: list[PriorAuthForm] = []
    escalation_reasons: list[str] = []

    for file_path in state.get("pending_uploads", []):
        path = Path(file_path)
        try:
            if path.suffix.lower() == ".pdf":
                extracted = extract_form_from_pdf(path)
            else:
                extracted = [extract_form(path)]
        except ValidationError as exc:
            escalation_reasons.append(
                f"{file_path}: extraction failed schema validation "
                f"({exc.error_count()} error(s))"
            )
            continue

        for form in extracted:
            new_forms.append(form)
            reasons = form.escalation_reasons()
            if reasons:
                escalation_reasons.append(
                    f"{form.source_file}: " + "; ".join(reasons)
                )

    summary = _ingestion_summary(new_forms, escalation_reasons)

    return {
        "forms": state.get("forms", []) + new_forms,
        "pending_uploads": [],
        "needs_escalation": bool(escalation_reasons),
        "escalation_reason": "; ".join(escalation_reasons) or None,
        # Overwritten later in the same turn by summarize_node/
        # qa_node if a question also arrives this turn (see
        # graph.py) -- if not, this is what the user actually sees
        # as the response to an upload-only turn, instead of a
        # stale response left over from a previous turn.
        "response": summary,
        "messages": [AIMessage(content=summary)],
    }


def _ingestion_summary(
    new_forms: list[PriorAuthForm],
    escalation_reasons: list[str],
) -> str:
    """Builds a short, human-readable summary of what was ingested."""
    if not new_forms:
        if escalation_reasons:
            issues = "; ".join(escalation_reasons)
            return f"No forms were successfully ingested. Issues: {issues}"
        return "No forms were ingested."

    names = ", ".join(form.source_file for form in new_forms)
    summary = f"Ingested {len(new_forms)} form(s): {names}."
    if escalation_reasons:
        summary += f" Flagged for review: {'; '.join(escalation_reasons)}"
    return summary
