# Intelligent Form Agent — Technical Considerations & Tradeoffs

A reference record of the system's design decisions and the reasoning behind
them, organized by topic. This describes the current design — it is not a
turn-by-turn history of how the project was built.

---

## 1. Ingestion & Extraction

**Vision LLM over OCR, including a hybrid alternative that was considered and
rejected.** The sample forms are checkbox-heavy (sex, review type, service
setting, therapy type). Traditional OCR is unreliable at detecting checkbox
*state* — this is a known weak point relative to vision-LLM extraction, which
reads the image directly and maps it into a target schema in one step. A
template-matching hybrid (deterministic extraction for the known checkbox
grid, LLM only for free text) was considered for its lower cost and
predictability, but rejected in favor of a single pipeline — see the
normalization principle below.

**Three OCR/layout-parsing alternatives were researched for the checkbox
weak point above, none adopted for this release:**

- **Docling** (IBM, open-source, LangChain/LlamaIndex integration) —
  layout-aware checkbox/radio detection goes beyond raw Tesseract, but
  documentation flags struggles with scanned/handwritten/camera-captured
  input, close to this project's actual profile. Its LangChain integration
  targets RAG document-chunking, a different problem than field-level
  extraction into a validated schema.
- **LLMWhisperer** (Unstract) — not itself an LLM; advanced OCR plus a
  parsing/layout layer, per its own documentation. Marketed specifically
  for this project's exact domain (insurance/healthcare forms with
  handwritten fields and checkboxes), with form-mode output rendering
  checkbox state directly. Caveat: every source found was the vendor's own
  blog, with no independent benchmark located.
- **AWS Textract** — a dedicated Forms model with checkbox/key-value
  detection, HIPAA-eligible and PCI DSS certified. Evidence here is more
  balanced than the other two: AWS's own support community shows genuine
  accuracy complaints tied to image quality and formatting consistency,
  and AWS's own recommended fix for low-confidence cases is pairing with
  human review (Amazon Augmented AI) — echoing this project's own "don't
  trust confidence alone" escalation design.

None were benchmarked against this project's actual sample forms — the
appropriate next step before treating any as conclusive. Adopting any as a
first-pass filter alongside the vision-LLM path would reintroduce the
two-pipeline inconsistency tradeoff already rejected in the normalization
principle above.

**Every input format is normalized to an image and routed through one vision
extraction path** — PNG/JPG, scanned PDF, and even a text-native PDF with
real embedded text. A text-native PDF could technically skip vision entirely
(e.g. via `pdfplumber`/`PyPDF2` positional text parsing), but that would mean
maintaining two extraction pipelines that could behave inconsistently on
logically similar forms, and would need its own confidence semantics distinct
from the vision path's. Accepted tradeoff: a text-native PDF pays for a
vision LLM call it strictly didn't need, in exchange for one pipeline and one
consistent confidence/escalation model across every input type.

**PDF support**: `extract_form_from_pdf()` renders each page via PyMuPDF and
reuses `extract_form()` per page unchanged — no separate PDF-specific
extraction logic. PyMuPDF was chosen over `pdf2image` specifically to avoid a
system-level Poppler dependency, keeping "runs end-to-end with minimal setup"
true on a fresh machine.

**Out of scope, explicitly:** handwriting recognition (ICR). Printed-text
extraction via vision LLM is in scope; handwriting is a materially harder,
open-ended problem and is documented as a stated boundary rather than left
silently unsupported.

---

## 2. Structured Output & Schema Design

**Pydantic models + structured/tool-calling output**, not freeform "return
JSON" prompting — guarantees types and validation at extraction time rather
than downstream parsing failures. No fine-tuning: modern LLMs handle
structured extraction well when guided by schema and structured output, and
no labeled dataset of forms exists to justify fine-tuning's cost for this
project's scope.

**Design principle: any field representing "a value read from the document"
must be `Optional`,** so the model always has a valid way to report genuine
absence. A required (non-`Optional`) field for document-sourced data isn't a
safety mechanism — it removes the model's only honest option when a value is
genuinely illegible or missing, and increases the risk of the model
fabricating something (e.g. echoing a field's own printed label as if it were
the value) rather than preventing that risk. `str`-required with no default
is reserved only for fields the system sets itself (`source_file`) or values
the model is always asked to self-assess regardless of document content
(`extraction_confidence`). Absence of anything that matters is caught by the
system's own structural checks in `escalation_reasons()` (Section 4), never
by Pydantic's required-field validation.

**Provider fields are grouped to match the form's actual printed layout**,
not a flattened convenience structure: `RequestingProviderDetail` (left
column, includes Contact Name/Phone as printed there) and
`ServiceProviderDetail` (right column, includes the Primary Care Provider's
own name/phone/fax, printed separately from the service provider's own
contact details).

---

## 3. Agent Architecture: One Explicit Graph, Not Multi-Agent

**Explicit LangGraph state graph with conditional routing, not an autonomous
multi-agent system.** The project's core goal is minimizing human review
*reliably*. Autonomous multi-agent handoff — agents freely deciding which
tools or agents to call next — is more flexible for open-ended tasks but
sacrifices predictability and traceability, which is directly at odds with a
reliability goal: unpredictable routing becomes its own source of error
requiring review. An explicit graph means any output can be traced to exactly
which node fired and why — more debuggable, more testable, easier to defend.

**Current graph shape:** one `StateGraph` — `extraction` → (conditionally)
`router` → (conditionally) `summarize` or `qa` — compiled with a
`MemorySaver` checkpointer. Two nodes handle the query pipeline, not four:
the router outputs an `action` (`summarize` | `qa`) and a `scope` (which
forms), and both `summarize_node` and `qa_node` handle 1..N forms through the
same code path — the "holistic synthesis across forms" behavior is a
prompt-level instruction that activates when scope is larger than one form,
not separate node logic for single- vs. multi-form cases. `extraction_node`
also absorbs what would otherwise be a separate "confidence check" step:
each form's `needs_escalation` is a computed field derived directly from the
extraction result, so a standalone node would only re-read a value already
computed.

**Sequential, not parallel, when a turn has both a new upload and a new
question.** `extraction` always runs before `router`, guaranteeing the router
sees fully up-to-date `forms` rather than racing extraction for a form
uploaded in the same turn it's asked about.

**`has_new_query` is an explicit state field**, distinct from checking
whether `messages` is non-empty. `messages` accumulates forever via the
`add_messages` reducer, so from the second turn onward it's always
non-empty — it cannot by itself distinguish "conversation history exists"
from "this specific turn included a new question." `has_new_query` is set
explicitly on every call to `build_turn_update()` (both `True` and `False`
cases, never omitted, since LangGraph's partial-update semantics leave
omitted keys unchanged) and is what the graph's routing actually checks.

**Checkpointing uses an explicit msgpack allowlist**, not the default
permissive setting. `PriorAuthForm` is a custom type stored in
`state["forms"]` and checkpointed every turn; LangGraph's default serializer
warns (and in future versions will block) reconstruction of any custom type
it doesn't explicitly recognize — defense-in-depth against a compromised
checkpoint store being able to trigger arbitrary object reconstruction on
load. `graph.py` builds an explicit `JsonPlusSerializer(allowed_msgpack_modules=
[("src.schemas", "PriorAuthForm")])` and passes it to `MemorySaver(serde=...)`.
`LANGGRAPH_STRICT_MSGPACK=true` is recommended as an additional `.env`
setting — cheap defense-in-depth given the explicit allowlist already exists.

**Modularity:** node functions are independently testable units under
`/src/nodes/`, schemas in `/src/schemas.py`, graph wiring isolated in
`/src/graph.py`. Helpers shared by more than one node (`resolve_forms_in_scope`,
`latest_user_text`, `form_context`, `conversation_history_text`) live in
`/src/nodes/shared.py` rather than being duplicated or privately
cross-imported between node modules.

**`form_context()` is the single source of prompt data for `summarize_node`
and `qa_node`** — it covers every populated field under `services_requested`
(service lines, setting, therapy, home health, DME), not just service lines.
Sections with no real data (e.g. no therapy type checked) are omitted
entirely rather than shown as empty/`None`, so the prompt stays clean without
under-representing what the form actually contains.

---

## 4. Escalation Policy

**A deterministic gate, not a free LLM judgment call.** `needs_escalation` is
a computed field derived from `escalation_reasons()` — a method that returns
the actual list of specific reasons a form needs review, rather than a bare
boolean with no memory of why. Both are on `PriorAuthForm` itself, computed
from data the model has already produced, not decided independently by a
second LLM call.

**Escalation triggers, all structural/threshold-based:**
- Overall extraction confidence below threshold
- No requested service/procedure present at all (`service_lines` empty)
- Either provider's name missing — requesting provider (who's asking) and
  service provider (who's delivering care) both matter for an insurer's
  decision; one being present does not make the other's absence acceptable
- No patient name (a document-sourced field, so absence is caught here
  rather than by schema-level required-ness — see Section 2)
- A specifically **critical** field (patient identity, either provider's
  identity, or requested services/therapy) individually flagged as
  low-confidence by the model

**Non-critical fields are still recorded but do not trigger escalation on
their own** — e.g. an illegible issuer name or fax number. Escalating on any
flagged field, regardless of relevance, would conflict directly with the
project's "minimize manual review" goal, since uncertainty in non-essential
fields is routine, not exceptional. Implemented via a `CRITICAL_FIELD_PREFIXES`
allowlist checked against flagged field paths.

**Escalation payload includes context**, not just a bare flag: the extracted
(partial) structured data, a reference to the original form, the specific
per-field reason, and the conversation context (what the user was asking when
it triggered, if mid-conversation) — a human reviewer needs to see *why*,
not just that something was flagged.

**Open question, deliberately unresolved:** whether a service line missing
its diagnosis/ICD code specifically (as opposed to the service line existing
at all) should itself be escalation-worthy. Left open rather than expanding
the critical set unilaterally.

---

## 5. Multi-Turn Memory & the Query Pipeline

**Two separate mechanisms are both required for a follow-up question to
actually work** — storage and usage, easy to conflate:
- **Storage:** the checkpointer plus the `add_messages` reducer persist the
  full conversation across separate `graph.invoke()` calls, keyed by
  `thread_id`.
- **Usage:** `summarize_node`/`qa_node` actually read that history
  (`conversation_history_text()`) and include it in their own prompt, not
  just the newest message. Storage without usage would mean the history
  exists but the LLM never sees it.

**Full history, unwindowed — a deliberately simple choice.** No
summarization or truncation of older turns, since a form-review assistant's
conversations will realistically stay short; this assumption is stated
explicitly (in `conversation_history_text()`'s docstring) as something to
revisit if it stops holding, rather than left implicit. Both nodes assume the
last message in `messages` is the current turn's own message (added by
`turn_input.py` before the graph runs) and treat everything before it as
history — a documented assumption, verified by tests confirming the current
turn's own text isn't duplicated into the history block.

**Every terminal node appends an `AIMessage`, not just sets `response`.**
`extraction_node`, `summarize_node`, and `qa_node` all do this. Two reasons:
it's what makes the assistant's own prior replies available as conversation
history on later turns, and it prevents a stale `response` from a previous
turn being shown after an unrelated upload-only turn — `extraction_node`
always builds a short ingestion summary and sets it as `response`; if a
question also arrives the same turn, `summarize_node`/`qa_node` simply
overwrite it afterward in the same graph execution, since `response` isn't
reducer-accumulated. No extra graph logic is needed for that case — it falls
out of node execution order within a turn.

---

## 6. RAG for ICD Codes

**Exact-match dictionary lookup, not embedding-based semantic search — the
appropriate retrieval method for this data, not a simplified stand-in for
"real" RAG.** ICD-10-CM codes are precise identifiers; `"Z47.1"` either
matches a reference entry or it doesn't, and semantic similarity between code
strings doesn't correspond to semantic similarity between diagnoses.
Implemented as one new module (`src/icd_lookup.py`) and a single touch point
in the existing `form_context()` — no changes to either node, the graph, or
the schema, since prompt-context construction for both `summarize_node` and
`qa_node` was already centralized in one place.

**Reference set (7 codes, matching this project's sample forms) verified
individually against icd10data.com/AAPC, not generated from memory.**
Medical codes are precise; two of the seven, checked during verification,
would have been meaningfully wrong if pulled from unverified recall — one
differed in laterality, encounter type, and healing status. Verification
mattered specifically because the entire point of doing this via lookup
rather than letting the LLM guess is that the reference data itself has to
be correct. Unknown codes fall back to displaying the raw code, never
silently dropped and never fabricated.

**Scaling beyond the sample-data code set is explicitly out of scope for
this release**, and the reason isn't just "a bigger table" — it's a
different retrieval strategy. Exact match stays correct for the common case
(a clean, correctly-extracted code). Handling the full official ICD-10-CM
set (70,000+ codes) for arbitrary real-world forms would need embedding-based
semantic search over code *descriptions* running alongside exact match, to
cover two cases exact match can't: a code garbled by extraction (no exact
match exists at all), and forms that print only a free-text diagnosis
description with no clean code, requiring matching that description's
wording against official terminology. A hybrid strategy — exact match first,
falling back to embedding search over a vector store (Chroma, FAISS, pgvector)
on a miss — is the appropriate production design, deferred as a real scope
expansion rather than built against forms that don't exist in this project's
data.

---

## 7. Cost Management

**Key tradeoff:** multimodal LLM extraction (per-call token cost, scales
with volume, generalizes well) vs. deterministic/template tools (near-zero
marginal cost, brittle, requires ongoing engineering effort as layouts
change) — resolved in favor of the LLM path, per Section 1.

**Levers in use:**
- Cheaper model tier for simple tasks: `gpt-4o-mini` for router
  classification, `gpt-4o` reserved for extraction and answer/summary
  generation, which need stronger reasoning.
- Caching: each form is extracted once and reused across every subsequent
  question/summary, rather than re-extracted per query.

---

## 8. Domain Specificity (Healthcare / Insurance)

**Core architecture stays generic and schema-driven**, per the assignment's
explicit "wide variety of forms" framing — but the demo and select creative
extensions (ICD code explanation, urgency flagging, prior-auth-specific
insights) lean into the healthcare/insurance domain reflected in the actual
sample data. This shows domain awareness in the demo without narrowing the
underlying engineering solution.

---

## 9. Known Limitations

**LLM self-reported confidence does not reliably correlate with actual
correctness.** This is a general property of LLMs, not a bug specific to
this project, and was confirmed directly: extraction errors on real forms
(a misread checkbox, an incorrect field value) were observed alongside a
reported `extraction_confidence` of `1.0`, and explicit prompt-level
calibration guidance did not change this — a clean negative result, not an
ambiguous one. The escalation policy is deliberately built to not depend
solely on self-report: structural checks (Section 4) catch missing-critical-
data cases regardless of what confidence the model reports, and the same
form has been observed to produce different results across separate runs at
`temperature=0` — LLM APIs are not perfectly deterministic run-to-run, which
further underscores why escalation leans on structural presence checks
rather than a single self-reported score.

**Real fix identified, not built:** a second-pass verification node —
re-showing the model the image alongside its own extracted JSON and asking
it to check for disagreements — would likely be meaningfully more reliable
than self-report, since checking a specific claim against evidence is a
different task than rating one's own first-pass output. Cost: roughly 2x LLM
calls/latency per form. Deliberately deferred given the project's time
constraints; see Section 11.

**Text-native PDFs pay for a vision LLM call they don't strictly need** — an
accepted tradeoff for pipeline uniformity (Section 1).

**ICD RAG covers only the codes present in this project's sample data** —
see Section 6 for the scaling path.

---

## 10. Documentation & Deliverables

**Two documents, two purposes:** this file is the decision record (what was
chosen and why); `docs/architecture_walkthrough.md` is a mechanics trace of
how a single turn actually flows through the system (startup, routing,
multi-turn memory, escalation surfacing) — useful for understanding the
codebase, not for understanding why it's shaped this way.

**Notebooks, two of them, different purposes:**
`notebooks/demo_example_runs.ipynb` satisfies the assignment's required
demonstration (single-form QA, single-form summary, multi-form holistic
answer); `notebooks/adversarial_escalation_tests.ipynb` exercises the
escalation policy against real broken forms (missing provider, missing
patient name, missing services, across PNG and PDF) as engineering-rigor
evidence beyond the minimum requirement. No separate "extraction only"
notebook — `scripts/try_extraction.py` already covers that ground.

**Sample form images are not committed to the repository**, out of caution
around PII, even though the samples used are believed to be synthetic or
fictitious — `data/` ships with only a placeholder so the expected project
structure is visible; real sample forms are shared separately. Stated
explicitly in the README rather than left as an unexplained empty folder.

---

## 11. Open / Future Work

- **Verification-pass node** for extraction confidence calibration (Section 9)
- **ICD RAG scaling**: hybrid exact-match + embedding search over the full
  ICD-10-CM code set (Section 6)
- **Diagnosis-code-level critical field**: whether a service line missing its
  ICD code specifically should be escalation-worthy (Section 4)
- **Frontend** (optional, minimal): deferred until backend is complete and
  demonstrated
