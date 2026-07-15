"""Minimal ICD-10-CM code lookup for RAG-style diagnosis enrichment.

Exact-match retrieval, not embedding-based semantic search --
deliberately: ICD codes are precise identifiers, not free text.
"Z47.1" either matches the reference table or it doesn't; semantic
similarity doesn't help distinguish one code from another, so exact
lookup IS the appropriate retrieval method here, not a
simplification of "real" RAG.

This is the seam identified earlier (technical_considerations.md,
Sections 17/24): extending form_context() (src/nodes/shared.py) to
enrich icd_code values with plain-language descriptions before they
reach summarize_node/qa_node's prompts -- no other node or graph
change needed.

Reference set: the ICD-10-CM codes appearing in this project's real
and adversarial sample forms, verified individually against
icd10data.com / AAPC rather than generated from memory -- medical
codes are precise, and getting one wrong would defeat the purpose
of doing this via lookup instead of letting the LLM guess. NOT the
full official ICD-10-CM code set (70,000+ codes) -- deliberately
out of scope for "minimal RAG"; see technical_considerations.md.
"""

from __future__ import annotations

from typing import Optional

ICD_REFERENCE: dict[str, str] = {
    "Z47.1": "Aftercare following joint replacement surgery",
    "E10.69": (
        "Type 1 diabetes mellitus with other specified complication"
    ),
    "S82.201H": (
        "Unspecified fracture of shaft of right tibia, subsequent "
        "encounter for open fracture type I or II with delayed "
        "healing"
    ),
    "M14.87": (
        "Arthropathies in other specified diseases classified "
        "elsewhere, ankle and foot"
    ),
    "S43.1": "Subluxation and dislocation of acromioclavicular joint",
    "I75.011": "Atheroembolism of right upper extremity",
    "W89.0XXA": "Exposure to welding light (arc), initial encounter",
}


def lookup_icd_description(code: Optional[str]) -> Optional[str]:
    """Returns a plain-language description for an ICD-10-CM code.

    Exact match only (after normalizing whitespace/case) -- returns
    None if the code isn't in the reference set, rather than
    guessing. Callers should treat None as "no enrichment
    available" and fall back to showing the raw code, not silently
    drop it.
    """
    if not code:
        return None
    return ICD_REFERENCE.get(code.strip().upper())
