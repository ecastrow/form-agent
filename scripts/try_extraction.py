"""Manual smoke test: run extraction against a real form file.

Handles both images and PDFs -- routes to extract_form_from_pdf()
for .pdf files (printing every page's form) or extract_form() for
a single image, same branching extraction_node uses internally.

Unlike the unit tests, this makes a real, billed OpenAI API call --
it's for checking actual extraction quality, not for CI/repeated
runs. Requires OPENAI_API_KEY in your .env file.

Run with:
    python scripts/try_extraction.py data/sample_daniel.png
    python scripts/try_extraction.py data/some_scanned_claim.pdf
"""

import json
import sys
from pathlib import Path

# Makes this script importable-and-runnable the same way regardless
# of how it's invoked (`python scripts/try_extraction.py`, IPython's
# %run, double-click, etc.) or what directory you're in when you
# run it. Without this, `python scripts/try_extraction.py` puts only
# the scripts/ folder on sys.path, not the project root -- so `src`
# wouldn't be found. Same issue as the earlier pytest
# ModuleNotFoundError; conftest.py fixed it for pytest, this line
# fixes it here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from src.nodes.extraction import extract_form, extract_form_from_pdf
from src.schemas import PriorAuthForm

load_dotenv()


def _print_form(form: PriorAuthForm) -> None:
    """Prints one form's full extracted JSON, plus escalation info."""
    print(f"--- {form.source_file} " + "-" * 40)
    print(json.dumps(form.model_dump(), indent=2, default=str))

    reasons = form.escalation_reasons()
    if reasons:
        print(f"\nNEEDS ESCALATION: {reasons}")
    else:
        print("\nNo escalation needed.")
    print()


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/try_extraction.py <file_path>")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    print(f"Extracting: {file_path}\n")

    if file_path.suffix.lower() == ".pdf":
        forms = extract_form_from_pdf(file_path)
    else:
        forms = [extract_form(file_path)]

    print(f"Extracted {len(forms)} form(s).\n")
    for form in forms:
        _print_form(form)


if __name__ == "__main__":
    main()
