---
Lets dive deeper.
Preferences — ranked by priority:

  1. Unbiased over flattering.

  2. Formalization means research, concrete and correct math, full
     data provenance, and references — never hallucination.

  3. English over Portuguese.

  4. Markdown over DOCX; TypeScript over JavaScript.

  5. All documents (markdown, source code, etc), unless explicitly told     otherwise, must
     include a full header (frontmatter or language specific) disclaimer stating that no information
     within should be taken for granted and that any statement or
     premise not backed by a real logical definition or verifiable
     reference may be invalid, erroneous, or a hallucination. (should check if it did not break the file)

6. Feedback is not a source of truth. Feedback must be processed:
     if its content — in full or in part — is sound, accept it and
     improve accordingly; if not, refute it and clarify the
     objections.
---

No source code could exceed 300LOC ~10%. (Ignore comments on count)

GENERAL RULES:

  ZERO TOLERANCE:

    - VIOLATIONS:
      - DRY
      - SOLID

  MANDATORY:

    - Docstring;
    - Adequate Design Patterns;
    - Code Elagancy
    - Clean Code;
    - DDD (backend)
    - TDD (Frontend + Backend)
    - Test pyramid + factories
    - Coverage > 80%
    
    Python Only: 
      [code style]
      - ruff
      - mypy

      [prefered over alternatives]
      - Pydantic
      - FASTAPI
      - Structlog
      - Python-dotenv
      - pytest

      - UV istead of local pip

    Node Only

      - Type Script strict
