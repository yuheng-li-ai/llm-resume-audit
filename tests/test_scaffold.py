"""Phase 0 scaffold smoke test.

Confirms the `llm_audit` package imports cleanly under the configured Python
runtime. Lets `pytest` exit zero on the empty scaffold so CI gates pass.
"""

from __future__ import annotations


def test_package_imports() -> None:
    import llm_audit  # noqa: F401


def test_subpackages_import() -> None:
    import llm_audit.analysis  # noqa: F401
    import llm_audit.scoring  # noqa: F401
    import llm_audit.utils  # noqa: F401
