"""Shared pytest configuration for the starling test suite."""

from __future__ import annotations


def pytest_collection_modifyitems(config, items):
    """Run the issue #167 coverage-accounting tests last.

    They must observe the parity gates' executed records AFTER the gates have
    run. Both modules are imported at collection time, but with an explicit
    file order like
      pytest tests/test_native_parity_contract.py tests/test_ggml_parity.py
    the accounting test would run before any gate and see empty records.
    """
    accounting = {
        "test_parity_coverage_accounting",             # tests/test_ggml_parity.py
        "test_required_targets_never_silently_pass",   # tests/test_native_parity_contract.py
    }
    tail = [it for it in items if it.name.split("[", 1)[0] in accounting]
    if tail:
        items[:] = [it for it in items
                    if it.name.split("[", 1)[0] not in accounting] + tail
