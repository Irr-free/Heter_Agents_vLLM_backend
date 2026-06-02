#!/usr/bin/env python3
"""Rerun the core-interference experiment into an isolated version2 directory."""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
BASE_SCRIPT = THIS_DIR / "core_interference_experiment.py"
VERSION2_DIR = THIS_DIR / "version2_run"
REQUESTED_PDF = THIS_DIR / "core_interference_tokens_per_sec_version2.pdf"


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "core_interference_experiment_base", BASE_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    base = load_base_module()
    VERSION2_DIR.mkdir(parents=True, exist_ok=True)

    base.OUT_DIR = VERSION2_DIR
    base.RESULT_JSON = VERSION2_DIR / "core_interference_results_version2.json"
    base.RESULT_CSV = VERSION2_DIR / "core_interference_raw_version2.csv"
    base.RESULT_MD = VERSION2_DIR / "core_interference_summary_version2.md"

    # Keep this rerun self-contained and avoid appending another long block to
    # the global iteration log. The per-run markdown summary is still written.
    base.append_iteration_record = lambda results: None

    base.main()

    tokens_pdf = VERSION2_DIR / "core_interference_tokens_per_sec.pdf"
    if not tokens_pdf.exists():
        raise RuntimeError(f"missing expected PDF: {tokens_pdf}")
    shutil.copy2(tokens_pdf, REQUESTED_PDF)
    print(f"wrote {REQUESTED_PDF}")


if __name__ == "__main__":
    main()
