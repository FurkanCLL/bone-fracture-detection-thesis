from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import run_audit


# Defines the paths used by the read-only Phase 1 audit.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit a raw YOLO object-detection dataset without modifying it.")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"), help="Directory containing raw dataset exports.")
    parser.add_argument("--output", type=Path, default=Path("outputs/data_quality/dataset_audit"), help="Directory for machine-readable artifacts and review images.")
    parser.add_argument("--report", type=Path, default=Path("docs/DATASET_AUDIT_REPORT.md"), help="Markdown report path.")
    parser.add_argument("--thesis-context", type=Path, default=Path("docs/THESIS.md"), help="Project context used only for previous-statistic comparison.")
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    summary = run_audit(arguments.raw_root, arguments.output, arguments.report, arguments.thesis_context)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
