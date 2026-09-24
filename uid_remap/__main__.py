"""CLI: python -m uid_remap IN_DIR OUT_DIR [options]

Only performs the requested UID transformation. Writes the remapped
files, the merged old->new mapping (reusable by the next batch via
--mapping-in), and a JSON report with the change list plus any
unclassified private UI fields.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .assigners import chained_assigner, hash_assigner
from .remapper import (
    RemapValidationError,
    load_mapping,
    remap_batch,
    save_mapping,
    save_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="uid_remap",
        description="Batch-remap DICOM instance UIDs, keeping references consistent.",
    )
    parser.add_argument("input_dir", help="directory of input .dcm files")
    parser.add_argument("output_dir", help="directory for remapped files")
    parser.add_argument("--mapping-in", help="JSON old->new map from a previous batch")
    parser.add_argument("--mapping-out",
                        help="where to write the merged mapping "
                             "(default: <output_dir>/uid_mapping.json)")
    parser.add_argument("--report",
                        help="where to write the change/unclassified report "
                             "(default: <output_dir>/remap_report.json)")
    parser.add_argument("--root", default="2.25",
                        help="UID root for generated UIDs (default 2.25)")
    parser.add_argument("--salt", default="",
                        help="salt for the deterministic UID generator")
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    inputs = sorted(p for p in input_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    if not inputs:
        print(f"no input files in {input_dir}", file=sys.stderr)
        return 2

    mapping_in = load_mapping(args.mapping_in) if args.mapping_in else None
    assign = chained_assigner(mapping_in or {}, hash_assigner(args.root, args.salt))

    try:
        result = remap_batch(inputs, args.output_dir, assign, mapping_in)
    except RemapValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    mapping_out = Path(args.mapping_out) if args.mapping_out else Path(args.output_dir) / "uid_mapping.json"
    report_path = Path(args.report) if args.report else Path(args.output_dir) / "remap_report.json"
    save_mapping(result.mapping, mapping_out)
    save_report(result, report_path)

    print(f"remapped {len(result.output_files)} file(s) -> {args.output_dir}")
    print(f"mapping ({len(result.mapping)} UIDs): {mapping_out}")
    print(f"changes: {len(result.changes)}, "
          f"unclassified private UI fields: {len(result.unclassified)}")
    for warning in result.warnings:
        print(f"warning: {warning}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

