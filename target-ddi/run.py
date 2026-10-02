#!/usr/bin/env python3
"""Command-line entry point. Run `python run.py --help` for usage."""
import argparse
import sys

from targetddi.runner import SECTIONS, run


def main():
    p = argparse.ArgumentParser(
        description="Reproduce the TARGET-DDI paper. Every section is resumable: finished units are skipped.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="sections:\n" + "\n".join(
            f"  {k:17s} {d}{'' if in_all else '  [not in all]'}" for k, (d, in_all) in SECTIONS.items()))
    p.add_argument("--workdir", default="ddi_work", help="output directory (default: ./ddi_work)")
    p.add_argument("--sections", default="all",
                   help="comma-separated section names, 'all', or 'none' (default: all)")
    p.add_argument("--fast", action="store_true", help="short smoke test: few epochs, one seed")
    p.add_argument("--budget-hours", type=float, default=1e6,
                   help="stop cleanly before starting a unit once this many hours have passed")
    p.add_argument("--resume-from", action="append", default=[],
                   help="copy finished work from an earlier output directory (repeatable)")
    p.add_argument("--selftest-only", action="store_true",
                   help="run setup and the synthetic self-test only (no downloads, no training)")
    a = p.parse_args()

    if a.sections == "all":
        sections = {k for k, (_, in_all) in SECTIONS.items() if in_all}
    elif a.sections == "none":
        sections = set()
    else:
        sections = {s.strip() for s in a.sections.split(",") if s.strip()}
        unknown = sections - set(SECTIONS)
        if unknown:
            p.error(f"unknown section(s): {', '.join(sorted(unknown))}")
    run(a.workdir, sections, fast=a.fast, budget_hours=a.budget_hours,
        resume_from=a.resume_from, selftest_only=a.selftest_only)


if __name__ == "__main__":
    sys.exit(main())
