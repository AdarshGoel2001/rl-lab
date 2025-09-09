#!/usr/bin/env python3
"""
Aggregate test outputs into a concise Markdown report using:
- JSONL issue logs from test_issues.log
- Optional JUnit XML from pytest
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict


def load_issues(path: Path) -> list[dict[str, Any]]:
    issues = []
    if not path.exists():
        return issues
    with open(path, "r") as f:
        for line in f:
            try:
                issues.append(json.loads(line))
            except Exception:
                continue
    return issues


def load_junit(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(path).getroot()
        # Handle either <testsuite ...> or <testsuites> with nested suites
        if root.tag == "testsuite":
            return {
                "tests": int(root.attrib.get("tests", 0)),
                "errors": int(root.attrib.get("errors", 0)),
                "failures": int(root.attrib.get("failures", 0)),
                "skipped": int(root.attrib.get("skipped", 0)),
                "time": float(root.attrib.get("time", 0.0)),
            }
        suites = root.findall("testsuite")
        if suites:
            total = {"tests": 0, "errors": 0, "failures": 0, "skipped": 0, "time": 0.0}
            for s in suites:
                total["tests"] += int(s.attrib.get("tests", 0))
                total["errors"] += int(s.attrib.get("errors", 0))
                total["failures"] += int(s.attrib.get("failures", 0))
                total["skipped"] += int(s.attrib.get("skipped", 0))
                total["time"] += float(s.attrib.get("time", 0.0))
            return total
        return {}
    except Exception:
        return {}


def render_markdown(issues: list[dict[str, Any]], junit: Dict[str, Any]) -> str:
    cat_counts = Counter(i.get("category", "unknown") for i in issues)
    sev_counts = Counter(i.get("severity", "info") for i in issues)

    lines = []
    lines.append("# Test Suite Report")
    if junit:
        lines.append(
            f"PyTest Summary: {junit.get('tests',0)} tests, {junit.get('failures',0)} failures, {junit.get('errors',0)} errors, {junit.get('skipped',0)} skipped, time {junit.get('time',0.0):.2f}s"
        )
    lines.append("")
    lines.append("## Issue Summary")
    lines.append("- By Category: " + ", ".join(f"{k}: {v}" for k, v in cat_counts.most_common()))
    lines.append("- By Severity: " + ", ".join(f"{k}: {v}" for k, v in sev_counts.most_common()))
    lines.append("")
    lines.append("## Recent Issues (up to 50)")
    for rec in issues[-50:]:
        where = (rec.get("file") or "?") + (":" + str(rec.get("line")) if rec.get("line") else "")
        lines.append(f"- [{rec.get('severity','info')}] {rec.get('category','misc')} @ {where}: {rec.get('message','')} ")
    return "\n".join(lines)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    issues_path = repo_root / "test_issues.log"
    junit_path = repo_root / "reports/pytest_junit.xml"
    report_md_path = repo_root / "reports/test_report.md"

    issues = load_issues(issues_path)
    junit = load_junit(junit_path)
    md = render_markdown(issues, junit)

    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    report_md_path.write_text(md)
    print(f"Wrote report: {report_md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
