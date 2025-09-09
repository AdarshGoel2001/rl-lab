#!/usr/bin/env python3
"""
Run the comprehensive test suite, aggregate logs, and snapshot artifacts.

Steps:
1) Optionally clear prior test_issues.log
2) Run pytest across tests/ with JUnit XML
3) Generate Markdown report from logs and JUnit XML
4) Snapshot artifacts into reports/session_<timestamp>/

Usage examples:
  python scripts/run_comprehensive_tests.py
  python scripts/run_comprehensive_tests.py --k parallel --maxfail 1
  python scripts/run_comprehensive_tests.py --extras -vv
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full RL lab test suite and aggregate logs")
    p.add_argument("--clear-log", action="store_true", help="Remove existing test_issues.log before running")
    p.add_argument("--junit", default="reports/pytest_junit.xml", help="Path to JUnit XML output")
    p.add_argument("--k", default=None, help="PyTest -k expression to select tests")
    p.add_argument("--maxfail", default="1", help="pytest --maxfail value")
    p.add_argument("--extras", nargs=argparse.REMAINDER, default=[], help="Additional pytest args (after --extras)")
    return p.parse_args()


def run_pytest(repo_root: Path, junit_path: Path, k: str | None, maxfail: str, extras: list[str]) -> int:
    cmd = [sys.executable, "-m", "pytest", "tests", "-q", f"--maxfail={maxfail}", f"--junitxml={junit_path}"]
    if k:
        cmd.extend(["-k", k])
    if extras:
        cmd.extend(extras)
    print("[run]", " ".join(map(str, cmd)))
    try:
        res = subprocess.run(cmd, cwd=repo_root, check=False)
        return res.returncode
    except FileNotFoundError:
        print("[error] pytest not found. Install pytest to run the suite.")
        return 127


def run_report(repo_root: Path) -> int:
    cmd = [sys.executable, str(repo_root / "scripts/generate_test_report.py")]
    print("[run]", " ".join(map(str, cmd)))
    res = subprocess.run(cmd, cwd=repo_root, check=False)
    return res.returncode


def load_junit_summary(junit_path: Path) -> dict:
    if not junit_path.exists():
        return {}
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(junit_path).getroot()
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


def snapshot_artifacts(repo_root: Path, junit_path: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = repo_root / "reports" / f"session_{timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)

    # Copy artifacts if present
    issues_src = repo_root / "test_issues.log"
    report_md = repo_root / "reports/test_report.md"
    for src in [issues_src, junit_path, report_md]:
        if src.exists():
            shutil.copy2(src, session_dir / src.name)
    return session_dir


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    junit_path = repo_root / args.junit
    junit_path.parent.mkdir(parents=True, exist_ok=True)

    if args.clear_log:
        log_path = repo_root / "test_issues.log"
        if log_path.exists():
            print(f"[info] Removing existing log: {log_path}")
            try:
                log_path.unlink()
            except Exception:
                pass

    code = run_pytest(repo_root, junit_path, args.k, args.maxfail, args.extras)
    rep_code = run_report(repo_root)
    session_dir = snapshot_artifacts(repo_root, junit_path)

    junit_summary = load_junit_summary(junit_path)
    print("\n=== Test Run Summary ===")
    if junit_summary:
        print(
            f"Tests: {junit_summary.get('tests',0)}, Failures: {junit_summary.get('failures',0)}, "
            f"Errors: {junit_summary.get('errors',0)}, Skipped: {junit_summary.get('skipped',0)}, "
            f"Time: {junit_summary.get('time',0.0):.2f}s"
        )
    print(f"JUnit XML: {junit_path}")
    print(f"Issues log: {repo_root / 'test_issues.log'}")
    print(f"Markdown report: {repo_root / 'reports/test_report.md'}")
    print(f"Session snapshot: {session_dir}")
    print("========================\n")

    # Return pytest exit code (non-zero if failures)
    return code if code != 0 else rep_code


if __name__ == "__main__":
    sys.exit(main())
