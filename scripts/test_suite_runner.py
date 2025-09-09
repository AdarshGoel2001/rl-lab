#!/usr/bin/env python3
"""
Orchestrate running the comprehensive test suite with pytest, capturing
JUnit XML and forwarding a concise summary. Does not modify repo files.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive test suite")
    parser.add_argument("--junit", default="reports/pytest_junit.xml", help="Path to JUnit XML output")
    parser.add_argument("--k", default=None, help="PyTest -k expression for selecting tests")
    parser.add_argument("--maxfail", default="1", help="Stop after N failures (pytest --maxfail)")
    parser.add_argument("--extras", nargs=argparse.REMAINDER, help="Additional pytest args", default=[])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    reports_dir = repo_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    junit_path = repo_root / args.junit
    junit_path.parent.mkdir(parents=True, exist_ok=True)

    pytest_args = [sys.executable, "-m", "pytest", "tests", "-q", f"--maxfail={args.maxfail}", f"--junitxml={junit_path}"]
    if args.k:
        pytest_args.extend(["-k", args.k])
    if args.extras:
        pytest_args.extend(args.extras)

    print("Running:", " ".join(map(str, pytest_args)))
    try:
        result = subprocess.run(pytest_args, cwd=repo_root, check=False)
        print(f"PyTest exited with code {result.returncode}")
        return result.returncode
    except FileNotFoundError:
        print("pytest not found. Please install pytest to run the suite.")
        return 127


if __name__ == "__main__":
    sys.exit(main())

