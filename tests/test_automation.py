from pathlib import Path

from scripts.generate_test_report import main as generate_report_main


def test_test_suite_scripts_exist():
    assert Path("scripts/test_suite_runner.py").exists()
    assert Path("scripts/generate_test_report.py").exists()


def test_generate_report_creates_markdown(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parent.parent
    # Create a tiny issues log in repo root
    log_path = repo_root / "test_issues.log"
    log_path.write_text('{"ts":"t","category":"code_quality","severity":"info","message":"m"}\n')
    # Run report
    generate_report_main()
    assert (repo_root / "reports/test_report.md").exists()

