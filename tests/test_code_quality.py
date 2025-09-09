import ast
from pathlib import Path
from typing import Dict, Set

from tests._helpers import issue_logger, repo_files


def _find_unused_imports(py_path: Path) -> Set[str]:
    try:
        tree = ast.parse(py_path.read_text(encoding="utf-8"))
    except Exception as e:
        issue_logger.log("code_quality", "warning", f"Failed to parse {py_path}: {e}", file=str(py_path))
        return set()

    imported: Set[str] = set()
    used: Set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                imported.add(alias.asname or alias.name.split(".")[0])

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            for alias in node.names:
                imported.add(alias.asname or alias.name)

        def visit_Name(self, node: ast.Name) -> None:
            used.add(node.id)

    Visitor().visit(tree)
    return {name for name in imported if name not in used}


def _find_unreachable_code(py_path: Path):
    issues = []
    try:
        tree = ast.parse(py_path.read_text(encoding="utf-8"))
    except Exception:
        return issues

    def scan_block(stmts):
        terminated = False
        for i, stmt in enumerate(stmts):
            if terminated:
                issues.append((getattr(stmt, "lineno", None), "Unreachable statement after return/raise/break/continue"))
            if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                terminated = True
            # Recurse into bodies
            for body_name in ("body", "orelse", "finalbody"):
                sub = getattr(stmt, body_name, None)
                if isinstance(sub, list):
                    scan_block(sub)

    scan_block(tree.body)
    return issues


def test_code_quality_static_analysis():
    """Static checks: unused imports and simple unreachable code detection.
    Logs issues to test_issues.log without failing tests.
    """
    src_root = Path(__file__).resolve().parent.parent / "src"
    if not src_root.exists():
        return

    for path in repo_files(".py"):
        # Only analyze source code, skip tests and scripts
        if not str(path).startswith(str(src_root)):
            continue
        
        # Skip __init__.py files as they commonly have intentional re-exports
        if path.name == "__init__.py":
            continue

        unused = _find_unused_imports(path)
        for name in sorted(unused):
            issue_logger.log(
                category="code_quality",
                severity="info",
                message=f"Unused import '{name}'",
                file=str(path),
            )

        unreachable = _find_unreachable_code(path)
        for lineno, msg in unreachable:
            issue_logger.log(
                category="code_quality",
                severity="warning",
                message=msg,
                file=str(path),
                line=lineno,
            )

    assert True  # Do not fail CI; issues are logged instead

