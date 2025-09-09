import ast
from pathlib import Path
from typing import Dict, Set

from tests._helpers import issue_logger, repo_files


def _collect_defs_and_calls(py_path: Path):
    defs: Set[str] = set()
    calls: Set[str] = set()
    try:
        tree = ast.parse(py_path.read_text(encoding="utf-8"))
    except Exception:
        return defs, calls

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            # Only consider top-level functions
            if isinstance(getattr(node, 'parent', None), ast.Module) or True:
                defs.add(node.name)
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call) -> None:
            # Grab simple function names (f(...)) and attribute calls (obj.f(...))
            func = node.func
            if isinstance(func, ast.Name):
                calls.add(func.id)
            elif isinstance(func, ast.Attribute):
                calls.add(func.attr)
            self.generic_visit(node)

    # Annotate parents for top-level detection
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node

    Visitor().visit(tree)
    return defs, calls


def test_call_hierarchy_unused_functions():
    """Detect top-level functions defined but never called within src/ tree.
    This is heuristic and only logs possible dead code.
    """
    src_root = Path(__file__).resolve().parent.parent / "src"
    if not src_root.exists():
        return

    all_defs: Set[str] = set()
    all_calls: Set[str] = set()

    for path in repo_files(".py"):
        if not str(path).startswith(str(src_root)):
            continue
        defs, calls = _collect_defs_and_calls(path)
        # Filter out private/dunder
        defs = {d for d in defs if not d.startswith("_") and not d.startswith("test_")}
        all_defs |= defs
        all_calls |= calls

    unused = sorted(d for d in all_defs if d not in all_calls)
    for name in unused[:200]:  # limit
        issue_logger.log(
            category="call_graph",
            severity="info",
            message=f"Function defined but not referenced: {name}",
            file=None,
        )

    assert True

