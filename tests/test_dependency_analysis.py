import ast
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from tests._helpers import issue_logger, repo_files


def _module_name_from_path(path: Path, src_root: Path) -> str:
    rel = path.relative_to(src_root)
    parts = rel.with_suffix("").parts
    return "src." + ".".join(parts)


def _collect_import_graph(src_root: Path) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = defaultdict(set)
    path_by_mod: Dict[str, Path] = {}
    for path in src_root.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        mod = _module_name_from_path(path, src_root)
        path_by_mod[mod] = path
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name.startswith("src."):
                        graph[mod].add(name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("src."):
                    graph[mod].add(node.module)
    return graph


def _find_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
    visited: Set[str] = set()
    stack: Set[str] = set()
    cycles: List[List[str]] = []

    def dfs(node: str, path: List[str]):
        visited.add(node)
        stack.add(node)
        path.append(node)
        for neigh in graph.get(node, set()):
            if neigh not in visited:
                dfs(neigh, path)
            elif neigh in stack:
                # found a cycle
                try:
                    i = path.index(neigh)
                    cycles.append(path[i:] + [neigh])
                except ValueError:
                    cycles.append([neigh, node, neigh])
        stack.remove(node)
        path.pop()

    for n in list(graph.keys()):
        if n not in visited:
            dfs(n, [])
    return cycles


def test_dependency_graph_cycles_and_unused_modules():
    src_root = Path(__file__).resolve().parent.parent / "src"
    if not src_root.exists():
        return

    graph = _collect_import_graph(src_root)
    cycles = _find_cycles(graph)
    for cyc in cycles[:50]:
        issue_logger.log(
            category="dependency",
            severity="warning",
            message="Circular import detected: " + " -> ".join(cyc),
        )

    # Modules that nothing imports (potentially unused)
    imported_anywhere = set()
    for deps in graph.values():
        imported_anywhere |= deps
    defined = set(graph.keys())
    unused_mods = sorted(defined - imported_anywhere)
    for mod in unused_mods[:200]:
        issue_logger.log(
            category="dependency",
            severity="info",
            message=f"Module not imported elsewhere: {mod}",
        )

    assert True

