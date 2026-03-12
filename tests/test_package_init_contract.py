"""Regression tests for explicit package exports and import-string targets."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    REPO_ROOT / "src",
    REPO_ROOT / "lumen-app" / "src",
    *sorted((REPO_ROOT / "packages").glob("*/src")),
)
CONFIG_SOURCE = REPO_ROOT / "lumen-app" / "src" / "lumen_app" / "services" / "config.py"


def _should_skip_path(path: Path) -> bool:
    return any(
        part == "__pycache__" or part.endswith(".egg-info") or part.startswith(".")
        for part in path.parts
    )


def _iter_package_dirs() -> list[Path]:
    package_dirs: set[Path] = set()

    for source_root in SOURCE_ROOTS:
        for py_file in source_root.rglob("*.py"):
            if _should_skip_path(py_file):
                continue

            current = source_root
            for part in py_file.relative_to(source_root).parts[:-1]:
                current /= part
                package_dirs.add(current)

    return sorted(package_dirs)


def _extract_dotted_paths(keyword: str) -> list[str]:
    tree = ast.parse(CONFIG_SOURCE.read_text(encoding="utf-8"))
    values: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        for item in node.keywords:
            if (
                item.arg == keyword
                and isinstance(item.value, ast.Constant)
                and isinstance(item.value.value, str)
            ):
                values.add(item.value.value)

    return sorted(values)


def _resolve_module_source(module_path: str) -> Path | None:
    parts = module_path.split(".")

    for source_root in SOURCE_ROOTS:
        package_init = source_root.joinpath(*parts, "__init__.py")
        if package_init.parent.is_dir():
            return package_init

        module_file = source_root.joinpath(*parts).with_suffix(".py")
        if module_file.is_file():
            return module_file

    return None


def _extract_top_level_names(source_path: Path) -> set[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names: set[str] = set()

    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
            continue

        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[-1])
            continue

        if isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(_extract_assigned_names(target))
            continue

        if isinstance(node, ast.AnnAssign):
            names.update(_extract_assigned_names(node.target))

    return names


def _extract_assigned_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}

    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in target.elts:
            names.update(_extract_assigned_names(element))
        return names

    return set()


def test_source_package_dirs_define_init_modules() -> None:
    missing = [
        str(path.relative_to(REPO_ROOT))
        for path in _iter_package_dirs()
        if not (path / "__init__.py").is_file()
    ]

    assert not missing, "Missing __init__.py for package directories:\n" + "\n".join(
        missing
    )


def test_import_info_paths_resolve_to_symbols() -> None:
    missing_symbols: list[str] = []

    for keyword in ("registry_class", "add_to_server"):
        for dotted_path in _extract_dotted_paths(keyword):
            module_path, symbol_name = dotted_path.rsplit(".", 1)
            source_path = _resolve_module_source(module_path)

            if source_path is None or not source_path.is_file():
                missing_symbols.append(
                    f"{dotted_path} (missing module or package __init__.py)"
                )
                continue

            exported_names = _extract_top_level_names(source_path)
            if symbol_name not in exported_names:
                missing_symbols.append(
                    f"{dotted_path} (symbol not exported from {source_path.relative_to(REPO_ROOT)})"
                )

    assert not missing_symbols, "Broken import-string targets:\n" + "\n".join(
        missing_symbols
    )
