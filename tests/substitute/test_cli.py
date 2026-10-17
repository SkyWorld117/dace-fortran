"""The command lines of the tools parse (no module of theirs is imported: they need the frontend's bridge)."""
import argparse
import ast
from pathlib import Path


def _parser_of(path: Path) -> argparse.ArgumentParser:
    """Rebuild the ArgumentParser a tool's main() makes from its add_argument calls (the statements naming ``ap``,
    the group statements included), with the module's constants they use."""
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    constants = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                 if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                 and isinstance(n.value, ast.Constant | ast.BinOp | ast.List)
                 and n.targets[0].id.isupper()
                 and _literal(n.value)}
    lines = []
    for stmt in main.body:
        text = ast.unparse(stmt)
        if text.startswith(("ap = ", "ap.add_argument", "mode = ", "mode.add_argument")):
            lines.append(text)
    namespace = {"argparse": argparse, "Path": Path, **constants}
    exec("\n".join(lines), namespace)  # noqa: S102
    return namespace["ap"]


def _literal(node) -> bool:
    try:
        ast.literal_eval(node)
        return True
    except ValueError:
        return False


def test_make_substitute_parses_its_flags():
    ap = _parser_of(Path(__file__).resolve().parents[2] / "dace_fortran" / "substitute" / "make_substitute.py")
    args = ap.parse_args(["--gpu", "--fma", "x.hlfir", "m::r", "out"])
    assert args.fma
    assert not ap.parse_args(["x.hlfir", "m::r", "out"]).fma  # no FMA by default
    assert not ap.parse_args(["--no-fma", "x.hlfir", "m::r", "out"]).fma
    assert ap.parse_args(["x.hlfir", "m::r", "out"]).scalarise
