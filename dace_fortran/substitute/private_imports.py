"""Module entities a generated binding imports that their module keeps private.

Usage: private_imports.py <bindings.f90> <MFC src dir>

Reads the binding's ``use <module>, only: ...`` lists, finds each module's source (``module <name>`` in a .fpp/.f90
under the source dir), and reports the imported names that module does not make public: a module whose
specification part has a bare ``private`` statement exports only what its ``public`` statements and the
declarations with the ``public`` attribute name.
"""

import re
import sys
from pathlib import Path


def joined_lines(text: str) -> list[str]:
    """Free-form source lines with ``&`` continuations joined and comments dropped."""
    out, cur = [], ""
    for raw in text.splitlines():
        line = raw.split("!")[0].rstrip() if not raw.lstrip().startswith("!$") else ""
        stripped = line.strip()
        if stripped.startswith("&"):
            stripped = stripped[1:].lstrip()
        if stripped.endswith("&"):
            cur += stripped[:-1] + " "
            continue
        out.append(cur + stripped)
        cur = ""
    return out


def imports(binding: Path) -> dict[str, set[str]]:
    used: dict[str, set[str]] = {}
    for line in joined_lines(binding.read_text()):
        m = re.match(r"use\s+(\w+)\s*,\s*only\s*:\s*(.*)", line, re.I)
        if not m:
            continue
        for item in m.group(2).split(","):
            name = item.split("=>")[-1].strip().lower()
            if name:
                used.setdefault(m.group(1).lower(), set()).add(name)
    return used


def module_spec(src: Path, module: str) -> list[str] | None:
    """Specification-part lines of ``module`` (up to ``contains``), or None when no source defines it."""
    for path in [*src.rglob("*.fpp"), *src.rglob("*.f90")]:
        lines = joined_lines(path.read_text(errors="replace"))
        for i, line in enumerate(lines):
            if re.fullmatch(rf"module\s+{module}", line.strip(), re.I):
                spec = []
                for nxt in lines[i + 1 :]:
                    if re.match(r"contains\b", nxt.strip(), re.I):
                        break
                    spec.append(nxt.strip())
                return spec
    return None


def main() -> int:
    binding, src = Path(sys.argv[1]), Path(sys.argv[2])
    for module, names in sorted(imports(binding).items()):
        spec = module_spec(src, module)
        if spec is None:
            continue
        statements = [s for part in spec for s in part.split(";")]
        if not any(re.fullmatch(r"private", s.strip(), re.I) for s in statements):
            continue
        public = set()
        for s in statements:
            m = re.match(r"public\s*(?:::)?\s*(.*)", s.strip(), re.I)
            if m:
                public |= {n.strip().lower() for n in m.group(1).split(",") if n.strip()}
            # A declaration with the public attribute: ``real, ..., public :: a, b``.
            attributes, sep, entities = s.partition("::")
            if sep and re.search(r",\s*public\b", attributes, re.I):
                public |= {re.split(r"[\s(=]", n.strip())[0].lower() for n in entities.split(",") if n.strip()}
        hidden = sorted(names - public)
        if hidden:
            print(f"{module}: {', '.join(hidden)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
