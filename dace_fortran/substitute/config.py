"""Options of the substitute tools from a TOML file, so that an application keeps its settings in one place.

The file holds one table per tool, named like the tool's module (``[make_substitute]``); a key is an option's name with
``_`` for ``-`` (``tile_fuse = true`` for ``--tile-fuse``), a list for a repeatable option.  A sub-table named like a
flag applies when the command line gives that flag (``[make_substitute.gpu]`` for ``--gpu``).  The file gives defaults:
the command line still overrides them, and repeatable options add to the file's list.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path


def parse_with_config(ap: argparse.ArgumentParser, tool: str, argv: list[str] | None = None) -> argparse.Namespace:
    """Parse ``argv`` with ``ap`` after taking the defaults of its ``--config`` file's ``[tool]`` table (``--config``
    is added to ``ap`` here).  A key that names none of ``ap``'s options is an error."""
    ap.add_argument("--config", type=Path, help=f"TOML file whose [{tool}] table sets defaults of these options")
    known, _ = ap.parse_known_args(argv)
    if known.config is not None:
        table = dict(tomllib.loads(known.config.read_text()).get(tool, {}))
        for flag in [k for k, v in table.items() if isinstance(v, dict)]:
            sub = table.pop(flag)
            if getattr(known, flag, False):
                table.update(sub)
        unknown = sorted(set(table) - {a.dest for a in ap._actions})
        if unknown:
            ap.error(f"{known.config} [{tool}]: no option {', '.join(unknown)}")
        ap.set_defaults(**table)
    return ap.parse_args(argv)
