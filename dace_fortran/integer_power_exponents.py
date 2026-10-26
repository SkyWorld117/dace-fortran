# Copyright 2025-2026 ETH Zurich and the dace-fortran authors. All rights reserved.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Retype integer-valued float ``**`` exponents to ``int`` so codegen emits
``ipow`` (repeated multiply) instead of libm ``pow``, which isn't bit-identical
to Fortran/C's integer-power codegen and drifts across real(8) reduction chains.
Only the exponent literal is rewritten; base and fractional exponents are untouched.
"""

from __future__ import annotations

import ast

import dace
from dace.transformation import pass_pipeline as ppl

from dace_fortran.dace_types import explicit_cf_compatible


class _ExponentIntegerizer(ast.NodeTransformer):
    """Rewrite integer-valued float ``**`` exponents to ``int`` literals."""

    def __init__(self) -> None:
        self.rewrites = 0

    @staticmethod
    def _as_int_constant(node: ast.expr) -> ast.expr | None:
        """Retype an integral float exponent, honoring explicit dtype rounding.

        The builder wraps literals in qualified or unqualified DaCe dtype calls.
        Only unwrap a typed integer within the dtype's consecutive-integer
        range, so e.g. ``float32(16777217.0)`` cannot lose its rounding step.
        """
        original = node
        sign = 1
        while isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            if isinstance(node.op, ast.USub):
                sign = -sign
            node = node.operand
        limit = None
        if isinstance(node, ast.Call):
            func = node.func
            limits = {'float16': 2**11, 'float32': 2**24, 'float64': 2**53}
            # Tasklet dtype constructors also occur as bare float32(...).
            # They are reserved DaCe tasklet globals, like the qualified form.
            cast = func.id if isinstance(func, ast.Name) else (
                func.attr if isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name) and func.value.id == 'dace' else None)
            if cast not in limits or len(node.args) != 1 or node.keywords:
                return None
            # dace::math::ipow takes an unsigned 32-bit magnitude.
            limit = min(limits[cast], 2**32-1)
            node = node.args[0]
            while isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                if isinstance(node.op, ast.USub):
                    sign = -sign
                node = node.operand
        if not isinstance(node, ast.Constant) or isinstance(node.value, bool):
            return None
        value = node.value
        if isinstance(value, float):
            if not value.is_integer():
                return None
        elif not (limit is not None and isinstance(value, int)):
            return None
        if limit is not None and abs(value) > limit:
            return None
        return ast.copy_location(ast.Constant(value=sign * int(value)), original)

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        """Retype an integer-valued float ``**`` exponent to ``int``."""
        self.generic_visit(node)
        if isinstance(node.op, ast.Pow):
            repl = self._as_int_constant(node.right)
            if repl is not None:
                node.right = repl
                self.rewrites += 1
        return node


@explicit_cf_compatible
class IntegerizePowerExponents(ppl.Pass):
    """Retype integer-valued float ``**`` exponents in tasklets to ``int``.

    Must run after tasklet splitting -- flips codegen from libm ``pow`` to
    ``ipow`` without touching the base expression or any connector.
    """

    def modifies(self) -> ppl.Modifies:
        """This pass only mutates tasklet bodies."""
        return ppl.Modifies.Tasklets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        """One-shot: the retype doesn't re-trigger the pass."""
        return False

    def apply_pass(self, sdfg: dace.SDFG, pipeline_results: dict) -> int | None:
        """Rewrite every Python tasklet's integer-valued float ``**`` exponents
        to ``int``, recursively including nested SDFGs. Returns rewrite count, or ``None`` if unchanged."""
        total = 0
        for node, _parent in sdfg.all_nodes_recursive():
            if not isinstance(node, dace.nodes.Tasklet):
                continue
            if node.code.language != dace.dtypes.Language.Python:
                continue
            body = node.code.code
            if not isinstance(body, list):
                continue
            tr = _ExponentIntegerizer()
            for stmt in body:
                tr.visit(stmt)
            if tr.rewrites:
                for stmt in body:
                    ast.fix_missing_locations(stmt)
                total += tr.rewrites
        return total or None
