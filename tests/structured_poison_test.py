"""Structurizer padding is representable without accepting real poison reads."""
import pytest

from dace_fortran.build_bridge import hb


def assignments(nodes):
    for node in nodes:
        if node.kind == 'assign':
            yield node
        yield from assignments(node.children)
        yield from assignments(node.else_children)


@pytest.mark.parametrize('dtype,expected', [('i32', '0'), ('index', '0'), ('f64', '?')])
def test_poison_yield_padding_is_only_lowered_for_integer_results(tmp_path, dtype, expected):
    path = tmp_path / 'padding.mlir'
    path.write_text(f"""module {{
  func.func @_QPpadding() {{
    %flag = arith.constant true
    %padding = ub.poison : {dtype}
    scf.while : () -> () {{
    %result = scf.if %flag -> ({dtype}) {{
      scf.yield %padding : {dtype}
    }} else {{
      scf.yield %padding : {dtype}
    }}
      scf.condition(%flag)
    }} do {{
      scf.yield
    }}
    return
  }}
}}
""")
    module = hb.HLFIRModule()
    assert module.parse_file(str(path))
    module.set_entry_symbol('_QPpadding')
    module.get_variables()
    values = [node.expr for node in assignments(module.get_ast()) if node.target.startswith('__sc_')]
    assert values and all(value == expected for value in values)
