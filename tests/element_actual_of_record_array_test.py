"""An array dummy of records bound to ONE ELEMENT of a module record array (sequence association, MFC's surface
tension passes ``q_prim_qp%vf(c_idx)`` to a routine whose dummy is ``v_vf(n)``): after inlining, the dummy's own
subscript composes with the element's (``vf(c + i - 1)``) and the image keeps its four dimensions. This simple
form passes; 2D_laplace_pressure_jump's deeper chain (``offset_q_prim_qp_vf_sf_d4`` unresolved) is still open.
Bridge-side: needs LLVM."""
from dace_fortran.bindings.frozen_signature import get_frozen_signature
from tests._util import build_sdfg

_SRC = """
MODULE m_recs
  IMPLICIT NONE
  TYPE scalar_field
    REAL(8), POINTER :: sf(:, :, :) => NULL()
  END TYPE
  TYPE vector_field
    TYPE(scalar_field), ALLOCATABLE :: vf(:)
  END TYPE
  TYPE(vector_field) :: q
CONTAINS
  SUBROUTINE pack_one(v, n, out, m)
    INTEGER, INTENT(IN) :: n, m
    TYPE(scalar_field), INTENT(IN) :: v(n)
    REAL(8), INTENT(OUT) :: out(0:m, 0:m, 0:m)
    INTEGER :: i, j, k, l
    DO i = 1, n
      DO l = 0, m
        DO k = 0, m
          DO j = 0, m
            out(j, k, l) = out(j, k, l) + i * v(i)%sf(j, k, l)
          END DO
        END DO
      END DO
    END DO
  END SUBROUTINE pack_one
  SUBROUTINE run(out, m, c)
    INTEGER, INTENT(IN) :: m, c
    REAL(8), INTENT(OUT) :: out(0:m, 0:m, 0:m)
    out = 0.0D0
    CALL pack_one(q%vf(c), 1, out, m)
  END SUBROUTINE run
END MODULE m_recs
"""


def test_element_actual_composes_with_the_dummy_subscript(tmp_path):
    sdfg = build_sdfg(_SRC, tmp_path / "sdfg", name="run", entry="m_recs::run").build()
    frozen = get_frozen_signature(sdfg)
    image = next(a for a in frozen.args if a.sdfg_name == "q_vf_sf")
    assert image.rank == 4
    assert not any(s.endswith("q_vf_sf_d4") for s in frozen.free_symbols)
