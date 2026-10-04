! MAXLOC used as an INLINE expression operand (not the direct RHS of an
! assignment).  ``hlfir-lift-reduction-operands`` must lift the nested
! ``hlfir.maxloc`` into a scalar temp so the consuming ``+ 1`` sees a
! scalar load; otherwise ``buildExpr`` strands ``?`` and the tasklet body
! ``(_ + 1)`` fails to parse.  WRF's Grell-Freitas ``k22(i) = MAXLOC(...)``
! is the motivating shape.
SUBROUTINE maxloc_expr_operand(n, arr, idx)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: arr(n)
  INTEGER, INTENT(OUT) :: idx
  idx = MAXLOC(arr, 1) + 1
END SUBROUTINE maxloc_expr_operand
