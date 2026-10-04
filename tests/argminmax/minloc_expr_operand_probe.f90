! MINLOC used as an INLINE expression operand -- same lift as the MAXLOC
! probe, but for the ArgMin node.
SUBROUTINE minloc_expr_operand(n, arr, idx)
  IMPLICIT NONE
  INTEGER, INTENT(IN) :: n
  REAL(8), INTENT(IN) :: arr(n)
  INTEGER, INTENT(OUT) :: idx
  idx = MINLOC(arr, 1) - 1
END SUBROUTINE minloc_expr_operand
