! Assumed-shape pointer targets with declared lower bounds (``qa(0:, -2:)``): the gather
! temp's extents come from the descriptor (``fir.box_dims``) and its lower bounds from the
! declares' ``fir.shift``.  ``q(ix)%x`` is selected at runtime, so the gather path is taken,
! and the loop bounds query the component whole (``LBOUND`` / ``UBOUND`` / ``SIZE``), which
! reads the gather temp's slice rebased to the component's lower bounds.
MODULE m
  IMPLICIT NONE
  TYPE t_ptr_2d
    REAL(8), POINTER :: x(:,:)
  END TYPE
CONTAINS
  SUBROUTINE run(qa, qb, ix, out)
    REAL(8), TARGET, INTENT(INOUT) :: qa(0:, -2:), qb(0:, -2:)
    INTEGER, INTENT(IN) :: ix
    REAL(8), INTENT(OUT) :: out(0:, -2:)
    TYPE(t_ptr_2d) :: q(2)
    INTEGER :: i, j
    q(1)%x => qa
    q(2)%x => qb
    DO j = LBOUND(q(ix)%x, 2), UBOUND(q(ix)%x, 2)
      DO i = LBOUND(q(ix)%x, 1), LBOUND(q(ix)%x, 1) + SIZE(q(ix)%x, 1) - 1
        out(i, j) = q(ix)%x(i, j)
      END DO
    END DO
  END SUBROUTINE run
END MODULE m
