! Pointer targets whose lower bounds are runtime values (``qa(m:, m:)``).  Both targets
! take theirs from the same ``intent(in)`` dummy, so the components provably agree on
! them and share one gather temp starting at ``m``.
MODULE m
  IMPLICIT NONE
  TYPE t_ptr_2d
    REAL(8), POINTER :: x(:,:)
  END TYPE
CONTAINS
  SUBROUTINE run(m, ix, qa, qb, out)
    INTEGER, INTENT(IN) :: m, ix
    REAL(8), TARGET, INTENT(INOUT) :: qa(m:, m:), qb(m:, m:)
    REAL(8), INTENT(OUT) :: out(:, :)
    TYPE(t_ptr_2d) :: q(2)
    INTEGER :: i, j
    q(1)%x => qa
    q(2)%x => qb
    DO j = 1, SIZE(out, 2)
      DO i = 1, SIZE(out, 1)
        out(i, j) = q(ix)%x(m + i - 1, m + j - 1)
      END DO
    END DO
  END SUBROUTINE run
END MODULE m
