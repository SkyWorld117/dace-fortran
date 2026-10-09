! Pointer targets with lower bounds other than 1: ``qa(0:n, -2:k)``.  A pointer member
! takes the bounds of its target, so ``q(c)%x(i, j)`` subscripts relative to (0, -2) and
! the gather temp has to keep those bounds.  With 1-based inner dims every access lands
! one row and three columns off (negative out-of-bounds memlets).  The last statement
! writes through the alias at the targets' first element, so the copy-out is covered too.
MODULE m
  IMPLICIT NONE
  TYPE t_ptr_2d
    REAL(8), POINTER :: x(:,:)
  END TYPE
CONTAINS
  SUBROUTINE run(n, k, qa, qb, out)
    INTEGER, INTENT(IN) :: n, k
    REAL(8), TARGET, INTENT(INOUT) :: qa(0:n, -2:k), qb(0:n, -2:k)
    REAL(8), INTENT(OUT) :: out(0:n, -2:k)
    TYPE(t_ptr_2d) :: q(2)
    INTEGER :: i, j
    q(1)%x => qa
    q(2)%x => qb
    DO j = -2, k
      DO i = 0, n
        out(i, j) = q(1)%x(i, j) + 2.0D0*q(2)%x(i, j)
      END DO
    END DO
    q(2)%x(0, -2) = q(1)%x(n, k)
  END SUBROUTINE run
END MODULE m
