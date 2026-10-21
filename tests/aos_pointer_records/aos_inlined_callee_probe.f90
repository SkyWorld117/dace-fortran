! An array of pointer records local to ``run`` is rebound there and read in an inlined
! callee through the callee's own dummy, a re-declare of the same storage: lifting
! ``run``'s array redirects those reads too and sweeps the re-declare.  In ``kernel``
! itself the array is a dummy argument, whose pointers its caller associated, so it is
! not lifted.
MODULE m
  IMPLICIT NONE
  TYPE t_ptr_2d
    REAL(8), POINTER :: x(:,:)
  END TYPE
CONTAINS
  SUBROUTINE run(n, k, qa, qb, out)
    INTEGER, INTENT(IN) :: n, k
    REAL(8), TARGET, INTENT(INOUT) :: qa(n, k), qb(n, k)
    REAL(8), INTENT(OUT) :: out(n, k)
    TYPE(t_ptr_2d) :: q(2)
    q(1)%x => qa
    q(2)%x => qb
    CALL kernel(n, k, q, out)
  END SUBROUTINE run

  SUBROUTINE kernel(n, k, q, out)
    INTEGER, INTENT(IN) :: n, k
    TYPE(t_ptr_2d), INTENT(IN) :: q(2)
    REAL(8), INTENT(OUT) :: out(n, k)
    INTEGER :: i, j
    DO j = 1, k
      DO i = 1, n
        out(i, j) = q(1)%x(i, j) - q(2)%x(i, j)
      END DO
    END DO
  END SUBROUTINE kernel
END MODULE m
