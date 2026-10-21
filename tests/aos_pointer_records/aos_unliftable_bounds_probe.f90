! Rebinds the gather cannot represent, which the pass has to leave alone: a component
! pointing at part of its target (``part``), components whose targets start at different
! lower bounds (``mixed``), an array of records with its own lower bound (``offset_aos``),
! and a remap whose runtime lower bound is computed after another component was rebound,
! so no gather temp can be created ahead of every rebind (``late_bound``).
MODULE m
  IMPLICIT NONE
  TYPE t_ptr_2d
    REAL(8), POINTER :: x(:,:)
  END TYPE
  TYPE t_two
    REAL(8), POINTER :: p(:), x(:,:)
  END TYPE
CONTAINS
  SUBROUTINE part(n, k, qa, out)
    INTEGER, INTENT(IN) :: n, k
    REAL(8), TARGET, INTENT(INOUT) :: qa(0:n, -2:k)
    REAL(8), INTENT(OUT) :: out(n, k + 3)
    TYPE(t_ptr_2d) :: q(1)
    INTEGER :: i, j
    q(1)%x => qa(1:n, :)
    DO j = 1, k + 3
      DO i = 1, n
        out(i, j) = q(1)%x(i, j)
      END DO
    END DO
  END SUBROUTINE part

  SUBROUTINE mixed(n, k, qa, qb, out)
    INTEGER, INTENT(IN) :: n, k
    REAL(8), TARGET, INTENT(INOUT) :: qa(0:n, -2:k), qb(n + 1, k + 3)
    REAL(8), INTENT(OUT) :: out(n + 1, k + 3)
    TYPE(t_ptr_2d) :: q(2)
    INTEGER :: i, j
    q(1)%x => qa
    q(2)%x => qb
    DO j = 1, k + 3
      DO i = 1, n + 1
        out(i, j) = q(1)%x(i - 1, j - 3) + q(2)%x(i, j)
      END DO
    END DO
  END SUBROUTINE mixed

  SUBROUTINE offset_aos(n, k, qa, qb, out)
    INTEGER, INTENT(IN) :: n, k
    REAL(8), TARGET, INTENT(INOUT) :: qa(n, k), qb(n, k)
    REAL(8), INTENT(OUT) :: out(n, k)
    TYPE(t_ptr_2d) :: q(0:1)
    INTEGER :: i, j
    q(0)%x => qa
    q(1)%x => qb
    DO j = 1, k
      DO i = 1, n
        out(i, j) = q(0)%x(i, j) + q(1)%x(i, j)
      END DO
    END DO
  END SUBROUTINE offset_aos

  SUBROUTINE late_bound(n, k, m, pa, qa, out)
    INTEGER, INTENT(IN) :: n, k, m
    REAL(8), TARGET, INTENT(INOUT) :: pa(n), qa(n, k)
    REAL(8), INTENT(OUT) :: out(n, k)
    TYPE(t_two) :: q(1)
    INTEGER :: i, j
    q(1)%p => pa
    q(1)%x(m:, m:) => qa
    DO j = 1, k
      DO i = 1, n
        out(i, j) = q(1)%p(i) + q(1)%x(m + i - 1, m + j - 1)
      END DO
    END DO
  END SUBROUTINE late_bound
END MODULE m
