! How a rebound pointer component takes its lower bounds: from the box it is assigned.
! ``section`` restarts them at 1 (``=> qa(:, :)``), ``remap`` sets them
! (``q(c)%x(5:, 7:) => qa``) and loops over them with ``LBOUND`` / ``UBOUND``, and
! ``remap_runtime`` sets them to a runtime value.  ``nested`` rebinds inside an ``IF``,
! ``local_target`` points at a static local, ``window`` accesses the target directly
! while the component points at it, and ``callee_window`` does so through an inlined
! callee's explicit-shape dummy (with bounds 1: the component's).  Most entries also write through a component, so the
! copy back to a target whose bounds differ from the component's is covered too.
MODULE m
  IMPLICIT NONE
  TYPE t_ptr_2d
    REAL(8), POINTER :: x(:,:)
  END TYPE
CONTAINS
  SUBROUTINE section(n, k, qa, qb, out)
    INTEGER, INTENT(IN) :: n, k
    REAL(8), TARGET, INTENT(INOUT) :: qa(0:n, -2:k), qb(0:n, -2:k)
    REAL(8), INTENT(OUT) :: out(0:n, -2:k)
    TYPE(t_ptr_2d) :: q(2)
    INTEGER :: i, j
    q(1)%x => qa(:, :)
    q(2)%x => qb(:, :)
    DO j = -2, k
      DO i = 0, n
        out(i, j) = q(1)%x(i + 1, j + 3) + 2.0D0*q(2)%x(i + 1, j + 3)
      END DO
    END DO
    q(2)%x(1, 1) = q(1)%x(n + 1, k + 3)
  END SUBROUTINE section

  SUBROUTINE remap(n, k, qa, qb, out)
    INTEGER, INTENT(IN) :: n, k
    REAL(8), TARGET, INTENT(INOUT) :: qa(0:n, -2:k), qb(0:n, -2:k)
    REAL(8), INTENT(OUT) :: out(0:n, -2:k)
    TYPE(t_ptr_2d) :: q(2)
    INTEGER :: i, j
    q(1)%x(5:, 7:) => qa
    q(2)%x(5:, 7:) => qb
    DO j = LBOUND(q(1)%x, 2), UBOUND(q(1)%x, 2)
      DO i = LBOUND(q(1)%x, 1), UBOUND(q(1)%x, 1)
        out(i - 5, j - 9) = q(1)%x(i, j) + 2.0D0*q(2)%x(i, j)
      END DO
    END DO
    q(2)%x(5, 7) = q(1)%x(n + 5, k + 9)
  END SUBROUTINE remap

  SUBROUTINE remap_runtime(n, k, m, qa, qb, out)
    INTEGER, INTENT(IN) :: n, k, m
    REAL(8), TARGET, INTENT(INOUT) :: qa(0:n, -2:k), qb(0:n, -2:k)
    REAL(8), INTENT(OUT) :: out(0:n, -2:k)
    TYPE(t_ptr_2d) :: q(2)
    INTEGER :: i, j
    q(1)%x(m:, m:) => qa
    q(2)%x(m:, m:) => qb
    DO j = -2, k
      DO i = 0, n
        out(i, j) = q(1)%x(i + m, j + 2 + m) + 2.0D0*q(2)%x(i + m, j + 2 + m)
      END DO
    END DO
    q(2)%x(m, m) = q(1)%x(n + m, k + 2 + m)
  END SUBROUTINE remap_runtime

  SUBROUTINE nested(n, k, flag, qa, qb, out)
    INTEGER, INTENT(IN) :: n, k, flag
    REAL(8), TARGET, INTENT(INOUT) :: qa(0:n, -2:k), qb(0:n, -2:k)
    REAL(8), INTENT(INOUT) :: out(0:n, -2:k)
    TYPE(t_ptr_2d) :: q(2)
    INTEGER :: i, j
    IF (flag > 0) THEN
      q(1)%x => qa
      q(2)%x => qb
      DO j = -2, k
        DO i = 0, n
          out(i, j) = q(1)%x(i, j) + 2.0D0*q(2)%x(i, j)
        END DO
      END DO
      q(2)%x(0, -2) = q(1)%x(n, k)
    END IF
  END SUBROUTINE nested

  SUBROUTINE local_target(out)
    REAL(8), INTENT(OUT) :: out(0:3, -2:2)
    REAL(8), TARGET :: loc(0:3, -2:2)
    TYPE(t_ptr_2d) :: q(1)
    INTEGER :: i, j
    DO j = -2, 2
      DO i = 0, 3
        loc(i, j) = 10.0D0*i + j
      END DO
    END DO
    q(1)%x => loc
    DO j = -2, 2
      DO i = 0, 3
        out(i, j) = q(1)%x(i, j)
      END DO
    END DO
  END SUBROUTINE local_target

  SUBROUTINE window(n, k, qv, u, t)
    INTEGER, INTENT(IN) :: n, k
    REAL(8), TARGET, INTENT(INOUT) :: qv(0:n, -2:k)
    REAL(8), INTENT(IN) :: u(0:n, -2:k)
    REAL(8), INTENT(OUT) :: t(0:n, -2:k)
    TYPE(t_ptr_2d) :: q(1)
    INTEGER :: i, j
    q(1)%x => qv
    DO j = -2, k
      DO i = 0, n
        q(1)%x(i, j) = q(1)%x(i, j) + 1.0D0
        t(i, j) = qv(i, j)*2.0D0
      END DO
    END DO
    qv = u
    DO j = -2, k
      DO i = 0, n
        t(i, j) = t(i, j) + q(1)%x(i, j)
      END DO
    END DO
  END SUBROUTINE window

  SUBROUTINE callee_window(n, k, qv, t)
    INTEGER, INTENT(IN) :: n, k
    REAL(8), TARGET, INTENT(INOUT) :: qv(n, k)
    REAL(8), INTENT(OUT) :: t(n, k)
    TYPE(t_ptr_2d) :: q(1)
    INTEGER :: i, j
    q(1)%x => qv
    DO j = 1, k
      DO i = 1, n
        q(1)%x(i, j) = q(1)%x(i, j) + 1.0D0
      END DO
    END DO
    CALL double_into(n, k, qv, t)
  END SUBROUTINE callee_window

  SUBROUTINE double_into(n, k, a, b)
    INTEGER, INTENT(IN) :: n, k
    REAL(8), INTENT(IN) :: a(n, k)
    REAL(8), INTENT(OUT) :: b(n, k)
    INTEGER :: i, j
    DO j = 1, k
      DO i = 1, n
        b(i, j) = 2.0D0*a(i, j)
      END DO
    END DO
  END SUBROUTINE double_into
END MODULE m
