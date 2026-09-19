! Fixture: every collapse(3) nest of the routine this discovery was built against.
!
! VERBATIM from the preprocessed source, so the parser is exercised by real bodies rather than
! by a synthetic approximation of them -- a fixture containing only the shapes the parser
! already handles cannot fail.  The array names are the source's; nothing here is compiled,
! the fixture is parsed as text.

subroutine shapes
!$acc parallel loop collapse(3)
# 408 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 408 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 408 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 408 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 408 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp target teams loop defaultmap(firstprivate:scalar) bind(teams,parallel) collapse(3) defaultmap(tofrom:aggregate) defaultmap(tofrom:allocatable) defaultmap(tofrom:pointer)
# 408 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
            do l = is3_viscous%beg, is3_viscous%end
                do k = iy%beg, iy%end
                    do j = is1_viscous%beg + 1, is1_viscous%end

# 412 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#if defined(MFC_OpenACC)
# 412 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$acc loop seq 
# 412 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 412 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp loop bind(thread)
# 412 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                        do i = iv%beg, iv%end
                            dqL_prim_dx_n(1)%vf(i)%sf(j, k, l) = (q_prim_qp%vf(i)%sf(j, k, l) - q_prim_qp%vf(i)%sf(j - 1, k, &
                                          & l))/(x_cc(j) - x_cc(j - 1))
                        end do
                    end do
                end do
            end do
!$acc parallel loop collapse(3)
# 422 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 422 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 422 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 422 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 422 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp target teams loop defaultmap(firstprivate:scalar) bind(teams,parallel) collapse(3) defaultmap(tofrom:aggregate) defaultmap(tofrom:allocatable) defaultmap(tofrom:pointer)
# 422 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
            do l = is3_viscous%beg, is3_viscous%end
                do k = is2_viscous%beg, is2_viscous%end
                    do j = is1_viscous%beg, is1_viscous%end - 1

# 426 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#if defined(MFC_OpenACC)
# 426 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$acc loop seq 
# 426 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 426 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp loop bind(thread)
# 426 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                        do i = iv%beg, iv%end
                            dqR_prim_dx_n(1)%vf(i)%sf(j, k, l) = (q_prim_qp%vf(i)%sf(j + 1, k, l) - q_prim_qp%vf(i)%sf(j, k, &
                                          & l))/(x_cc(j + 1) - x_cc(j))
                        end do
                    end do
                end do
            end do
!$acc parallel loop collapse(3)
# 456 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 456 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 456 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 456 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 456 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp target teams loop defaultmap(firstprivate:scalar) bind(teams,parallel) collapse(3) defaultmap(tofrom:aggregate) defaultmap(tofrom:allocatable) defaultmap(tofrom:pointer)
# 456 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                    do l = is3_viscous%beg, is3_viscous%end
                        do j = is2_viscous%beg + 1, is2_viscous%end
                            do k = is1_viscous%beg, is1_viscous%end

# 460 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#if defined(MFC_OpenACC)
# 460 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$acc loop seq 
# 460 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 460 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp loop bind(thread)
# 460 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                                do i = iv%beg, iv%end
                                    dqL_prim_dy_n(2)%vf(i)%sf(k, j, l) = (q_prim_qp%vf(i)%sf(k, j, l) - q_prim_qp%vf(i)%sf(k, &
                                                  & j - 1, l))/(y_cc(j) - y_cc(j - 1))
                                end do
                            end do
                        end do
                    end do
!$acc parallel loop collapse(3)
# 470 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 470 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 470 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 470 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 470 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp target teams loop defaultmap(firstprivate:scalar) bind(teams,parallel) collapse(3) defaultmap(tofrom:aggregate) defaultmap(tofrom:allocatable) defaultmap(tofrom:pointer)
# 470 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                    do l = is3_viscous%beg, is3_viscous%end
                        do j = is2_viscous%beg, is2_viscous%end - 1
                            do k = is1_viscous%beg, is1_viscous%end

# 474 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#if defined(MFC_OpenACC)
# 474 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$acc loop seq 
# 474 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 474 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp loop bind(thread)
# 474 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                                do i = iv%beg, iv%end
                                    dqR_prim_dy_n(2)%vf(i)%sf(k, j, l) = (q_prim_qp%vf(i)%sf(k, j + 1, l) - q_prim_qp%vf(i)%sf(k, &
                                                  & j, l))/(y_cc(j + 1) - y_cc(j))
                                end do
                            end do
                        end do
                    end do
!$acc parallel loop collapse(3)
# 510 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 510 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 510 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 510 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 510 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp target teams loop defaultmap(firstprivate:scalar) bind(teams,parallel) collapse(3) defaultmap(tofrom:aggregate) defaultmap(tofrom:allocatable) defaultmap(tofrom:pointer)
# 510 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                    do l = is3_viscous%beg, is3_viscous%end
                        do j = is2_viscous%beg + 1, is2_viscous%end
                            do k = is1_viscous%beg + 1, is1_viscous%end - 1

# 514 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#if defined(MFC_OpenACC)
# 514 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$acc loop seq 
# 514 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 514 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp loop bind(thread)
# 514 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                                do i = iv%beg, iv%end
                                    dqL_prim_dx_n(2)%vf(i)%sf(k, j, l) = (dqL_prim_dx_n(1)%vf(i)%sf(k, j, &
                                                  & l) + dqR_prim_dx_n(1)%vf(i)%sf(k, j, l) + dqL_prim_dx_n(1)%vf(i)%sf(k, j - 1, &
                                                  & l) + dqR_prim_dx_n(1)%vf(i)%sf(k, j - 1, l))

                                    dqL_prim_dx_n(2)%vf(i)%sf(k, j, l) = 25.e-2_wp*dqL_prim_dx_n(2)%vf(i)%sf(k, j, l)
                                end do
                            end do
                        end do
                    end do
!$acc parallel loop collapse(3)
# 527 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 527 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 527 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 527 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 527 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp target teams loop defaultmap(firstprivate:scalar) bind(teams,parallel) collapse(3) defaultmap(tofrom:aggregate) defaultmap(tofrom:allocatable) defaultmap(tofrom:pointer)
# 527 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                    do l = is3_viscous%beg, is3_viscous%end
                        do j = is2_viscous%beg, is2_viscous%end - 1
                            do k = is1_viscous%beg + 1, is1_viscous%end - 1

# 531 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#if defined(MFC_OpenACC)
# 531 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$acc loop seq 
# 531 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 531 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp loop bind(thread)
# 531 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                                do i = iv%beg, iv%end
                                    dqR_prim_dx_n(2)%vf(i)%sf(k, j, l) = (dqL_prim_dx_n(1)%vf(i)%sf(k, j + 1, &
                                                  & l) + dqR_prim_dx_n(1)%vf(i)%sf(k, j + 1, l) + dqL_prim_dx_n(1)%vf(i)%sf(k, j, &
                                                  & l) + dqR_prim_dx_n(1)%vf(i)%sf(k, j, l))

                                    dqR_prim_dx_n(2)%vf(i)%sf(k, j, l) = 25.e-2_wp*dqR_prim_dx_n(2)%vf(i)%sf(k, j, l)
                                end do
                            end do
                        end do
                    end do
!$acc parallel loop collapse(3)
# 566 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 566 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 566 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 566 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 566 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp target teams loop defaultmap(firstprivate:scalar) bind(teams,parallel) collapse(3) defaultmap(tofrom:aggregate) defaultmap(tofrom:allocatable) defaultmap(tofrom:pointer)
# 566 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                    do l = is3_viscous%beg, is3_viscous%end
                        do k = is2_viscous%beg + 1, is2_viscous%end - 1
                            do j = is1_viscous%beg + 1, is1_viscous%end

# 570 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#if defined(MFC_OpenACC)
# 570 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$acc loop seq 
# 570 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 570 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp loop bind(thread)
# 570 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                                do i = iv%beg, iv%end
                                    dqL_prim_dy_n(1)%vf(i)%sf(j, k, l) = (dqL_prim_dy_n(2)%vf(i)%sf(j, k, &
                                                  & l) + dqR_prim_dy_n(2)%vf(i)%sf(j, k, l) + dqL_prim_dy_n(2)%vf(i)%sf(j - 1, k, &
                                                  & l) + dqR_prim_dy_n(2)%vf(i)%sf(j - 1, k, l))

                                    dqL_prim_dy_n(1)%vf(i)%sf(j, k, l) = 25.e-2_wp*dqL_prim_dy_n(1)%vf(i)%sf(j, k, l)
                                end do
                            end do
                        end do
                    end do
!$acc parallel loop collapse(3)
# 583 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 583 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 583 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 583 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 583 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp target teams loop defaultmap(firstprivate:scalar) bind(teams,parallel) collapse(3) defaultmap(tofrom:aggregate) defaultmap(tofrom:allocatable) defaultmap(tofrom:pointer)
# 583 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                    do l = is3_viscous%beg, is3_viscous%end
                        do k = is2_viscous%beg + 1, is2_viscous%end - 1
                            do j = is1_viscous%beg, is1_viscous%end - 1

# 587 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#if defined(MFC_OpenACC)
# 587 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$acc loop seq 
# 587 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 587 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp loop bind(thread)
# 587 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                                do i = iv%beg, iv%end
                                    dqR_prim_dy_n(1)%vf(i)%sf(j, k, l) = (dqL_prim_dy_n(2)%vf(i)%sf(j + 1, k, &
                                                  & l) + dqR_prim_dy_n(2)%vf(i)%sf(j + 1, k, l) + dqL_prim_dy_n(2)%vf(i)%sf(j, k, &
                                                  & l) + dqR_prim_dy_n(2)%vf(i)%sf(j, k, l))

                                    dqR_prim_dy_n(1)%vf(i)%sf(j, k, l) = 25.e-2_wp*dqR_prim_dy_n(1)%vf(i)%sf(j, k, l)
                                end do
                            end do
                        end do
                    end do
!$acc parallel loop collapse(3)
# 622 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 622 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 622 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 622 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 622 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp target teams loop defaultmap(firstprivate:scalar) bind(teams,parallel) collapse(3) defaultmap(tofrom:aggregate) defaultmap(tofrom:allocatable) defaultmap(tofrom:pointer)
# 622 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                        do j = is3_viscous%beg + 1, is3_viscous%end
                            do l = is2_viscous%beg, is2_viscous%end
                                do k = is1_viscous%beg, is1_viscous%end

# 626 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#if defined(MFC_OpenACC)
# 626 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$acc loop seq 
# 626 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 626 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp loop bind(thread)
# 626 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                                    do i = iv%beg, iv%end
                                        dqL_prim_dz_n(3)%vf(i)%sf(k, l, j) = (q_prim_qp%vf(i)%sf(k, l, j) - q_prim_qp%vf(i)%sf(k, &
                                                      & l, j - 1))/(z_cc(j) - z_cc(j - 1))
                                    end do
                                end do
                            end do
                        end do
!$acc parallel loop collapse(3)
# 636 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 636 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 636 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 636 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 636 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp target teams loop defaultmap(firstprivate:scalar) bind(teams,parallel) collapse(3) defaultmap(tofrom:aggregate) defaultmap(tofrom:allocatable) defaultmap(tofrom:pointer)
# 636 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                        do j = is3_viscous%beg, is3_viscous%end - 1
                            do l = is2_viscous%beg, is2_viscous%end
                                do k = is1_viscous%beg, is1_viscous%end

# 640 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#if defined(MFC_OpenACC)
# 640 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$acc loop seq 
# 640 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 640 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp loop bind(thread)
# 640 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                                    do i = iv%beg, iv%end
                                        dqR_prim_dz_n(3)%vf(i)%sf(k, l, j) = (q_prim_qp%vf(i)%sf(k, l, &
                                                      & j + 1) - q_prim_qp%vf(i)%sf(k, l, j))/(z_cc(j + 1) - z_cc(j))
                                    end do
                                end do
                            end do
                        end do
!$acc parallel loop collapse(3)
# 673 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 673 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 673 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 673 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 673 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp target teams loop defaultmap(firstprivate:scalar) bind(teams,parallel) collapse(3) defaultmap(tofrom:aggregate) defaultmap(tofrom:allocatable) defaultmap(tofrom:pointer)
# 673 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                        do l = is3_viscous%beg + 1, is3_viscous%end - 1
                            do k = is2_viscous%beg, is2_viscous%end
                                do j = is1_viscous%beg + 1, is1_viscous%end

# 677 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#if defined(MFC_OpenACC)
# 677 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$acc loop seq 
# 677 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 677 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp loop bind(thread)
# 677 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                                    do i = iv%beg, iv%end
                                        dqL_prim_dz_n(1)%vf(i)%sf(j, k, l) = (dqL_prim_dz_n(3)%vf(i)%sf(j, k, &
                                                      & l) + dqR_prim_dz_n(3)%vf(i)%sf(j, k, &
                                                      & l) + dqL_prim_dz_n(3)%vf(i)%sf(j - 1, k, &
                                                      & l) + dqR_prim_dz_n(3)%vf(i)%sf(j - 1, k, l))

                                        dqL_prim_dz_n(1)%vf(i)%sf(j, k, l) = 25.e-2_wp*dqL_prim_dz_n(1)%vf(i)%sf(j, k, l)
                                    end do
                                end do
                            end do
                        end do
!$acc parallel loop collapse(3)
# 691 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 691 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 691 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 691 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 691 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp target teams loop defaultmap(firstprivate:scalar) bind(teams,parallel) collapse(3) defaultmap(tofrom:aggregate) defaultmap(tofrom:allocatable) defaultmap(tofrom:pointer)
# 691 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                        do l = is3_viscous%beg + 1, is3_viscous%end - 1
                            do k = is2_viscous%beg, is2_viscous%end
                                do j = is1_viscous%beg, is1_viscous%end - 1

# 695 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#if defined(MFC_OpenACC)
# 695 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$acc loop seq 
# 695 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 695 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp loop bind(thread)
# 695 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                                    do i = iv%beg, iv%end
                                        dqR_prim_dz_n(1)%vf(i)%sf(j, k, l) = (dqL_prim_dz_n(3)%vf(i)%sf(j + 1, k, &
                                                      & l) + dqR_prim_dz_n(3)%vf(i)%sf(j + 1, k, &
                                                      & l) + dqL_prim_dz_n(3)%vf(i)%sf(j, k, l) + dqR_prim_dz_n(3)%vf(i)%sf(j, k, &
                                                      & l))

                                        dqR_prim_dz_n(1)%vf(i)%sf(j, k, l) = 25.e-2_wp*dqR_prim_dz_n(1)%vf(i)%sf(j, k, l)
                                    end do
                                end do
                            end do
                        end do
!$acc parallel loop collapse(3)
# 734 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 734 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 734 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 734 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 734 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp target teams loop defaultmap(firstprivate:scalar) bind(teams,parallel) collapse(3) defaultmap(tofrom:aggregate) defaultmap(tofrom:allocatable) defaultmap(tofrom:pointer)
# 734 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                        do l = is3_viscous%beg + 1, is3_viscous%end - 1
                            do j = is2_viscous%beg + 1, is2_viscous%end
                                do k = is1_viscous%beg, is1_viscous%end

# 738 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#if defined(MFC_OpenACC)
# 738 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$acc loop seq 
# 738 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 738 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp loop bind(thread)
# 738 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                                    do i = iv%beg, iv%end
                                        dqL_prim_dz_n(2)%vf(i)%sf(k, j, l) = (dqL_prim_dz_n(3)%vf(i)%sf(k, j, &
                                                      & l) + dqR_prim_dz_n(3)%vf(i)%sf(k, j, l) + dqL_prim_dz_n(3)%vf(i)%sf(k, &
                                                      & j - 1, l) + dqR_prim_dz_n(3)%vf(i)%sf(k, j - 1, l))

                                        dqL_prim_dz_n(2)%vf(i)%sf(k, j, l) = 25.e-2_wp*dqL_prim_dz_n(2)%vf(i)%sf(k, j, l)
                                    end do
                                end do
                            end do
                        end do
!$acc parallel loop collapse(3)
# 751 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 751 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 751 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 751 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 751 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp target teams loop defaultmap(firstprivate:scalar) bind(teams,parallel) collapse(3) defaultmap(tofrom:aggregate) defaultmap(tofrom:allocatable) defaultmap(tofrom:pointer)
# 751 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                        do l = is3_viscous%beg + 1, is3_viscous%end - 1
                            do j = is2_viscous%beg, is2_viscous%end - 1
                                do k = is1_viscous%beg, is1_viscous%end

# 755 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#if defined(MFC_OpenACC)
# 755 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$acc loop seq 
# 755 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 755 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp loop bind(thread)
# 755 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                                    do i = iv%beg, iv%end
                                        dqR_prim_dz_n(2)%vf(i)%sf(k, j, l) = (dqL_prim_dz_n(3)%vf(i)%sf(k, j + 1, &
                                                      & l) + dqR_prim_dz_n(3)%vf(i)%sf(k, j + 1, &
                                                      & l) + dqL_prim_dz_n(3)%vf(i)%sf(k, j, l) + dqR_prim_dz_n(3)%vf(i)%sf(k, j, &
                                                      & l))

                                        dqR_prim_dz_n(2)%vf(i)%sf(k, j, l) = 25.e-2_wp*dqR_prim_dz_n(2)%vf(i)%sf(k, j, l)
                                    end do
                                end do
                            end do
                        end do
!$acc parallel loop collapse(3)
# 792 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 792 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 792 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 792 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 792 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp target teams loop defaultmap(firstprivate:scalar) bind(teams,parallel) collapse(3) defaultmap(tofrom:aggregate) defaultmap(tofrom:allocatable) defaultmap(tofrom:pointer)
# 792 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                        do j = is3_viscous%beg + 1, is3_viscous%end
                            do l = is2_viscous%beg + 1, is2_viscous%end - 1
                                do k = is1_viscous%beg, is1_viscous%end

# 796 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#if defined(MFC_OpenACC)
# 796 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$acc loop seq 
# 796 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 796 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp loop bind(thread)
# 796 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                                    do i = iv%beg, iv%end
                                        dqL_prim_dy_n(3)%vf(i)%sf(k, l, j) = (dqL_prim_dy_n(2)%vf(i)%sf(k, l, &
                                                      & j) + dqR_prim_dy_n(2)%vf(i)%sf(k, l, j) + dqL_prim_dy_n(2)%vf(i)%sf(k, l, &
                                                      & j - 1) + dqR_prim_dy_n(2)%vf(i)%sf(k, l, j - 1))

                                        dqL_prim_dy_n(3)%vf(i)%sf(k, l, j) = 25.e-2_wp*dqL_prim_dy_n(3)%vf(i)%sf(k, l, j)
                                    end do
                                end do
                            end do
                        end do
!$acc parallel loop collapse(3)
# 809 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 809 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 809 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 809 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 809 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp target teams loop defaultmap(firstprivate:scalar) bind(teams,parallel) collapse(3) defaultmap(tofrom:aggregate) defaultmap(tofrom:allocatable) defaultmap(tofrom:pointer)
# 809 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                        do j = is3_viscous%beg, is3_viscous%end - 1
                            do l = is2_viscous%beg + 1, is2_viscous%end - 1
                                do k = is1_viscous%beg, is1_viscous%end

# 813 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#if defined(MFC_OpenACC)
# 813 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$acc loop seq 
# 813 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 813 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp loop bind(thread)
# 813 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                                    do i = iv%beg, iv%end
                                        dqR_prim_dy_n(3)%vf(i)%sf(k, l, j) = (dqL_prim_dy_n(2)%vf(i)%sf(k, l, &
                                                      & j + 1) + dqR_prim_dy_n(2)%vf(i)%sf(k, l, &
                                                      & j + 1) + dqL_prim_dy_n(2)%vf(i)%sf(k, l, &
                                                      & j) + dqR_prim_dy_n(2)%vf(i)%sf(k, l, j))

                                        dqR_prim_dy_n(3)%vf(i)%sf(k, l, j) = 25.e-2_wp*dqR_prim_dy_n(3)%vf(i)%sf(k, l, j)
                                    end do
                                end do
                            end do
                        end do
!$acc parallel loop collapse(3)
# 849 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 849 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 849 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 849 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 849 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp target teams loop defaultmap(firstprivate:scalar) bind(teams,parallel) collapse(3) defaultmap(tofrom:aggregate) defaultmap(tofrom:allocatable) defaultmap(tofrom:pointer)
# 849 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                        do j = is3_viscous%beg + 1, is3_viscous%end
                            do l = is2_viscous%beg, is2_viscous%end
                                do k = is1_viscous%beg + 1, is1_viscous%end - 1

# 853 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#if defined(MFC_OpenACC)
# 853 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$acc loop seq 
# 853 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 853 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp loop bind(thread)
# 853 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                                    do i = iv%beg, iv%end
                                        dqL_prim_dx_n(3)%vf(i)%sf(k, l, j) = (dqL_prim_dx_n(1)%vf(i)%sf(k, l, &
                                                      & j) + dqR_prim_dx_n(1)%vf(i)%sf(k, l, j) + dqL_prim_dx_n(1)%vf(i)%sf(k, l, &
                                                      & j - 1) + dqR_prim_dx_n(1)%vf(i)%sf(k, l, j - 1))

                                        dqL_prim_dx_n(3)%vf(i)%sf(k, l, j) = 25.e-2_wp*dqL_prim_dx_n(3)%vf(i)%sf(k, l, j)
                                    end do
                                end do
                            end do
                        end do
!$acc parallel loop collapse(3)
# 865 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 865 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 865 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 865 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"

# 865 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp target teams loop defaultmap(firstprivate:scalar) bind(teams,parallel) collapse(3) defaultmap(tofrom:aggregate) defaultmap(tofrom:allocatable) defaultmap(tofrom:pointer)
# 865 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                        do j = is3_viscous%beg, is3_viscous%end - 1
                            do l = is2_viscous%beg, is2_viscous%end
                                do k = is1_viscous%beg + 1, is1_viscous%end - 1

# 869 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#if defined(MFC_OpenACC)
# 869 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$acc loop seq 
# 869 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#elif defined(MFC_OpenMP)
# 869 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
!$omp loop bind(thread)
# 869 "/scratch/zyi/workspace/dev-DaCe/MFC-DaCe/build-d257/dace_mirror/src/simulation/m_viscous.fpp"
#endif
                                    do i = iv%beg, iv%end
                                        dqR_prim_dx_n(3)%vf(i)%sf(k, l, j) = (dqL_prim_dx_n(1)%vf(i)%sf(k, l, &
                                                      & j + 1) + dqR_prim_dx_n(1)%vf(i)%sf(k, l, &
                                                      & j + 1) + dqL_prim_dx_n(1)%vf(i)%sf(k, l, &
                                                      & j) + dqR_prim_dx_n(1)%vf(i)%sf(k, l, j))

                                        dqR_prim_dx_n(3)%vf(i)%sf(k, l, j) = 25.e-2_wp*dqR_prim_dx_n(3)%vf(i)%sf(k, l, j)
                                    end do
                                end do
                            end do
                        end do
end subroutine shapes
