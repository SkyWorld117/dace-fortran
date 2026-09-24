# Rules for a Fortran shim that calls a DaCe AOT library

A DaCe-generated library hands back **device addresses**.  A host Fortran routine that wants to
compute with those addresses cannot do it with ordinary array syntax, so it writes a small shim:
view the device address as a Fortran array, then run an `!$acc parallel loop` over it with
`deviceptr`.  That shim has three clauses, and **two of the three failure modes are silent** —
the run completes, the numbers are wrong, and nothing reports an error.

This page is the rule.  It is a property of any Fortran shim calling a DaCe library, not of any
one codebase.

## The idiom

```fortran
call ensure_buf(d_buf, cap, nbytes, 'buf')              ! 1. the address is non-NULL
call c_f_pointer(d_buf, buf_f, [n])                     ! 2. view it as a Fortran array
!$acc parallel loop collapse(4) deviceptr(buf_f) present(out)   ! 3. name it, and name what you write
do ...
  out(p, q, r, i) = buf_f(...)
end do
```

## Why each clause is load-bearing

### 1. `ensure_buf` — a NULL device address is a bug you want named

`c_f_pointer` accepts a NULL `c_ptr` without complaint and hands back a pointer that dereferences
to nothing.  The loop then reads or writes **through address 0**.  Whatever you get for that is a
kernel fault a long way from the line that caused it, so guard the address before you view it.

Grow-on-demand must be keyed on the **pointer**, not on a shared capacity variable: buffers of one
class often share a `cap`, and testing `cap` alone means only the first of them is ever allocated
while the rest stay NULL.

### 2. `c_f_pointer` — the view, and the name that must match

The name you pass to `deviceptr(...)` has to be the pointer `c_f_pointer` wrote into.  A mismatch
is a compile error, so this clause is the *safe* one — it is listed only because the other two are
read against it.

### 3. `present` — **this is the silent one**

`deviceptr(x)` means "x is already a device address, do **not** consult the present table for it".

That is only half the story.  The loop still **writes a host array**, and `deviceptr` says nothing
about *that* array.  If the written array is not `present`:

* the write does not reach the device,
* nothing errors,
* the run finishes, and
* the host copy even looks plausible afterwards.

The array must be declared device-resident by the host code first — under DaCe's Fortran front end,
`$:GPU_DECLARE(create=...)`.  `present(...)` then fails **loudly** if that precondition ever stops
holding, rather than silently copying nothing.  That asymmetry is the reason to write the clause
explicitly even when it looks redundant.

## Checking it

Missing `present` is a static property of the source, so it is checkable without running anything:

```bash
python -m dace_fortran.acc_residency --check-deviceptr out --check-deviceptr res file1.fpp file2.fpp
```

Naming the arrays is deliberate: deciding *in general* which identifier in a loop body is a host
module array — rather than a local or a dummy — needs scope information this pass does not have,
and naming them is both cheap and exact.  Exit status is `1` on any violation, `0` when clean, and
an **unreadable file is a violation, not a skip** — a path typo must not report what a clean tree
reports.

From Python, `dace_fortran.acc_residency.check_deviceptr_files(paths, targets)` returns
`(n_files, [violation_line, ...])`; `deviceptr_writes_without_present(source, targets)` is the
per-file analysis.  Positive controls live in `tests/acc_residency_test.py` — the check's first
version tested the directive *head* for `"loop"`, which stops at the first clause and therefore
yields `"parallel"`, so it skipped every directive and reported a clean zero both with and without
`present`.  A check that has never failed is not yet a check.

## Where this was measured

MFC's port, 2026-09: a shim copied a staging buffer device→host, transposed it in a serial host
loop, and pushed it back with `!$acc update device` — **108 round trips per run**, 553 MB at 64³
and 32.1 GB at 256³.  It gave back the entire advantage of the ported kernels, which were 1.9×
cheaper than the stock loops.  Keeping the transpose on the device removed it: total GPU time at
64³ fell from 131.2 ms to 76.2 ms, and at 256³ from 6167 ms to 2826 ms.

The fix depends on `present(Re_avg_rsx_vf)` and nothing in that tree would have noticed a one-line
deletion of the clause.  That is what this check is for.
