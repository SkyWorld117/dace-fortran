"""Substituting one routine of a compiled Fortran program by its DaCe build.

The routine's SDFG is specialised to the values a recording of the program shows it runs with, optimised for the
GPU, compiled, and linked in place of the routine's object through a generated Fortran binding that checks the
specialised values on every call and hands the call back to the original routine when one differs.
``make_recorder`` builds the recording stand-in, ``make_substitute`` the substitute; the other modules are the
passes they run.
"""
