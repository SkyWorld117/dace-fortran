"""Link-time names of module procedures in supported caller compilers.

The generated stand-in and binding are compiled by the caller's compiler. This only
selects the symbol to wrap; it does not translate Fortran descriptors between compilers.
"""


def procedure_symbol(module: str, procedure: str, compiler: str = "nvfortran") -> str:
    """Return the caller's linker symbol for a Fortran module procedure."""
    module, procedure = module.lower(), procedure.lower()
    if compiler == "nvfortran":
        return f"{module}_{procedure}_"
    if compiler == "gfortran":
        return f"__{module}_MOD_{procedure}"
    raise ValueError(f"unsupported Fortran caller compiler: {compiler}")
