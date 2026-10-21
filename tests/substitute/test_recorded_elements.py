"""recorded_elements: the recorded elements of module arrays of records (and of scalars) the build folds into control
flow. Ranks with unequal shares of the grid record different bounds: such an array is not folded."""
from pathlib import Path

import dace

from dace_fortran.substitute.make_substitute import recorded_elements


def test_an_array_an_element_of_which_differs_between_the_ranks_is_not_folded(tmp_path):
    sdfg = dace.SDFG("records")
    for name in ("idwbuff_beg", "idwbuff_end"):
        sdfg.add_array(name, [3], dace.int32)
    sdfg.add_array("flags", [2], dace.int32)
    rank = {0: 79, 1: 78}  # m on two ranks of a 159-cell axis
    records = []
    for r, m in rank.items():
        lines = [f"m_global_parameters::idwbuff({d})%beg -4" for d in (1, 2, 3)]
        lines += [f"m_global_parameters::idwbuff(1)%end {m + 4}", "m_global_parameters::idwbuff(2)%end 162",
                  "m_global_parameters::idwbuff(3)%end 162", "m_global_parameters::flags(1) 1",
                  "m_global_parameters::flags(2) 0"]
        records.append(tmp_path / f"s.record.{r}")
        records[-1].write_text("\n".join(lines) + "\n")
    assert recorded_elements(records, sdfg) == {"idwbuff_beg": [-4, -4, -4], "flags": [1, 0]}
    assert recorded_elements(records[:1], sdfg)["idwbuff_end"] == [83, 162, 162]


def test_each_rank_s_elements_that_differ_are_named_as_the_in_call_analysis_reads_them(tmp_path):
    from dace_fortran.substitute.make_substitute import recorded_rank_elements

    sdfg = dace.SDFG("records")
    sdfg.add_array("idwint_end", [2], dace.int32)
    sdfg.add_array("idwint_beg", [2], dace.int32)
    record = tmp_path / "s.record"  # the ranks' records, and all of them in one
    for r, m in {0: 79, 1: 78}.items():
        Path(f"{record}.{r}").write_text(f"m::idwint(1)%end {m}\nm::idwint(2)%end 158\n"
                                         "m::idwint(1)%beg 0\nm::idwint(2)%beg 0\n")
    record.write_text(Path(f"{record}.0").read_text() + Path(f"{record}.1").read_text())
    assert recorded_rank_elements([record], sdfg) == [{"idwint_end[0]@0": 79, "idwint_end[1]@0": 158},
                                                      {"idwint_end[0]@0": 78, "idwint_end[1]@0": 158}]
