"""recorded_elements: the recorded elements of module arrays of records (and of scalars) the build folds into control
flow. Ranks with unequal shares of the grid record different bounds: such an array is not folded."""
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
