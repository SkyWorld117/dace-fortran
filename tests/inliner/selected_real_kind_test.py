"""Kind folding agrees with native Fortran or leaves the intrinsic for that compiler."""
import shutil
import subprocess

import fparser.two.Fortran2003 as f03
from fparser.two.utils import walk
from dace_fortran.fparser_inliner import parse_and_improve
import pytest

from dace_fortran.inliner.ast_desugaring.analysis import const_eval_basic_type, _const_eval_int

EXPRESSIONS=(
    'selected_real_kind(6,37)', 'selected_real_kind(7,37)',
    'selected_real_kind(8,37)', 'selected_real_kind(6,100)',
    'selected_real_kind(15,307)', 'selected_real_kind(7)',
    'selected_real_kind(p=7)', 'selected_real_kind(p=6,r=100)',
    'selected_real_kind(6,r=100)', 'selected_real_kind(7,radix=2)',
    'selected_real_kind(r=37,p=7)',
)

@pytest.fixture(scope='module')
def native_kinds(tmp_path_factory):
    compiler=shutil.which('gfortran')
    if not compiler:pytest.skip('native Fortran compiler required')
    directory=tmp_path_factory.mktemp('real-kind-oracle')
    path=directory/'oracle.f90';exe=directory/'oracle'
    path.write_text('program oracle\n'+'\n'.join('print *, '+e for e in EXPRESSIONS)+'\nend program\n')
    subprocess.run([compiler,str(path),'-o',str(exe)],check=True,capture_output=True)
    return dict(zip(EXPRESSIONS,map(int,subprocess.check_output([str(exe)],text=True).split())))

@pytest.mark.parametrize('expression',EXPRESSIONS)
def test_folded_kind_matches_native(expression,native_kinds):
    tree=parse_and_improve({'kind_case.f90':'program kind_case\ninteger :: unknown\ninteger, parameter :: k='+expression+'\nend program\n'})
    node=next(iter(walk(tree,f03.Intrinsic_Function_Reference)))
    assert int(const_eval_basic_type(node,{}))==native_kinds[expression]
    assert _const_eval_int(node,{})==native_kinds[expression]

@pytest.mark.parametrize('expression',[
    'selected_real_kind(2,4)', 'selected_real_kind(20,300)',
    'selected_real_kind(6,4000)', 'selected_real_kind(r=37)',
    'selected_real_kind(7,radix=10)', 'selected_real_kind(p=unknown,r=37)',
])
def test_ambiguous_or_unmodelled_kind_is_left_to_compiler(expression):
    tree=parse_and_improve({'kind_case.f90':'program kind_case\ninteger :: unknown\ninteger, parameter :: k='+expression+'\nend program\n'})
    node=next(iter(walk(tree,f03.Intrinsic_Function_Reference)))
    assert const_eval_basic_type(node,{}) is None
    assert _const_eval_int(node,{}) is None
