from __future__ import annotations
import typing
from pathlib import Path
from pyhipp.core import DataTable
from pyhipp.io import h5

def test_dependency_pyhipp():
    dt = DataTable({'a': [1,2,3], 'b': [2.,3.,4.] })
    print(dt)
    

def test_dependency_h5py(tmp_path: Path):
    p = tmp_path / 'file_1.hdf5'
    h5.File.dump_to(p, {'a': 1, 'b': '2'}, 'w')
    