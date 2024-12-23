from __future__ import annotations
import typing
from typing import Self
from sidm.yangyao.profiles import ExpThin, NFW, AdiabaticModel
from .conftest import us, cosm

def test_create_nfw_halo(nfw_halo: NFW):
    print(nfw_halo)
    print(nfw_halo.R_v, nfw_halo.M_v)
    print(nfw_halo.profile)
    print(nfw_halo.rotation)

def test_create_disk(disk: ExpThin):
    print(disk)
    print(disk.M, disk.R)
    print(disk.profile)
    print(disk.rotation)

def test_adiabatic_contraction(nfw_halo: NFW, disk: ExpThin):
    none_model = AdiabaticModel()
    raw_model = AdiabaticModel('raw')
    for m in [none_model, raw_model]:
        halo_contracted = m(nfw_halo, disk)
        print(halo_contracted)
    