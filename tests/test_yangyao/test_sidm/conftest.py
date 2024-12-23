from __future__ import annotations
import pytest
from sidm.yangyao.profiles import ExpThin, NFW
from pyhipp.astro.cosmology.model import predefined as cosms

cosm = cosms['tng']  # Planck 15
us = cosm.unit_system


@pytest.fixture
def nfw_halo():
    M_v = 1e12 / us.u_m_to_sol          # Msun to internal units
    R_v = cosm.halo_theory.vir_props_crit(M_v, z=0).r_phy
    c = 10
    return NFW(M_v, R_v, c)


@pytest.fixture
def disk():
    M = 10**8.5 / us.u_m_to_sol
    R = 3.5e3 / us.u_l_to_pc            # pc to internal units
    return ExpThin(M, R)
