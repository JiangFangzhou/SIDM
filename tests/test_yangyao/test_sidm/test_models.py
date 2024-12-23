from __future__ import annotations
import typing
from typing import Self
import pytest
from .test_profiles import NFW, ExpThin, AdiabaticModel
from sidm.yangyao.models import SIDMModel
from .conftest import us, cosm
from astropy import units as U

def test_models(nfw_halo: NFW, disk: ExpThin):
    t_age = 5.0e9 / us.u_t_to_yr            # yr to internal units
    sigma_per_m = (1. * U.cm**2 / U.g).to(us.u_l**2/us.u_m).value
    v_sigma_per_m = 100.0 / us.u_v_to_kmps * sigma_per_m
    models = [
        SIDMModel(None, t_age=t_age),
        SIDMModel('jeans', t_age=t_age, sigma_per_m=sigma_per_m),
        SIDMModel('tanh', t_age=t_age, v_sigma_per_m=v_sigma_per_m)
    ]
    for model in models:
        contracted_halo = AdiabaticModel('raw')(nfw_halo, disk)
        for halo in [nfw_halo, contracted_halo]:
            for sidm_halo in [model(halo), model(halo, disk)]:
                print(sidm_halo)
                print(sidm_halo.profile)
                print(sidm_halo.rotation)
