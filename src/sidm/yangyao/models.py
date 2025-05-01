from __future__ import annotations
import typing
from typing import Self
from numba.experimental import jitclass
from pyhipp.numerical.interpolate import bisearch_interp
import numpy as np
import numba
from .profiles import _SphSymmProfile, SphSymmProfile, Halo, Baryon, _Context, ExpSph
from functools import cached_property
from scipy.optimize import minimize
from pyhipp.core.abc import HasDictRepr


@numba.njit
def _solve_jeans_eq(
        pf_src: _SphSymmProfile, pf: _SphSymmProfile, lrs: np.ndarray):

    G = pf_src._ctx.us.gravity_constant

    lr_nodes, cM_cents = pf_src._lr_nodes, pf_src._cM_cents

    Ints = np.zeros_like(lr_nodes)
    i = len(Ints)-1
    while i > 0:
        i -= 1
        lr1, lr2 = lr_nodes[i], lr_nodes[i+1]
        lr = 0.5 * (lr1 + lr2)
        dlr = lr2 - lr1
        r = 10.0**lr
        cM = cM_cents[i]

        rho = pf.rho(lr)
        Ints[i] = Ints[i+1] + cM * rho * dlr / r

    amp = G / np.log10(np.e)
    sigma_sqs = np.zeros_like(lrs)
    for i, lr in enumerate(lrs):
        Int = bisearch_interp(lr_nodes, Ints, lr)
        rho = pf.rho(lr)
        sigma_sqs[i] = Int * amp / rho

    return sigma_sqs


@numba.njit
def _find_zero_point_for_r1(xs, ys):
    i = len(xs)
    assert i >= 2

    i -= 1
    if ys[i] > 0:
        return xs[i]

    while i > 0:
        i -= 1
        if ys[i] > 0:
            break

    if ys[i] <= 0.:
        return xs[i]

    x1, x2 = xs[i], xs[i+1]
    y1, y2 = ys[i], ys[i+1]

    k = (x2-x1) / (y2-y1)
    x0 = k * (0.0 - y1) + x1

    return x0


@jitclass
class _IsoCore:
    _lrs: numba.float64[:]
    _rho_bs: numba.float64[:]
    _R_b: float
    _ctx: _Context

    _rs: numba.float64[:]
    _dVs: numba.float64[:]
    _xs: numba.float64[:]
    _rho_b0: float
    _fs: numba.float64[:]
    _f_cents: numba.float64[:]
    _sigma_0: float
    _rho_0: float
    _rhos: numba.float64[:]
    _M: float

    def __init__(self, lrs, rho_bs, R_b, ctx):

        rs = 10.0**lrs
        Vs = 4.0 / 3.0 * np.pi * rs**3
        dVs = np.diff(Vs)

        xs = rs / R_b
        rho_b0 = rho_bs[0] * xs[0]
        fs = rho_bs * xs / rho_b0
        f_cents = 0.5 * (fs[1:] + fs[:-1])

        self._lrs = lrs
        self._rho_bs = rho_bs
        self._R_b = R_b
        self._ctx = ctx

        self._rs = rs
        self._dVs = dVs

        self._xs = xs
        self._rho_b0 = rho_b0
        self._fs = fs
        self._f_cents = f_cents
        self._sigma_0 = 0.
        self._rho_0 = 0.
        self._rhos = np.zeros_like(xs)
        self._M = 0.

    def solve(self, sigma_0, rho_0):
        G = self._ctx.cosm.us.gravity_constant
        amp = 4.0 * np.pi * G * self._R_b**2 / sigma_0**2
        a1 = amp * rho_0
        a2 = amp * self._rho_b0
        xs, dVs, f_cents = self._xs, self._dVs, self._f_cents
        rhos = self._rhos

        x = xs[0]
        h = 0
        h_prime = a2 / 2.0
        p = h_prime * x * x
        rhos[0] = rho_0
        M = 0.
        for i in range(1, len(xs)):
            dx = xs[i] - xs[i-1]
            x_c = 0.5 * (xs[i] + xs[i-1])
            x_c_sq = x_c * x_c
            h += p * dx / x_c_sq

            f_c = f_cents[i-1]
            p += dx * x_c_sq * (a1 * np.exp(-h) + a2 * f_c / x_c)

            rho = rho_0 * np.exp(-h)
            rho_c = 0.5 * (rhos[i-1] + rho)
            dM = dVs[i-1] * rho_c
            M += dM
            rhos[i] = rho

        self._sigma_0 = sigma_0
        self._rho_0 = rho_0
        self._M = M

    def cMs(self):
        rhos, dVs = self._rhos, self._dVs
        rho_cents = 0.5 * (rhos[1:] + rhos[:-1])
        dMs = dVs * rho_cents
        cMs = np.zeros_like(rhos)
        cMs[1:] = np.cumsum(dMs)
        return cMs


class StitchedHalo(Halo):
    def __init__(self, halo: Halo,
                 baryon: Baryon,
                 t_age: float,
                 sigma_per_m: float):

        self.M_v = halo.M_v
        self.R_v = halo.R_v

        self.halo = halo
        self.baryon = baryon
        self.sigma_per_m = sigma_per_m
        self.t_age = t_age
        self.ctx = halo.profile.ctx

    @cached_property
    def lr1(self):
        sigma_per_m, t_age = self.sigma_per_m, self.t_age

        pf_h = self.halo.profile
        pf_b = self.baryon.profile
        _pf_tot = pf_h.combined([pf_b])._impl
        _pf_h = pf_h._impl

        lrs = pf_h._impl._lr_cents
        rhos = pf_h._impl._rho_cents
        sigmas = np.sqrt(_solve_jeans_eq(_pf_tot, _pf_h, lrs))
        amp = 4. / np.sqrt(np.pi) * sigma_per_m * t_age
        fns = amp * rhos * sigmas - 1.

        lr1 = _find_zero_point_for_r1(lrs, fns)
        rho_r1 = bisearch_interp(lrs, rhos, lr1)
        sigma_r1 = bisearch_interp(lrs, sigmas, lr1)
        M_r1 = _pf_h.M_enc(lr1)

        self.rho_r1 = rho_r1
        self.sigma_r1 = sigma_r1
        self.M_r1 = M_r1

        return lr1

    @cached_property
    def iso_core(self):
        halo, bary = self.halo, self.baryon
        _pf_h = halo.profile._impl
        _pf_b = bary.profile._impl

        ctx = self.ctx._impl
        n_rs = ctx.n_r_bins + 1
        lr1 = self.lr1
        rho_r1, M_r1, sigma_r1 = self.rho_r1, self.M_r1, self.sigma_r1

        lrs = np.linspace(_pf_h._lr_min, lr1, n_rs)
        rho_bs, R_b = _pf_b.rhos(lrs), bary.R
        iso_core = _IsoCore(lrs, rho_bs, R_b, ctx)

        def loss(args):
            lsigma_0, lrho_0 = args
            iso_core.solve(10.0**lsigma_0, 10.0**lrho_0)
            drho = (iso_core._rhos[-1] - rho_r1)/rho_r1
            dM = (iso_core._M - M_r1)/M_r1
            return drho*drho + dM*dM

        sigma_0_lo = 0.5 * sigma_r1
        sigma_0_hi = 2.0 * sigma_r1
        rho_0_lo = rho_r1
        rho_0_hi = _pf_h._rho_cents[0]
        bounds = [(np.log10(sigma_0_lo), np.log10(sigma_0_hi)),
                  (np.log10(rho_0_lo), np.log10(rho_0_hi))]
        guess = np.log10(sigma_r1), np.log10(rho_0_lo)
        opts = {'maxiter': ctx.optim_max_iter}
        sol = minimize(
            loss, guess, bounds=bounds, options=opts,
            method='Nelder-Mead')
        sigma_0, rho_0 = 10.0**sol.x
        iso_core.solve(sigma_0, rho_0)

        self.sol = sol
        return iso_core

    @cached_property
    def profile(self):
        h_cdm, h_iso = self.halo.profile._impl, self.iso_core
        lr1 = self.lr1
        dlr1 = h_iso._lrs[-1] - h_iso._lrs[-2]

        lrs, cMs = h_cdm._lr_nodes, h_cdm._cM_nodes
        sel = lrs > lr1 + dlr1
        lrs = np.concatenate([h_iso._lrs, lrs[sel]])
        cMs = np.concatenate([h_iso.cMs(), cMs[sel]])
        dMs = np.diff(cMs)

        out = SphSymmProfile.from_bins(lrs, dMs, ctx=self.ctx)
        return out


class JeansModel(HasDictRepr):
    repr_attr_keys = ['t_age', 'sigma_per_m']

    def __init__(self, sigma_per_m: float, t_age: float):

        self.sigma_per_m = sigma_per_m
        self.t_age = t_age

    def __call__(self, halo: Halo, baryon: Baryon = None):
        if baryon is None:
            M_b = halo.M_v * 1.0e-10
            R_b = halo.R_v * 1.0e-1
            baryon = ExpSph(M_b, R_b)
        halo_s = StitchedHalo(halo, baryon, t_age=self.t_age,
                              sigma_per_m=self.sigma_per_m)
        return halo_s


class TanhHalo(Halo):
    def __init__(self, halo: Halo,
                 baryon: Baryon,
                 t_age: float,
                 v_sigma_per_m: float,
                 rc2r1=0.45):
        '''
        @halo: DM only (baryon excluded); 
        @baryon: baryon only.
        '''

        self.M_v = halo.M_v
        self.R_v = halo.R_v

        self.halo = halo
        self.baryon = baryon
        self.v_sigma_per_m = v_sigma_per_m
        self.t_age = t_age
        self.rc2r1 = rc2r1
        self.ctx = halo.profile.ctx

    @cached_property
    def lr1(self):
        v_sig, t_age = self.v_sigma_per_m, self.t_age

        _pf_h = self.halo.profile._impl
        lrs = _pf_h._lr_cents
        rhos = _pf_h._rho_cents
        amp = 4. / np.sqrt(np.pi) * v_sig * t_age
        fns = amp * rhos - 1.

        lr1 = _find_zero_point_for_r1(lrs, fns)
        rho_r1 = bisearch_interp(lrs, rhos, lr1)
        M_r1 = _pf_h.M_enc(lr1)

        self.rho_r1 = rho_r1
        self.M_r1 = M_r1

        return lr1

    @cached_property
    def profile(self):
        h_cdm = self.halo.profile._impl
        lrs = h_cdm._lr_nodes
        cMs = h_cdm._cM_nodes

        rs = 10.0**lrs
        rc = self.rc2r1 * 10.0**self.lr1
        r2rc = rs / rc

        cMs = cMs * np.tanh(r2rc)
        dMs = np.diff(cMs)
        out = SphSymmProfile.from_bins(lrs, dMs, ctx=self.ctx)

        return out


class TanhModel(HasDictRepr):
    repr_attr_keys = ['t_age', 'v_sigma_per_m']

    def __init__(self, v_sigma_per_m: float, t_age: float):

        self.v_sigma_per_m = v_sigma_per_m
        self.t_age = t_age

    def __call__(self, halo: Halo, baryon: Baryon = None):
        halo_sidm = TanhHalo(halo, baryon, t_age=self.t_age,
                             v_sigma_per_m=self.v_sigma_per_m)
        return halo_sidm


class NoneModel(HasDictRepr):
    def __init__(self, t_age: float):
        self.t_age = t_age

    def __call__(self, halo: Halo, baryon: Baryon = None):
        return halo


class SIDMModel(HasDictRepr):
    model_map = {
        'tanh': TanhModel,
        'jeans': JeansModel,
        None: NoneModel
    }

    def __init__(self, model=None, **model_kw):
        impl = self.model_map[model](**model_kw)
        self.impl = impl
        self.model = model
        self.model_kw = model_kw

    def __call__(self, halo: Halo, baryon: Baryon = None) -> Halo:
        return self.impl(halo, baryon)
