from __future__ import annotations
import typing
from typing import Self
from numba.experimental import jitclass
from pyhipp.astro.cosmology.model import LambdaCDM, predefined as predef_cosms
from pyhipp.astro.quantity.unit_system import _UnitSystem
from pyhipp.astro.cosmology.flat_lambda_cdm_nr import _FlatLambdaCDMNR
from pyhipp.numerical.interpolate import bisearch_interp
import numpy as np
import numba
from pyhipp.core.abc import HasDictRepr
from functools import cached_property
from scipy import special


@jitclass
class _Context:
    cosm: _FlatLambdaCDMNR
    us: _UnitSystem
    n_r_bins: int
    optim_max_iter: int

    def __init__(self, cosm: _FlatLambdaCDMNR, n_r_bins, optim_max_iter) -> None:
        self.cosm = cosm
        self.us = cosm.us
        self.n_r_bins = n_r_bins
        self.optim_max_iter = optim_max_iter


default_cosm = predef_cosms['tng']     # Planch 2015


class Context(HasDictRepr):

    repr_attr_keys = ('n_r_bins', 'optim_max_iter')

    def __init__(self, impl: _Context):

        self._impl = impl

    @classmethod
    def from_cosm(
            cls, cosm: LambdaCDM = default_cosm, n_r_bins=128,
            optim_max_iter=1024):
        _cosm = _FlatLambdaCDMNR(hubble=cosm.hubble,
                                 omega_m0=cosm.omega_m0, omega_b0=cosm.omega_b0,
                                 n_spec=cosm.n_spec, sigma_8=cosm.sigma_8)
        _ctx = _Context(_cosm, n_r_bins, optim_max_iter)
        return cls(_ctx)


default_ctx = Context.from_cosm()


@jitclass
class _SphSymmProfile:
    _lr_nodes: numba.float64[:]
    _dMs: numba.float64[:]
    _ctx: _Context

    _lr_min: float
    _lr_max: float
    _n_r_bins: int
    _n_r_nodes: int

    _lr_cents: numba.float64[:]
    _r_cents: numba.float64[:]
    _r_nodes: numba.float64[:]

    _M_soft: float
    _dVs: numba.float64[:]
    _rho_cents: numba.float64[:]
    _cM_nodes: numba.float64[:]
    _cM_cents: numba.float64[:]

    def __init__(self, lr_nodes, dMs, ctx) -> None:
        '''
        All data are not copied.
        '''

        self._lr_nodes = lr_nodes
        self._dMs = dMs
        self._ctx = ctx

        self.__set_up()

    def __set_up(self):
        lr_nodes, dMs = self._lr_nodes, self._dMs

        lr_min, lr_max = lr_nodes[0], lr_nodes[-1]
        n_r_bins = len(dMs)
        n_r_nodes = n_r_bins + 1
        assert n_r_nodes == len(lr_nodes)

        lr_cents = 0.5 * (lr_nodes[1:] + lr_nodes[:-1])
        r_nodes = 10.0**lr_nodes
        r_cents = 10.0**lr_cents

        Vs = (4./3.) * np.pi * r_nodes**3
        dVs = np.diff(Vs)
        rho_cents = dMs / dVs

        rho_0 = rho_cents[0]
        r_min = r_nodes[0]
        M_soft = 4.0 / 3.0 * np.pi * r_min**3 * rho_0

        cM_nodes = np.empty(n_r_nodes, dtype=np.float64)
        cM_nodes[0] = M_soft
        cM_nodes[1:] = M_soft + np.cumsum(dMs)
        cM_cents = 0.5 * (cM_nodes[1:] + cM_nodes[:-1])

        self._lr_min = lr_min
        self._lr_max = lr_max
        self._n_r_bins = n_r_bins
        self._n_r_nodes = n_r_nodes
        self._lr_cents = lr_cents
        self._r_nodes = r_nodes
        self._r_cents = r_cents

        self._M_soft = M_soft
        self._dVs = dVs
        self._rho_cents = rho_cents
        self._cM_nodes = cM_nodes
        self._cM_cents = cM_cents

    def M_enc(self, lr: float):
        lr_min = self._lr_min
        if lr < lr_min:
            return self._M_soft * 10.0 ** (3.0 * (lr - lr_min))
        return bisearch_interp(self._lr_nodes, self._cM_nodes, lr)

    def M_encs(self, lrs: np.ndarray):
        out = np.empty_like(lrs)
        for i, lr in enumerate(lrs):
            out[i] = self.M_enc(lr)
        return out

    def rho(self, lr: float):
        return bisearch_interp(self._lr_cents, self._rho_cents, lr)

    def rhos(self, lrs: np.ndarray):
        out = np.empty_like(lrs)
        for i, lr in enumerate(lrs):
            out[i] = self.rho(lr)
        return out


class SphSymmProfile(HasDictRepr):

    repr_attr_keys = ('n_r_nodes', 'lr_min', 'lr_max', 'M_total')

    def __init__(self, impl: _SphSymmProfile) -> None:

        self._impl = impl

    @property
    def n_r_nodes(self):
        return self._impl._n_r_nodes

    @property
    def lr_min(self):
        return self._impl._lr_min

    @property
    def lr_max(self):
        return self._impl._lr_max
    
    @property
    def lr_nodes(self):
        return self._impl._lr_nodes
    
    @property
    def lr_cents(self):
        return self._impl._lr_cents

    @property
    def M_total(self):
        return self._impl._cM_nodes[-1]

    @property
    def ctx(self):
        return Context(self._impl._ctx)
    
    def M_encs(self, lrs: np.ndarray):
        return self._impl.M_encs(lrs)
    
    def rhos(self, lrs: np.ndarray):
        return self._impl.rhos(lrs)

    @classmethod
    def from_bins(cls, lr_nodes: np.ndarray, dMs: np.ndarray,
                  ctx=default_ctx):
        impl = _SphSymmProfile(lr_nodes, dMs, ctx._impl)
        return cls(impl)

    def combined(self, pfs: tuple[SphSymmProfile], lrs=None):
        if lrs is None:
            lrs = self._impl._lr_nodes
        cMs = self._impl.M_encs(lrs)
        for pf in pfs:
            cMs += pf._impl.M_encs(lrs)
        dMs = np.diff(cMs)
        return self.from_bins(lrs, dMs, self.ctx)


@jitclass
class _RotCurve:

    _lr_nodes: numba.float64[:]
    _Vc_sq_nodes: numba.float64[:]
    _ctx: _Context

    _lr_min: float
    _lr_max: float
    _Vc_sq_soft: float
    _Vc_sq_out: float

    def __init__(self, lr_nodes, Vc_sq_nodes, ctx) -> None:
        self._lr_nodes = lr_nodes
        self._Vc_sq_nodes = Vc_sq_nodes
        self._ctx = ctx
        self._lr_min = lr_nodes[0]
        self._lr_max = lr_nodes[-1]
        self._Vc_sq_soft = Vc_sq_nodes[0]
        self._Vc_sq_out = Vc_sq_nodes[-1]

    def Vc_sq(self, lr: float):
        lr_min, lr_max = self._lr_min, self._lr_max
        if lr < lr_min:
            return self._Vc_sq_soft * 10.0 ** (2.0 * (lr - lr_min))
        elif lr > lr_max:
            return self._Vc_sq_out * 10.0 ** (lr_max - lr)
        return bisearch_interp(self._lr_nodes, self._Vc_sq_nodes, lr)

    def Vc_sqs(self, lrs: np.ndarray):
        out = np.empty_like(lrs)
        for i, lr in enumerate(lrs):
            out[i] = self.Vc_sq(lr)
        return out


class RotCurve(HasDictRepr):
    repr_attr_keys = ('n_r_nodes',)

    def __init__(self, impl: _RotCurve) -> None:
        self._impl = impl

    @property
    def n_r_nodes(self):
        return len(self._impl._lr_nodes)

    @property
    def ctx(self):
        return Context(self._impl._ctx)

    @classmethod
    def from_bins(cls, lr_nodes: np.ndarray, Vc_sq_nodes: np.ndarray,
                  ctx=default_ctx):
        impl = _RotCurve(lr_nodes, Vc_sq_nodes, ctx._impl)
        return cls(impl)


class SphSymmObj(HasDictRepr):

    repr_attr_keys = 'profile',
    profile: SphSymmProfile
    ctx: Context

    @cached_property
    def rotation(self):
        pf = self.profile._impl
        G = pf._ctx.us.gravity_constant
        rs, lrs = pf._r_nodes, pf._lr_nodes
        cMs = pf._cM_nodes

        Vc_sqs = G * cMs / rs
        return RotCurve.from_bins(lrs, Vc_sqs, self.ctx)

    def replaced(self, **init_kws) -> Self:
        raise NotImplementedError('Not implemented yet.')


class Halo(SphSymmObj):
    repr_attr_keys = SphSymmObj.repr_attr_keys + ('M_v', 'R_v', 'V_v')
    M_v: float
    R_v: float

    @property
    def V_v(self):
        G = self.ctx._impl.us.gravity_constant
        V_v = np.sqrt(G * self.M_v / self.R_v)
        return V_v


class Baryon(SphSymmObj):
    repr_attr_keys = SphSymmObj.repr_attr_keys + ('M', 'R')

    M: float
    R: float


class SphSymmHalo(Halo):
    def __init__(self,
                 R_v, profile: SphSymmProfile):

        M_v = profile._impl.M_enc(np.log10(R_v))
        self.M_v = M_v
        self.R_v = R_v
        self.profile = profile

    @property
    def ctx(self):
        return self.profile.ctx


class NFW(Halo):

    repr_attr_keys = Halo.repr_attr_keys + ('c', 'R_s', 'R_in', 'R_out')

    def __init__(
            self, M_v: float, R_v: float, c: float,
            f_in=1.0e-5, f_out=10.0,
            ctx=default_ctx):

        R_in = f_in * R_v
        R_out = f_out * R_v
        R_s = R_v / c

        self.M_v = M_v
        self.R_v = R_v
        self.c = c
        self.R_s = R_s
        self.R_in = R_in
        self.R_out = R_out
        self.ctx = ctx

    @cached_property
    def profile(self):
        n_rs = self.ctx._impl.n_r_bins + 1
        R_in, R_out, R_s = self.R_in, self.R_out, self.R_s
        M_v, c = self.M_v, self.c

        lrs = np.linspace(np.log10(R_in), np.log10(R_out), n_rs)
        rs = 10.0**lrs
        xs = rs / R_s
        mu_c = self.mu_at(c)
        mus = self.mu_at(xs)
        cMs = (M_v/mu_c) * mus
        dMs = np.diff(cMs)

        return SphSymmProfile.from_bins(lrs, dMs, self.ctx)

    @staticmethod
    def mu_at(x: np.ndarray):
        return np.log(1. + x) - x / (1. + x)

    def replaced(self, **init_kws):
        kw = {
            'M_v': self.M_v,
            'R_v': self.R_v,
            'c': self.c,
            'f_in': self.R_in / self.R_v,
            'f_out': self.R_out / self.R_v,
            'ctx': self.ctx,
        } | init_kws
        return self.__class__(**kw)


class SphSymmBaryon(Baryon):
    def __init__(self,
                 R, profile: SphSymmProfile):

        self.M = profile.M_total
        self.R = R
        self.profile = profile

    @property
    def ctx(self):
        return self.profile.ctx

    def replaced(self, **init_kws):
        kws = {
            'R': self.R,
            'profile': self.profile,
        } | init_kws
        return self.__class__(**kws)


class ExpSph(Baryon):

    def __init__(self, M: float, R: float, f_in=1.0e-4, f_out=20.0,
                 ctx=default_ctx):

        Sigma0 = M / (2.0 * np.pi * R**2)
        R_in = f_in * R
        R_out = f_out * R

        self.M = M
        self.R = R
        self.R_in = R_in
        self.R_out = R_out
        self.Sigma0 = Sigma0
        self.ctx = ctx

    @cached_property
    def profile(self):
        n_rs = self.ctx._impl.n_r_bins + 1
        R_in, R_out = self.R_in, self.R_out
        M, R = self.M, self.R

        lrs = np.linspace(np.log10(R_in), np.log10(R_out), n_rs)
        rs = 10.0**lrs
        xs = rs / R
        mus = self.mu_at(xs)
        cMs = M * mus
        dMs = np.diff(cMs)

        return SphSymmProfile.from_bins(lrs, dMs, self.ctx)

    @staticmethod
    def mu_at(x: np.ndarray):
        return 1.0 - (1.0 + x) * np.exp(-x)


class ExpThin(Baryon):
    '''
    
    Useful relations:
    R_eff = R * 1.678 (Yu Rong+ 2024).
    For Hernquist rho = M_b r_sb / (2 pi r) / (r + r_sb)^3, r_e = 1.815 r_sb 
    (Shengqi Yang+ 24).
    '''

    def __init__(self, M: float, R: float, f_in=1.0e-2, f_out=20.0,
                 ctx=default_ctx):

        Sigma0 = M / (2.0 * np.pi * R**2)
        R_in = f_in * R
        R_out = f_out * R

        self.M = M
        self.R = R
        self.R_in = R_in
        self.R_out = R_out
        self.Sigma0 = Sigma0
        self.ctx = ctx

    def replaced(self, **init_kws):
        kws = {
            'M': self.M,
            'R': self.R,
            'f_in': self.R_in / self.R,
            'f_out': self.R_out / self.R,
            'ctx': self.ctx,
        } | init_kws
        return self.__class__(**kws)

    @cached_property
    def sph_symm(self):
        f_in, f_out = self.R_in / self.R, self.R_out / self.R
        return ExpSph(self.M, self.R, f_in=f_in, f_out=f_out, ctx=self.ctx)

    @cached_property
    def profile(self):
        return self.sph_symm.profile

    @cached_property
    def rotation(self):
        ctx = self.ctx
        n_rs = ctx._impl.n_r_bins + 1
        G = ctx._impl.us.gravity_constant

        R_in, R_out = self.R_in, self.R_out
        R, Sigma0 = self.R, self.Sigma0
        lrs = np.linspace(np.log10(R_in), np.log10(R_out), n_rs)
        rs = 10.0**lrs
        ys = rs / (2.0 * R)
        I0s = special.i0(ys)
        K0s = special.k0(ys)
        I1s = special.i1(ys)
        K1s = special.k1(ys)
        Vc_sqs = 4.0 * np.pi * G * Sigma0 * \
            R * ys * ys * (I0s * K0s - I1s * K1s)
        rot = RotCurve.from_bins(lrs, Vc_sqs, ctx)
        return rot

@numba.njit
def _adiabatic_evolve(pf_h: _SphSymmProfile, pf_b: _SphSymmProfile,
                      f_b: float, step: float, eps_lr: float,
                      M_soft: float):

    lr_fs = pf_h._lr_nodes.copy()
    M_is = pf_h._cM_nodes
    lr_f_prev = lr_fs[0] - 100.
    for i in range(len(lr_fs)):
        lr_i, M_i = lr_fs[i], M_is[i]/(1. - f_b) + M_soft
        C = np.log10(M_i) + lr_i
        lr_f = lr_i
        while True:
            M_f = pf_b.M_enc(lr_f) + M_i
            lr_f_next = C - np.log10(M_f)
            lr_f_next = (lr_f_next - lr_f) * step + lr_f
            if np.abs(lr_f_next - lr_f) < eps_lr:
                break
            lr_f = lr_f_next
        lr_f = max(lr_f, lr_f_prev + eps_lr)
        lr_fs[i] = lr_f

        lr_f_prev = lr_f

    pf_h_contracted = _SphSymmProfile(lr_fs, pf_h._dMs, pf_h._ctx)
    return pf_h_contracted


def adiabatic_evolve(
        halo: Halo, baryon: Baryon, eps_lr=1.0e-4, step=0.5, M_soft=0.):

    f_b = baryon.M / (baryon.M + halo.M_v)
    pf = _adiabatic_evolve(halo.profile._impl, baryon.profile._impl,
                           f_b=f_b, M_soft=M_soft, eps_lr=eps_lr, step=step)
    pf = SphSymmProfile(pf)

    return SphSymmHalo(halo.R_v, pf)


class AdiabaticModel(HasDictRepr):
    repr_attr_keys = ('eps_lr', 'step', 'M_soft')

    def __init__(self, model=None,
                 eps_lr=1.0e-4, step=0.5, M_soft=0.):
        '''
        @model: 'raw' | None.
        '''
        self.model = model
        self.eps_lr = eps_lr
        self.step = step
        self.M_soft = M_soft

    def __call__(self, halo: Halo, baryon: Baryon):
        model = self.model
        if model is None:
            return halo
        else:
            assert model == 'raw'
            return adiabatic_evolve(halo, baryon, eps_lr=self.eps_lr,
                                    step=self.step, M_soft=self.M_soft)
