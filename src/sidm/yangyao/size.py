from __future__ import annotations
import typing
from typing import Self
from numba.experimental import jitclass
from pyhipp.core import DataDict
from pyhipp.astro.cosmology.model import LambdaCDM, predefined as predef_cosms
from pyhipp.astro.quantity.unit_system import _UnitSystem
from pyhipp.astro.cosmology.flat_lambda_cdm_nr import _FlatLambdaCDMNR
from pyhipp.numerical.interpolate import bisearch_interp
import numpy as np
import numba
from pyhipp.core.abc import HasDictRepr
from functools import cached_property
from scipy import special
from .profiles import Halo, Baryon, AdiabaticModel, SphSymmHalo, ExpThin, \
    _RotCurve, SphSymmBaryon, SphSymmProfile
from .models import JeansModel, TanhModel, SIDMModel
from dataclasses import dataclass


class GalHalo(HasDictRepr):
    repr_attr_keys = (
        'halo', 'baryon', 'dm_init', 'dm',
        'spin', 'sAM_h', 'J_h', 't_age',
        'self_grav', 'fR_soft', 'R_soft', 'fM_soft', 'M_soft',
        'm_ada', 'm_sidm', 'M_b', 'M_h', 'M_dm', 'prof_b'
    )

    def __init__(self, halo: Halo, baryon: Baryon,
                 spin=0.035, t_age=0.0,
                 self_grav=True, fR_soft=0.01, fM_soft=1.0e-4,
                 ada_kw={}, sidm_kw={},
                 ):
        '''
        @halo: total, including un-cooled gas that will become into `baryon`.
        @ada: None | 'raw'.
        @sidm: None | 'jeans' | 'tanh'.
        '''
        R_b, M_b, M_h, R_h, V_h = baryon.R, baryon.M, halo.M_v, \
            halo.R_v, halo.V_v
        M_dm = M_h - M_b
        dm_init = halo.replaced(M_v=M_dm)

        M_soft = fM_soft * M_b
        R_soft = fR_soft * R_b

        m_ada = AdiabaticModel(**ada_kw, M_soft=M_soft)
        m_sidm = SIDMModel(**sidm_kw, t_age=t_age)
        
        sAM_h = np.sqrt(2.) * R_h * V_h * spin
        J_h = M_h * sAM_h

        self.halo = halo
        self.baryon = baryon
        self.dm_init = dm_init

        self.spin = spin
        self.sAM_h = sAM_h
        self.J_h = J_h
        self.t_age = t_age

        self.self_grav = self_grav
        self.fR_soft = fR_soft
        self.R_soft = R_soft
        self.fM_soft = fM_soft
        self.M_soft = M_soft
        self.m_ada = m_ada
        self.m_sidm = m_sidm

        self.M_b = M_b
        self.M_h = M_h
        self.M_dm = M_dm
        self.dm = self.apply_models(dm_init, baryon)
        
    def apply_models(self, dm: Halo, b: Baryon):
        dm_ada = self.m_ada(dm, b)
        dm_sidm = self.m_sidm(dm_ada, b)
        return dm_sidm

    @cached_property
    def prof_b(self):
        b, dm = self.baryon, self.dm
        M_soft, R_soft = self.M_soft, self.R_soft
        G = b.ctx._impl.us.gravity_constant

        rot_b, rot_dm = b.rotation._impl, dm.rotation._impl
        pf_b = b.profile._impl

        lr_cs = pf_b._lr_cents
        r_cs = 10.0**lr_cs
        lrs = pf_b._lr_nodes
        rs = 10.0**lrs
        dMs = pf_b._dMs
        cMs = pf_b.M_encs(lrs)
        As = np.pi * rs**2
        Sigmas = cMs / As

        Vc_sqs_b = rot_b.Vc_sqs(lr_cs)
        Vc_sqs_dm = rot_dm.Vc_sqs(lr_cs)
        Vc_sqs = Vc_sqs_dm + G * M_soft / r_cs
        if self.self_grav:
            Vc_sqs = Vc_sqs + Vc_sqs_b
        Vc_sqs = Vc_sqs * r_cs / (r_cs + R_soft)

        Vcs_b = np.sqrt(Vc_sqs)
        Vcs_dm = np.sqrt(Vc_sqs_dm)
        Vcs = np.sqrt(Vc_sqs)
        sAMs = r_cs * Vcs
        dJs = dMs * sAMs
        cJs = np.concatenate([[0], np.cumsum(dJs)])
        sAMs2h = sAMs / self.sAM_h

        M_tot = dMs.sum()
        J_tot = cJs[-1]
        sAM_tot = J_tot / M_tot
        sAM_tot2h = sAM_tot / self.sAM_h
        l_ams = sAMs / sAM_tot

        return DataDict({
            'lrs': lrs, 'lr_cs': lr_cs,
            'dMs': dMs, 'cMs': cMs, 'M_tot': M_tot,
            'Sigmas': Sigmas,
            'Vcs': Vcs, 'Vcs_b': Vcs_b, 'Vcs_dm': Vcs_dm,
            'sAMs': sAMs, 'sAMs2h': sAMs2h,
            'sAM_tot': sAM_tot, 'sAM_tot2h': sAM_tot2h,
            'dJs': dJs, 'cJs': cJs, 'J_tot': J_tot,
            'l_ams': l_ams,
        })


class MMW98(HasDictRepr):
    '''
    Fixed J_d and forced exponential thin disk.
    '''
    repr_attr_keys = ('gh', 'step', 'max_n_iter', 'eps_lr', 'eps_lrho',
                      'f_r_max', 'trace', 'n_iters_dm', 'n_iters_baryon')

    @dataclass
    class Iter:
        dm: SphSymmHalo
        baryon: ExpThin
        f_R: float
        i_R: int
        Js: np.ndarray
        lrs: np.ndarray

    def __init__(
        self,
        gh: GalHalo,
        a_spin=1.,
        b_spin=1.,
        step=0.5,
        max_n_iter=128,
        eps_lr=1.0e-3,
        eps_lrho=1.0e-3,
        f_r_max=100.,
        trace=True,
    ):
        '''
        f_r_max: R_eff * f_r_max gives the max radius to integrate f_R.
        '''
        self.gh = gh
        self.a_spin = a_spin
        self.b_spin = b_spin
        self.step = step
        self.max_n_iter = max_n_iter
        self.eps_lr = eps_lr
        self.eps_lrho = eps_lrho
        self.f_r_max = f_r_max
        self.trace = trace
        self.iters: list[MMW98.Iter] = []
        self.n_iters_dm = 0
        self.n_iters_baryon = 0

    @cached_property
    def baryon(self) -> Baryon:
        gh = self.gh
        b, dm, dm_init = gh.baryon, gh.dm, gh.dm_init

        lrs = dm_init.profile.lr_cents
        lrhos = np.log10(dm.profile.rhos(lrs))
        while True:
            b = self.__iter_for_R(dm, b)
            dm = gh.apply_models(dm_init, b)
            self.n_iters_dm += 1

            lrhos_next = np.log10(dm.profile.rhos(lrs))
            dlrho = np.abs(lrhos_next - lrhos).max()
            if dlrho < self.eps_lrho or self.n_iters_baryon >= self.max_n_iter:
                break
            lrhos = lrhos_next

        self.dm = dm
        return b

    def __iter_for_R(self, dm: Halo, b: Baryon):
        gh = self.gh
        spin = self.a_spin * gh.spin**self.b_spin
        amp = 1. / np.sqrt(2.) * spin * dm.R_v
        while True:
            iter = MMW98.__integ_rot(dm, b, gh.self_grav, self.f_r_max)
            if self.trace:
                self.iters.append(iter)
            R = amp * iter.f_R
            dlR = np.abs(np.log10(R) - np.log10(b.R))
            b = b.replaced(R=R)
            self.n_iters_baryon += 1
            if dlR < self.eps_lr or self.n_iters_baryon >= self.max_n_iter:
                break
        return b

    @staticmethod
    def __integ_rot(
            dm: Halo, b: Baryon, self_grav: bool, f_r_max: float):
        V_v, R_b, M_b = dm.V_v, b.R, b.M
        rot_dm, rot_b = dm.rotation._impl, b.rotation._impl
        lR_max = np.log10(1.678 * R_b * f_r_max)

        lrs = rot_b._lr_nodes
        dlrs = lrs[1:] - lrs[:-1]
        lr_cs = 0.5 * (lrs[1:] + lrs[:-1])
        r_cs = 10.0**lr_cs
        u_cs = r_cs / R_b
        Vc_sqs = rot_dm.Vc_sqs(lr_cs)
        if self_grav:
            Vc_sqs += rot_b.Vc_sqs(lr_cs)
        Vc2vir = np.sqrt(Vc_sqs) / V_v

        dI = Vc2vir * np.exp(- u_cs) * (u_cs**3) * dlrs / np.log10(np.e)
        Is = np.zeros_like(lrs)
        Is[1:] = np.cumsum(dI)
        Js = M_b * R_b * V_v * Is

        i_R = (lrs <= lR_max).nonzero()[0][-1]
        f_R = 2.0 / Is[i_R]

        return MMW98.Iter(dm, b, f_R, i_R, Js, lrs)


@numba.njit
def _find_r_by_sAM(dMs: np.ndarray, sAMs: np.ndarray, lrs: np.ndarray,
                   rot_dm: _RotCurve, self_grav: bool, eps_lr: float,
                   M_soft: float, r_soft: float, step=0.5, max_n_iter=128):
    G = rot_dm._ctx.us.gravity_constant
    lrs_out = np.empty_like(lrs)
    cM = 0.
    lr_prev = -100.0
    for i in range(len(lrs)):
        dM, sAM, lr = dMs[i], sAMs[i], lrs[i]
        lsAM = np.log10(sAM)
        n_iter = 0
        while True:
            r = 10.0**lr
            Vc_sq = rot_dm.Vc_sq(lr) + G * M_soft / r
            if self_grav:
                Vc_sq += G * (cM + dM*.5) / r
            Vc_sq *= r / (r+r_soft)
            lr_next = lsAM - 0.5 * np.log10(Vc_sq)
            lr_next = (lr_next - lr) * step + lr
            lr_next = max(lr_next, lr_prev + eps_lr)
            dlr = np.abs(lr_next - lr)
            lr = lr_next
            n_iter += 1
            if dlr < eps_lr or n_iter >= max_n_iter:
                break

        lr_prev = lr
        lrs_out[i] = lr
        cM += dM
    return lrs_out


class JPdf:
    def __init__(self, lrs: np.ndarray, dMs: np.ndarray,
                 dJs: np.ndarray, sAMs: np.ndarray,
                 J_tot: float, sAM_tot: float):

        assert len(lrs)-1 == len(dMs) == len(dJs) == len(sAMs)

        self.lrs = lrs
        self.dMs = dMs
        self.dJs = dJs
        self.sAMs = sAMs
        self.J_tot = J_tot
        self.sAM_tot = sAM_tot

    @staticmethod
    def from_gal_halo(gh: GalHalo):
        pf_b = gh.prof_b
        lrs = pf_b['lrs']
        dMs, dJs, sAMs = pf_b['dMs', 'dJs', 'sAMs']
        J_tot, sAM_tot = pf_b['J_tot', 'sAM_tot']
        return JPdf(lrs, dMs, dJs, sAMs, J_tot, sAM_tot)

    @staticmethod
    def from_fn_prob(halo: Halo, spin: float, M_b: float,
                     shape='normal',
                     l_width=0.5, l_min=1.0e-4, l_max=5., n_bins=256):

        J_h = np.sqrt(2.0) * halo.M_v * halo.R_v * halo.V_v * spin
        m_b = M_b / halo.M_v
        J_b = J_h * m_b
        sAM_b = J_b / M_b

        llmin, llmax = np.log10(l_min), np.log10(l_max)
        lls = np.linspace(llmin, llmax, n_bins)
        dlls = lls[1:] - lls[:-1]
        ll_cs = 0.5 * (lls[1:] + lls[:-1])
        l_cs = 10.0**ll_cs
        sAMs = l_cs * sAM_b

        if shape == 'normal':
            pdf_cs = np.exp(-0.5 * (l_cs-1.)**2 / l_width**2) * l_cs
        elif shape == 'log-normal':
            pdf_cs = np.exp(-0.5 * (ll_cs-0.)**2 / l_width**2)
        else:
            raise ValueError(f'Unknown shape: {shape}')
        dPs = pdf_cs * dlls
        cPs = np.concatenate([[0.], np.cumsum(dPs)])
        cPs /= cPs[-1]
        dPs = cPs[1:] - cPs[:-1]
        dMs = M_b * dPs
        dJs = sAMs * dMs
        Js = np.concatenate([[0.], np.cumsum(dJs)])

        pf_h = halo.profile._impl
        lrs = np.logspace(pf_h._lr_min-2., pf_h._lr_max, n_bins+1)

        return JPdf(lrs, dMs, dJs, sAMs, J_b, sAM_b)


class FixedJPdf:

    @dataclass
    class Iter:
        dm: SphSymmHalo
        baryon: SphSymmBaryon

    def __init__(self,
                 gh: GalHalo, j_pdf: JPdf,  eps_lr=1.0e-3,
                 max_n_iter=128, trace=True,
                 ):

        self.gh = gh
        self.j_pdf = j_pdf
        self.eps_lr = eps_lr
        self.max_n_iter = max_n_iter
        self.trace = trace
        self.iters: list[FixedJPdf.Iter] = []

    @cached_property
    def baryon(self):
        gh, j_pdf = self.gh, self.j_pdf

        dMs, sAMs = j_pdf.dMs, j_pdf.sAMs
        m_ada, m_sidm = gh.m_ada, gh.m_sidm

        b, dm, dm_init = gh.baryon, gh.dm, gh.dm_init
        R_b = b.R
        M_soft, R_soft = gh.M_soft, gh.R_soft
        self_grav, eps_lr = gh.self_grav, self.eps_lr
        lrs = j_pdf.lrs
        n_iters = 0
        while True:
            rot_dm = dm.rotation._impl
            lrs_next = _find_r_by_sAM(
                dMs, sAMs, lrs[1:], rot_dm, self_grav, eps_lr, M_soft,
                R_soft)
            lrs_next = self.__bin_c2e(lrs_next)
            dlr = np.abs(lrs_next - lrs).max()
            lrs = lrs_next

            pf = SphSymmProfile.from_bins(lrs, dMs, ctx=b.ctx)
            b = SphSymmBaryon(R_b, pf)
            dm = gh.apply_models(dm_init, b)
            n_iters += 1
            if self.trace:
                self.iters.append(FixedJPdf.Iter(dm=dm, baryon=b))
            if dlr < eps_lr or n_iters >= self.max_n_iter:
                break

        return b

    def __bin_c2e(self, xs):
        xs_c = 0.5 * (xs[1:] + xs[:-1])
        dx0, dx1 = xs[1] - xs[0], xs[-1] - xs[-2]
        x0, x1 = xs_c[0], xs_c[-1]
        xs = np.concatenate([[x0-dx0], xs_c, [x1+dx1]])
        return xs
