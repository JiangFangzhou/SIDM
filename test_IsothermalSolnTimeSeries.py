#########################################################################

# Program that solves the Jeans-Poisson equation for the density profile 
# of SIDM halo in the presence of Hernquist baryon mass distribution.

# In this version, we start with a NFW CDM halo of given halo age, as 
# well as the Henquist baryon distribution, and we find the central DM 
# density (rho_dm0) and the velocity dispersion (sigma_0) by minimizing
# the figure of merit, delta, which measures the fractional difference
# in density and enclosed mass between the SIDM profile and the CDM 
# profile at r_1 ( the characteristic radius within which an average DM 
# particle has experienced one or more self-interaction )

# In this program, we solve for (rho_dm0, sigma_0) for series of t_age,
# for a DMO halo that was used in Nishikawa+20 Fig.6 -- the purpose is to
# compare the isothermal solutions of rho_dm0 to those from solving the 
# fluid-equations. The solutions of (rho_dm0, sigma_0) are stored in
# ./OUTPUT/IsothermalSolnTimeSeries_rhodm0_lgM%.2f_c%.1f_sigmamx%.1f.txt
# and the corresponding profiles are stored in 
# './OUTPUT/IsothermalSolnTimeSeries_density_HiDens_lgM%.2f_c%.1f_sigmamx%.1f.txt'
# './OUTPUT/IsothermalSolnTimeSeries_density_LoDens_lgM%.2f_c%.1f_sigmamx%.1f.txt'
# './OUTPUT/IsothermalSolnTimeSeries_radius_lgM%.2f_c%.1f_sigmamx%.1f.txt'

# Before each repeated run of the same target CDM halo info, clean up the 
# existing output files ./OUTPUT/IsothermalSolnTimeSeries*.txt

# Arthur Fangzhou Jiang 2021 Caltech and Carnegie

######################## set up the environment #########################

import config as cfg
import cosmo as co
import profiles as pr
import aux

import sys
import time
import numpy as np
from scipy.optimize import minimize

import matplotlib as mpl # must import before pyplot
mpl.use('Qt5Agg')
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.size'] = 16  
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

########################### user control ################################

#---target CDM halo

tage_grid = np.logspace(-1.,2.3,50) # [Gyr]

lgMv = 9.89 # [M_sun]
c = 15.8  
sigmamx = 5. # [cm^2/g] 

cfg.Rres = 1e-2 # [kpc] resolution radius

#---objective function
def delta(p,rhob0,r0,rhoCDM1,MCDM1,r):
    """
    Given p=[rho_dm0,sigma_0], evaluate the relative error in stitching
    
        delta = sqrt( delta_rho^2 + delta_M^2 )
        
    where 
    
        delta_rho = | rho_iso(r_1) - rho_CDM(r_1) | / rho_CDM(r_1)
        detla_M = | M_iso(r_1) - M_CDM(r_1) | / M_CDM(r_1)
        
    Syntax:
    
        delta(p,rho1,M1)
        
    where
    
        p: [log(rho_dm0),log(sigma_0)] in [M_sun/kpc^3, kpc/Gyr] (array)
        rhob0: Hernquist central density of baryons [M_sun/kpc^3] (float)
        r0: Hernquist scale radius of baryon distribution [kpc] (float)
        rho1: CDM density to match at r_1 [M_sun/kpc^3] (float)
        M1: CDM enclosed mass to match at r_1 [M_sun] (float)
        r: radii between 0 and r_0, where we compute the SIDM profile
           [kpc] (array)
    """
    rhodm0 = 10.**p[0]
    sigma0 = 10.**p[1]
    a = cfg.FourPiG * r0**2 *rhodm0 / sigma0**2
    b = cfg.FourPiG * r0**2 *rhob0 / sigma0**2
    h = pr.h(r/r0,a,b)
    rho = rhodm0*np.exp(h)
    M = pr.Miso(r,rho)
    drho = (rho[-1] - rhoCDM1) / rhoCDM1    
    dM = (M[-1] - MCDM1) / MCDM1
    return drho**2 + dM**2

#---output control
outfile_rhodm0 = './OUTPUT/IsothermalSolnTimeSeries_rhodm0_lgM%.2f_c%.1f_sigmamx%.1f.txt'%(lgMv,c,sigmamx)
outfile_density_LoDens = './OUTPUT/IsothermalSolnTimeSeries_density_LoDens_lgM%.2f_c%.1f_sigmamx%.1f.txt'%(lgMv,c,sigmamx)
outfile_density_HiDens = './OUTPUT/IsothermalSolnTimeSeries_density_HiDens_lgM%.2f_c%.1f_sigmamx%.1f.txt'%(lgMv,c,sigmamx)
outfile_radius = './OUTPUT/IsothermalSolnTimeSeries_radius_lgM%.2f_c%.1f_sigmamx%.1f.txt'%(lgMv,c,sigmamx)

#---plot control
MakePlot = True
outfig1 = './FIGURE/test_IsothermalSolnTimeSeries_lgMv%.2f_c%.1f_sigmamx%.1f_tage%.4f.pdf'#%(lgMv,c,sigmamx,tage)

lw = 2.5
size = 30.
edgewidth = 0.

r_FullRange = np.logspace(-3,3,200) # [kpc] for plotting the full profile

############################### compute #################################

#---compute the CDM profile
Mv = 10.**lgMv
halo = pr.NFW(Mv,c,Delta=200.,z=0.)
rs = halo.rs
rhos = halo.rho0

#---solve for rho_dm0 and sigma_0
 
f_rhodm0 = open(outfile_rhodm0, 'a')
f_density_LoDens = open(outfile_density_LoDens, 'a')
f_density_HiDens = open(outfile_density_HiDens, 'a')
f_radius = open(outfile_radius, 'a')

for i,tage in enumerate(tage_grid):

    #---find the r_1 radius 
    r1 = pr.r1(halo,sigmamx=sigmamx,tage=tage)

    # evaluate vel dispersion, density, and enclosed mas at r_1
    sigmaCDM1 = halo.sigma(r1)
    rhoCDM1 = halo.rho(r1)
    MCDM1 = halo.M(r1)
    rhoCDMres = halo.rho(cfg.Rres)
    
    # radius series over which we perform the integration
    r = np.logspace(-3.,np.log10(r1),500) 
    
    # define baryon properties -- here we consider DM-only case, so we 
    # use an arbitrarily low value for baryon density and an arbitrarily 
    # large baryon size
    rhob0 = 1e-6 
    r0 = 100.
    
    #---search for the low-density solution -- note that we have 
    #   manually tested the searching range for the low-density soln
    #   in the program test_SolveSIDMprofile_GivenCDMandBaryon_scan.py
    lgrhodm0_init = 0.5*(np.log10(rhoCDM1)+np.log10(rhoCDMres))
    lgsigma0_init = np.log10(sigmaCDM1)
    lgrhodm0_lo = np.log10(rhoCDM1) 
    lgrhodm0_hi = np.log10(rhoCDM1*1e3) # <<< test: upper bound
    lgsigma0_lo = np.log10(0.5*sigmaCDM1)
    lgsigma0_hi = np.log10(2.0*sigmaCDM1)
    res = minimize(delta,[lgrhodm0_init,lgsigma0_init],
        args=(rhob0,r0,rhoCDM1,MCDM1,r),
        bounds=((lgrhodm0_lo,lgrhodm0_hi),(lgsigma0_lo,lgsigma0_hi)),
        )
    rhodm0 = 10.**res.x[0]
    sigma0 = 10.**res.x[1]
    a = cfg.FourPiG * r0**2 *rhodm0 / sigma0**2
    b = cfg.FourPiG * r0**2 *rhob0 / sigma0**2
    h = pr.h(r/r0,a,b)
    rho = rhodm0 * np.exp(h)
    M = pr.Miso(r,rho)
    Vc = np.sqrt(cfg.G*M/r)
    # register
    rhodm0_LoDens = rhodm0
    sigma0_LoDens = sigma0
    rho_LoDens = rho
    Vc_LoDens = Vc
    
    #---search for the intermediate-density solution -- note that we have 
    #   manually tested the searching range for the intermediate-density 
    #   soln in the programs
    #       test_SolveSIDMprofile_GivenCDMandBaryon_scan.py
    #       test_SolveSIDMprofile_GivenCDMandBaryon_DMO.py
    #   Also note that there could a an even higher-density solution,
    #   which we have essentially excluded from the searching range
    lgrhodm0_init = np.log10(rhoCDMres) 
    lgsigma0_init = np.log10(2.*sigmaCDM1)
    lgrhodm0_lo = np.log10(1e2*rhoCDM1)
    lgrhodm0_hi = np.log10(1e4*rhoCDMres)
    lgsigma0_lo = np.log10(0.5*sigmaCDM1)
    lgsigma0_hi = np.log10(2.0*sigmaCDM1)
    res = minimize(delta,[lgrhodm0_init,lgsigma0_init],
        args=(rhob0,r0,rhoCDM1,MCDM1,r),
        bounds=((lgrhodm0_lo,lgrhodm0_hi),(lgsigma0_lo,lgsigma0_hi)),
        method='Powell', # <<< important !
        )
    rhodm0 = 10.**res.x[0]
    sigma0 = 10.**res.x[1]
    a = cfg.FourPiG * r0**2 *rhodm0 / sigma0**2
    b = cfg.FourPiG * r0**2 *rhob0 / sigma0**2
    h = pr.h(r/r0,a,b)
    rho = rhodm0 * np.exp(h)
    M = pr.Miso(r,rho)
    Vc = np.sqrt(cfg.G*M/r)
    # register
    rhodm0_HiDens = rhodm0
    sigma0_HiDens = sigma0
    rho_HiDens = rho
    Vc_HiDens = Vc
    
    #---output
    print('    t=%10.6f, rho0/rhos=%10.6f(lo) %10.6f(hi)'%(tage,rhodm0_LoDens/rhos,rhodm0_HiDens/rhos))
    np.savetxt(f_rhodm0, 
        np.hstack((tage,r1,rhodm0_LoDens,rhodm0_HiDens,sigma0_LoDens,sigma0_HiDens)), 
        fmt='%10.6e', newline=' ')
    f_rhodm0.write('\n')
    
    np.savetxt(f_density_LoDens, np.hstack((tage,rho_LoDens)), fmt='%10.6e', newline=' ')
    f_density_LoDens.write('\n')
    
    np.savetxt(f_density_HiDens, np.hstack((tage,rho_HiDens)), fmt='%10.6e', newline=' ')
    f_density_HiDens.write('\n')
    
    np.savetxt(f_radius, np.hstack((tage,r)), fmt='%10.6e', newline=' ')
    f_radius.write('\n')

    #--------------------------------------------------------------------
    if not MakePlot: continue
    
    plt.close('all') # close all previous figure windows

    # set up the figure window
    fig1 = plt.figure(figsize=(16,5), dpi=80, facecolor='w', edgecolor='k') 
    fig1.subplots_adjust(left=0.06, right=0.93,bottom=0.12, top=0.91,
        hspace=0.25, wspace=0.25)
    gs = gridspec.GridSpec(1, 3) 
    fig1.suptitle(r'$M_\mathrm{v}=10^{%.2f}M_\odot, c=%.1f, \sigma/m_\chi=%.1f\mathrm{cm}^2/\mathrm{g}, t_\mathrm{age}=%.4f\mathrm{Gyr}$'\
        %(lgMv,c,sigmamx, tage),fontsize=14)

    #---
    ax = fig1.add_subplot(gs[0,0])
    ax.set_xlim(0.001,50.)
    ax.set_ylim(1e5,1e13)
    #ax.set_ylim(0.01,1000.)
    ax.set_xlabel(r'$r$  [kpc]')
    ax.set_ylabel(r'$\rho$  [$M_\odot$kpc$^{-3}$]')
    #ax.set_ylabel(r'$\rho$  [GeV cm$^{-3}$]')
    #ax.set_xticks(major_ticks)
    #ax.set_xticks(minor_ticks,minor=True)
    #ax.set_yticks(major_ticks)
    #ax.set_yticks(minor_ticks,minor=True)
    # grid
    ax.grid(which='minor', alpha=0.2)                                                
    ax.grid(which='major', alpha=0.4)
    # 
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params('both',direction='in',top='on',right='on',length=10,
        width=1,which='major',)
    ax.tick_params('both',direction='in',top='on',right='on',length=5,
        width=1,which='minor')
    # for refined control of log-scale tick marks
    locmaj = mpl.ticker.LogLocator(base=10,numticks=12) 
    locmin = mpl.ticker.LogLocator(base=10.0,
        subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),numticks=12)
    ax.yaxis.set_major_locator(locmaj)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    # plot
    ax.plot(r,rho_LoDens,lw=lw,color='k',label=r'SIDM (low-density)')
    ax.plot(r,rho_HiDens,lw=lw,color='grey',label=r'SIDM (high-density)')
    ax.plot(r_FullRange,halo.rho(r_FullRange),lw=lw,color='k',ls='--',label=r'CDM')
    # reference line
    ax.plot(np.repeat(r1,2),ax.get_ylim(),color='k',lw=0.5*lw,ls=':')
    ax.plot(np.repeat(rs,2),ax.get_ylim(),color='k',lw=0.5*lw,ls='-.')
    # annotation 
    ax.text(r1,2.*ax.get_ylim()[0],r'$r_1$',color='k',fontsize=16,
        ha='right',va='bottom',transform=ax.transData,rotation=90)
    ax.text(rs,2.*ax.get_ylim()[0],r'$r_\mathrm{s}$',color='k',fontsize=16,
        ha='left',va='bottom',transform=ax.transData,rotation=90)
    # legend
    ax.legend(loc='upper right',fontsize=14,frameon=True)
    
    #---
    ax = fig1.add_subplot(gs[0,1])
    ax.set_xlim(0.001,50.)
    ax.set_ylim(1.,300.)
    ax.set_xlabel(r'$r$ [kpc]')
    ax.set_ylabel(r'$V_\mathrm{circ}$ [kpc/Gyr]')
    #ax.set_title(r'')
    # scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    # tick and tick label positions
    start,end = ax.get_ylim()
    major_ticks = np.arange(start, end, 50.)
    minor_ticks = np.arange(start, end, 10.)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks,minor=True)
    # for refined control of log-scale tick marks
    locmaj = mpl.ticker.SymmetricalLogLocator(base=10,linthresh=0.1) 
    locmin = mpl.ticker.SymmetricalLogLocator(base=10.0,
        subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),linthresh=0.1)
    ax.yaxis.set_major_locator(locmaj)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    # grid
    ax.grid(which='minor', alpha=0.2)                                                
    ax.grid(which='major', alpha=0.4)
    # tick length
    ax.tick_params('both',direction='in',top='on',right='on',length=10,
        width=1,which='major')
    ax.tick_params('both',direction='in',top='on',right='on',length=5,
        width=1,which='minor')
    # plot
    ax.plot(r,Vc_LoDens,lw=lw,color='k')
    ax.plot(r,Vc_HiDens,lw=lw,color='grey')
    ax.plot(r_FullRange,halo.Vcirc(r_FullRange),lw=lw,color='k',ls='--',)
    # reference line
    ax.plot(np.repeat(r1,2),ax.get_ylim(),color='k',lw=0.5*lw,ls=':')
    ax.plot(np.repeat(rs,2),ax.get_ylim(),color='k',lw=0.5*lw,ls='-.')
    # annotations
    ax.text(r1,2.*ax.get_ylim()[0],r'$r_1$',color='k',fontsize=16,
        ha='right',va='bottom',transform=ax.transData,rotation=90)
    ax.text(rs,2.*ax.get_ylim()[0],r'$r_\mathrm{s}$',color='k',fontsize=16,
        ha='left',va='bottom',transform=ax.transData,rotation=90)
    # legends
    #ax.legend(loc='upper left',numpoints=1,scatterpoints=1,fontsize=16,
    #    frameon=True) #bbox_to_anchor=(1.2, 0.9)

    #---
    ax = fig1.add_subplot(gs[0,2])
    ax.set_xlim(1.,200.)
    ax.set_ylim(1e5,1e14) 
    ax.set_xlabel(r'$\sigma_0$ [kpc/Gyr]',fontsize=18)
    ax.set_ylabel(r'$\rho_0$ [$M_\odot$/kpc$^3$]',fontsize=18)
    #ax.set_title(r'')
    # scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    # grid
    ax.grid(which='minor', alpha=0.2)                                                
    ax.grid(which='major', alpha=0.4)
    # tick and tick label positions
    #start, end = ax.get_xlim()
    #major_ticks = np.arange(start, end, 0.5)
    #minor_ticks = np.arange(start, end, 0.1)
    #ax.set_xticks(major_ticks)
    #ax.set_xticks(minor_ticks,minor=True)
    #start, end = ax.get_ylim()
    #major_ticks = np.arange(start, end, 0.5)
    #minor_ticks = np.arange(start, end, 0.1)
    #ax.set_yticks(major_ticks)
    #ax.set_yticks(minor_ticks,minor=True)
    # for refined control of log-scale tick marks
    locmaj = mpl.ticker.LogLocator(base=10,numticks=12) 
    locmin = mpl.ticker.LogLocator(base=10.0,
        subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
        numticks=12)
    ax.xaxis.set_major_locator(locmaj)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    locmaj = mpl.ticker.LogLocator(base=10,numticks=12) 
    locmin = mpl.ticker.LogLocator(base=10.0,
        subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
        numticks=12)
    ax.yaxis.set_major_locator(locmaj)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    # tick length
    ax.tick_params('both',direction='in',top='on',right='on',length=10,
        width=1,which='major',zorder=301)
    ax.tick_params('both',direction='in',top='on',right='on',length=5,
        width=1,which='minor',zorder=301)
    # plot
    ax.scatter(sigma0_LoDens,rhodm0_LoDens,marker='s',s=size,
        facecolor='k',edgecolor='k',linewidth=edgewidth,rasterized=True)
    ax.scatter(sigma0_HiDens,rhodm0_HiDens,marker='s',s=size,
        facecolor='grey',edgecolor='grey',linewidth=edgewidth,rasterized=True)
    # annotations
    #ax.text(0.05,0.9,r'',
    #    color='k',fontsize=18,ha='left',va='bottom',
    #    transform=ax.transAxes,rotation=0)
    # legend
    #ax.legend(loc='best',fontsize=16,frameon=True)

    #---save figure
    plt.savefig(outfig1%(lgMv,c,sigmamx,tage),dpi=300)
    fig1.canvas.manager.window.raise_()
    plt.get_current_fig_manager().window.setGeometry(50,50,1600,500)
    #fig1.show()

    #if i==3: sys.exit() # <<< test

f_rhodm0.close()
f_density_LoDens.close()
f_density_HiDens.close()
f_radius.close()