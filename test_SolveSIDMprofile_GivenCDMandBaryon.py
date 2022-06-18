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

# Arthur Fangzhou Jiang 2020 Caltech

######################## set up the environment #########################

#---user modules
import config as cfg
import profiles as pr
import galhalo as gh

#---standard python stuff
import numpy as np

#---for plot
import matplotlib as mpl # must import before pyplot
mpl.use('Qt5Agg')
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.size'] = 16  
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

########################### user control ################################

#---target CDM halo and baryon distribution

tage = 10. # [Gyr]
sigmamx = 0.5 # [cm^2/g]
lgMv = np.log10(2e12) # [M_sun]
c = 9.7
lgMb = np.log10(6.2e10) # [M_sun]
r0 = 3. # [kpc]
    
#---plot control
outfig1 = './FIGURE/test_SolveSIDMprofile_sigmamx%.1f_tage%.1f_lgMv%.2f_c%.1f_lgMb%.2f_rH%.2f.pdf'\
    %(sigmamx,tage,lgMv,c,lgMb,r0)
lw = 2.5
size = 30.
edgewidth = 0.
r_FullRange = np.logspace(-3,3,200) # [kpc] for plotting the full profile

############################### compute #################################

print('>>> computing SIDM profile ... ')

#---prepare the CDM profile to stitch to
# with baryons
Mv = 10.**lgMv
Mb = 10.**lgMb
halo_init = pr.NFW(Mv,c,Delta=100.,z=0.)
disk = pr.Hernquist(Mb,r0)
halo_contra = gh.contra(r_FullRange,halo_init,disk)[0] # <<< adiabatically contracted CDM halo

# dark-matter only
fb = Mb/Mv
disk_dmo = pr.Hernquist(0.001,100.) # <<< use a tiny mass and huge size for the DM-only case
halo_dmo = pr.NFW((1.-fb)*Mv,c,Delta=halo_init.Deltah,z=halo_init.z) # <<< DMO CDM halo

#---find r_1
r1 = pr.r1(halo_contra,sigmamx=sigmamx,tage=tage)
r1_dmo = pr.r1(halo_dmo,sigmamx=sigmamx,tage=tage)

#---with baryon
rhodm0,sigma0,rho,Vc,r = pr.stitchSIDMcore(r1,halo_contra,disk)

#---dark-matter only
rhodm0_dmo,sigma0_dmo,rho_dmo,Vc_dmo,r_dmo = pr.stitchSIDMcore(r1_dmo,halo_dmo,disk_dmo)

################################ plots ##################################

print('>>> plot ...')
plt.close('all') # close all previous figure windows

#------------------------------------------------------------------------

# set up the figure window
fig1 = plt.figure(figsize=(16,5), dpi=80, facecolor='w', edgecolor='k') 
fig1.subplots_adjust(left=0.06, right=0.93,bottom=0.12, top=0.91,
    hspace=0.25, wspace=0.25)
gs = gridspec.GridSpec(1, 3) 
fig1.suptitle(r'$M_\mathrm{v}=10^{%.1f}M_\odot, c=%.1f, t_\mathrm{age}=%.1f\mathrm{Gyr}, \sigma/m_\chi=%.1f\mathrm{cm}^2/\mathrm{g}, M_\mathrm{b}=10^{%.2f}M_\odot, r_\mathrm{0}=%.2f$kpc'\
    %(lgMv,c,tage,sigmamx,lgMb,r0),fontsize=14)

#---
ax = fig1.add_subplot(gs[0,0])
ax.set_xlim(0.01,30.)
ax.set_ylim(10.**5,10.**11)
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
    subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
    numticks=12)
ax.yaxis.set_major_locator(locmaj)
ax.yaxis.set_minor_locator(locmin)
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
# plot
ax.plot(r,rho,lw=lw,color='b',label=r'SIDM')
ax.plot(r_dmo,rho_dmo,lw=lw,color='k',label=r'SIDM DMO')
ax.plot(r_FullRange,halo_contra.rho(r_FullRange),lw=lw,color='b',ls='--')
ax.plot(r_FullRange,halo_dmo.rho(r_FullRange),lw=lw,color='k',ls='--')
# reference line
ax.plot(np.repeat(r1,2),ax.get_ylim(),color='k',lw=0.5*lw,ls=':')
# annotation 
ax.text(0.2,ax.get_ylim()[0]*0.8,r'0.2',
    color='k',fontsize=14,ha='center',va='top',
    transform=ax.transData,rotation=0)
ax.text(0.5,ax.get_ylim()[0]*0.8,r'0.5',
    color='k',fontsize=14,ha='center',va='top',
    transform=ax.transData,rotation=0)
ax.text(2,ax.get_ylim()[0]*0.8,r'2',
    color='k',fontsize=14,ha='center',va='top',
    transform=ax.transData,rotation=0)
ax.text(5,ax.get_ylim()[0]*0.8,r'5',
    color='k',fontsize=14,ha='center',va='top',
    transform=ax.transData,rotation=0)
# legend
ax.legend(loc='upper right',fontsize=14,frameon=True)

#---
ax = fig1.add_subplot(gs[0,1])
ax.set_xlim(0.01,30.)
ax.set_ylim(0.,250.)
ax.set_xlabel(r'$r$ [kpc]')
ax.set_ylabel(r'$V_\mathrm{circ}$ [kpc/Gyr]')
#ax.set_title(r'')
# scale
ax.set_xscale('log')
#ax.set_yscale('log')
# tick and tick label positions
start,end = ax.get_ylim()
major_ticks = np.arange(start, end, 50.)
minor_ticks = np.arange(start, end, 10.)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks,minor=True)
# for refined control of log-scale tick marks
#locmaj = mpl.ticker.SymmetricalLogLocator(base=10,linthresh=0.1) 
#locmin = mpl.ticker.SymmetricalLogLocator(base=10.0,
#    subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),linthresh=0.1)
#ax.yaxis.set_major_locator(locmaj)
#ax.yaxis.set_minor_locator(locmin)
#ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
# grid
ax.grid(which='minor', alpha=0.2)                                                
ax.grid(which='major', alpha=0.4)
# tick length
ax.tick_params('both',direction='in',top='on',right='on',length=10,
    width=1,which='major')
ax.tick_params('both',direction='in',top='on',right='on',length=5,
    width=1,which='minor')
# plot
ax.plot(r,Vc,lw=lw,color='b')
ax.plot(r_dmo,Vc_dmo,lw=lw,color='k')
ax.plot(r_FullRange,halo_contra.Vcirc(r_FullRange),lw=lw,color='b',ls='--',)
ax.plot(r_FullRange,halo_dmo.Vcirc(r_FullRange),lw=lw,color='k',ls='--',)
# reference line
ax.plot(np.repeat(r1,2),ax.get_ylim(),color='b',lw=0.5*lw,ls=':')
ax.plot(np.repeat(r1_dmo,2),ax.get_ylim(),color='k',lw=0.5*lw,ls=':')
# annotations
#ax.text(0.1,0.1,r'',color='g',fontsize=16,
#    ha='left',va='center',transform=ax.transAxes,rotation=0)
# legends
#ax.legend(loc='upper left',numpoints=1,scatterpoints=1,fontsize=16,
#    frameon=True) #bbox_to_anchor=(1.2, 0.9)

#---

ax = fig1.add_subplot(gs[0,2])
ax.set_xlim(5.,500.)
ax.set_ylim(10.**7,10.**11) 
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
#locmaj = mpl.ticker.LogLocator(base=10,numticks=12) 
#locmin = mpl.ticker.LogLocator(base=10.0,
#    subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
#    numticks=12)
#ax.xaxis.set_major_locator(locmaj)
#ax.xaxis.set_minor_locator(locmin)
#ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
# tick length
ax.tick_params('both',direction='in',top='on',right='on',length=10,
    width=1,which='major',zorder=301)
ax.tick_params('both',direction='in',top='on',right='on',length=5,
    width=1,which='minor',zorder=301)
# plot
ax.scatter(sigma0,rhodm0,marker='s',s=size,
    facecolor='b',edgecolor='k',linewidth=edgewidth,rasterized=True)
ax.scatter(sigma0_dmo,rhodm0_dmo,marker='s',s=size,
    facecolor='k',edgecolor='k',linewidth=edgewidth,rasterized=True)
# annotations
#ax.text(0.05,0.9,r'',
#    color='k',fontsize=18,ha='left',va='bottom',
#    transform=ax.transAxes,rotation=0)
# legend
#ax.legend(loc='best',fontsize=16,frameon=True)

#---save figure
plt.savefig(outfig1,dpi=300)
fig1.canvas.manager.window.raise_()
plt.get_current_fig_manager().window.setGeometry(50,50,1600,500)
fig1.show()
