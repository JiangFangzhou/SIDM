#########################################################################

# Program that plots the threshold line of gravothermal core-collapse 
# (GC) in the space of baryon-to-total ratio M_b/M_vir versus the 
# compactness of baryon distribution R_e/R_vir.
# This program uses the outputs of the program
# test_SolveSIDMprofile_ScanBarProperties.py

# Arthur Fangzhou Jiang 2022 Caltech and Carnegie

######################## set up the environment #########################

import config as cfg
import cosmo as co
import profiles as pr
import aux

import numpy as np
from scipy.interpolate import interp2d
import sys

import matplotlib as mpl # must import before pyplot
#mpl.use('TkAgg')         # use the 'TkAgg' backend for showing plots
mpl.use('Qt5Agg')
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.size'] = 17  
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

########################### user control ################################

delta_thres = 0.01 # the threshold of stitching error above which we consider a isotherm-Jeans doesn't exist and gravothermal core-collapse occurs

#---data

lgMv = np.log10(1e11) # [M_sun]
c = 20. # <<< variable: 5, 10, or 20
sigmamx = 1. # [cm^2/g] # <<< variable: 1. or 10.
tage = 10. # [Gyr]

file_rhodm0 = './OUTPUT/ScanBarProperties_rhodm0_lgM%.2f_c%.1f_sigmamx%.1f_tage%.1f.txt'#%(lgMv,c,sigmamx,tage)

#---plot control

lw=2
size=50
alpha=1
edgewidth = 1
outfig1 = './FIGURE/compare_GCthreshold_DifferentCrossSectionAndConcentration.pdf'

############################### compute #################################

#---load data
MbMv,ReRv,delta_min_c5sigmamx1 = np.genfromtxt(file_rhodm0%(lgMv,5.,1.,tage),usecols=(0,1,5,),unpack=True)
MbMv,ReRv,delta_min_c10sigmamx1 = np.genfromtxt(file_rhodm0%(lgMv,10.,1.,tage),usecols=(0,1,5,),unpack=True)
MbMv,ReRv,delta_min_c20sigmamx1 = np.genfromtxt(file_rhodm0%(lgMv,20.,1.,tage),usecols=(0,1,5,),unpack=True)
MbMv,ReRv,delta_min_c5sigmamx10 = np.genfromtxt(file_rhodm0%(lgMv,5.,10.,tage),usecols=(0,1,5,),unpack=True)
MbMv,ReRv,delta_min_c10sigmamx10 = np.genfromtxt(file_rhodm0%(lgMv,10.,10.,tage),usecols=(0,1,5,),unpack=True)
MbMv,ReRv,delta_min_c20sigmamx10 = np.genfromtxt(file_rhodm0%(lgMv,20.,10.,tage),usecols=(0,1,5,),unpack=True)

MbMv_grid = np.unique(MbMv)
ReRv_grid = np.unique(ReRv)

########################### diagnostic plots ############################

print('>>> plot ...')
plt.close('all') # close all previous figure windows

#------------------------------------------------------------------------

# set up the figure window
fig1 = plt.figure(figsize=(7,6), dpi=80, facecolor='w', edgecolor='k') 
fig1.subplots_adjust(left=0.13, right=0.96,bottom=0.1, top=0.97,
    hspace=0.3, wspace=0.45)
gs = gridspec.GridSpec(1, 1) 
#fig1.suptitle(r'$M_\mathrm{vir}=10^{%.2f}M_\odot$, $t_\mathrm{age}=%.1f$Gyr'%(lgMv,tage))

#--column 1 main panel
ax = fig1.add_subplot(gs[0,0])
ax.set_xlim(0.005,0.1)
ax.set_ylim(0.005,0.1)
ax.set_xlabel(r'$M_\mathrm{b}/M_\mathrm{vir}$')
ax.set_ylabel(r'$R_\mathrm{b,1/2}/R_\mathrm{vir}$')
#ax.set_title(r'')
# scale
ax.set_xscale('log')
ax.set_yscale('log')
# tick and tick label positions
#start,end = ax.get_ylim()
#major_ticks = np.arange(start, end, 0.5)
#minor_ticks = np.arange(start, end, 0.1)
#ax.set_yticks(major_ticks)
#ax.set_yticks(minor_ticks,minor=True)
# for refined control of log-scale tick marks
#locmaj = mpl.ticker.LogLocator(base=10,numticks=12) 
#locmin = mpl.ticker.LogLocator(base=10.0,
#    subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
#    numticks=12)
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
# plot threshold lines for gravothermal core-collapse
MbMv_fine = np.logspace(np.log10(MbMv_grid).min(),np.log10(MbMv_grid).max(),40)
ReMv_fine = np.logspace(np.log10(ReRv_grid).min(),np.log10(ReRv_grid).max(),40)
# 
f = interp2d(np.log10(MbMv_grid),np.log10(ReRv_grid),delta_min_c5sigmamx1,'linear')
delta_min_c5sigmamx1_fine = f(np.log10(MbMv_fine),np.log10(ReMv_fine))
C = plt.contour(MbMv_fine, ReMv_fine, delta_min_c5sigmamx1_fine,[delta_thres])
v = C.collections[0].get_paths()[0].vertices
MbMv_thres_c5sigmamx1 = v[:,0]
ReRv_thres_c5sigmamx1 = v[:,1]
ax.plot(MbMv_thres_c5sigmamx1,ReRv_thres_c5sigmamx1,lw=lw,c='lightblue',label=r'$\sigma_m=1\ $cm$^2$/g, $c=5$')
#
f = interp2d(np.log10(MbMv_grid),np.log10(ReRv_grid),delta_min_c10sigmamx1,'linear')
delta_min_c10sigmamx1_fine = f(np.log10(MbMv_fine),np.log10(ReMv_fine))
C = plt.contour(MbMv_fine, ReMv_fine, delta_min_c10sigmamx1_fine,[delta_thres])
v = C.collections[0].get_paths()[0].vertices
MbMv_thres_c10sigmamx1 = v[:,0]
ReRv_thres_c10sigmamx1 = v[:,1]
ax.plot(MbMv_thres_c10sigmamx1,ReRv_thres_c10sigmamx1,lw=lw,c='blue',label=r'$\sigma_m=1\ $cm$^2$/g, $c=10$')
#
f = interp2d(np.log10(MbMv_grid),np.log10(ReRv_grid),delta_min_c20sigmamx1,'linear')
delta_min_c20sigmamx1_fine = f(np.log10(MbMv_fine),np.log10(ReMv_fine))
C = plt.contour(MbMv_fine, ReMv_fine, delta_min_c20sigmamx1_fine,[delta_thres])
v = C.collections[0].get_paths()[0].vertices
MbMv_thres_c20sigmamx1 = v[:,0]
ReRv_thres_c20sigmamx1 = v[:,1]
ax.plot(MbMv_thres_c20sigmamx1,ReRv_thres_c20sigmamx1,lw=lw,c='darkblue',label=r'$\sigma_m=1\ $cm$^2$/g, $c=20$')
#
#
f = interp2d(np.log10(MbMv_grid),np.log10(ReRv_grid),delta_min_c5sigmamx10,'linear')
delta_min_c5sigmamx10_fine = f(np.log10(MbMv_fine),np.log10(ReMv_fine))
C = plt.contour(MbMv_fine, ReMv_fine, delta_min_c5sigmamx10_fine,[delta_thres])
v = C.collections[0].get_paths()[0].vertices
MbMv_thres_c5sigmamx10 = v[:,0]
ReRv_thres_c5sigmamx10 = v[:,1]
ax.plot(MbMv_thres_c5sigmamx10,ReRv_thres_c5sigmamx10,lw=lw,c='pink',label=r'$\sigma_m=10\ $cm$^2$/g, $c=5$')
#
f = interp2d(np.log10(MbMv_grid),np.log10(ReRv_grid),delta_min_c10sigmamx10,'linear')
delta_min_c10sigmamx10_fine = f(np.log10(MbMv_fine),np.log10(ReMv_fine))
C = plt.contour(MbMv_fine, ReMv_fine, delta_min_c10sigmamx10_fine,[delta_thres])
v = C.collections[0].get_paths()[0].vertices
MbMv_thres_c10sigmamx10 = v[:,0]
ReRv_thres_c10sigmamx10 = v[:,1]
ax.plot(MbMv_thres_c10sigmamx10,ReRv_thres_c10sigmamx10,lw=lw,c='red',label=r'$\sigma_m=10\ $cm$^2$/g, $c=10$')
#
f = interp2d(np.log10(MbMv_grid),np.log10(ReRv_grid),delta_min_c20sigmamx10,'linear')
delta_min_c20sigmamx10_fine = f(np.log10(MbMv_fine),np.log10(ReMv_fine))
C = plt.contour(MbMv_fine, ReMv_fine, delta_min_c20sigmamx10_fine,[delta_thres])
v = C.collections[0].get_paths()[0].vertices
MbMv_thres_c20sigmamx10 = v[:,0]
ReRv_thres_c20sigmamx10 = v[:,1]
ax.plot(MbMv_thres_c20sigmamx10,ReRv_thres_c20sigmamx10,lw=lw,c='darkred',label=r'$\sigma_m=10\ $cm$^2$/g, $c=20$')
#
#
# annotations
ax.text(0.7,0.2,r'$\downarrow$',
    color='k',fontsize=20,ha='left',va='bottom',transform=ax.transAxes,rotation=0,zorder=302)
ax.text(0.5,0.1,r'$\rightarrow$',
    color='k',fontsize=20,ha='left',va='bottom',transform=ax.transAxes,rotation=0,zorder=302)
ax.text(0.6,0.12,r'Gravothermal',
    color='k',fontsize=16,ha='left',va='bottom',transform=ax.transAxes,rotation=0,zorder=302)
ax.text(0.6,0.07,r'core-collapse',
    color='k',fontsize=16,ha='left',va='bottom',transform=ax.transAxes,rotation=0,zorder=302)
# legends
ax.legend(loc='upper left',numpoints=1,scatterpoints=1,fontsize=14,
    frameon=False) #bbox_to_anchor=(1.2, 0.9)

#---save figure
plt.savefig(outfig1,dpi=300)
fig1.canvas.manager.window.raise_()
plt.get_current_fig_manager().window.setGeometry(50,50,700,600)
fig1.show()