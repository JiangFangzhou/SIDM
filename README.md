# SIDM
An analytical model for the density profiles of self-interacting dark-matter halos with inhabitant galaxies, affiliated with the paper of Jiang et al. (2022)

- Installation

`git clone https://github.com/JiangFangzhou/SIDM.git`

- Model overview

This model combines the isothermal Jeans model and the model of adiabatic halo contraction into a simple semi-analytic procedure for computing the density profile of self-interacting dark-matter (SIDM) haloes with the gravitational influence from the inhabitant galaxies. The model agrees well with cosmological SIDM simulations over the entire core-forming stage and up to the onset of gravothermal core-collapse. Using this model, we show that the halo response to baryons is more diverse in SIDM than in CDM and depends sensitively on galaxy size, a desirable link in the context of the structural diversity of bright dwarf galaxies. The fast speed of the method facilitates analyses that would be challenging for numerical simulations, as detailed in Jiang et al. (2022).

- Workflow of the model

<img src="https://render.githubusercontent.com/render/math?math=M">
<img src="https://render.githubusercontent.com/render/math?math=c">

(1) Given a CDM halo described by an NFW profile (i.e., with known virial mass <img src="https://render.githubusercontent.com/render/math?math=M_\mathrm{vir}">, concentration <img src="https://render.githubusercontent.com/render/math?math=c">, and age <img src="https://render.githubusercontent.com/render/math?math=t_\mathrm{age}">), and given an inhabitant galaxy described by a Hernquist profile (parameterized by the mass <img src="https://render.githubusercontent.com/render/math?math=M_\mathrm{b}"> and scale size <img src="https://render.githubusercontent.com/render/math?math=r_0">), compute the adiabatically contracted halo profile.

(2)  Given the self-interaction cross-section, <img src="https://render.githubusercontent.com/render/math?math=\sigma_m"> , solve for the radius of frequent scattering, <img src="https://render.githubusercontent.com/render/math?math=r_1"> , using the density profile and velocity-dispersion profile of the contracted CDM halo. 

(3)  Integrate the spherical Jeans-Poisson equation to obtain an isothermal core profile -- do this iteratively to find the central DM density <img src="https://render.githubusercontent.com/render/math?math=\rho_\mathrm{0}">  and the central velocity dispersion <img src="https://render.githubusercontent.com/render/math?math=v_\mathrm{0}">  by minimizing the relative stitching error at <img src="https://render.githubusercontent.com/render/math?math=r_1">.

- Modules

profiles.py: density-profile classes (NFW, coreNFW, Hernquist, Miyamoto-Nagai, Burkert, Einasto, Dekel-Zhao, etc.), as well as the isothermal Jeans model for SIDM halos of Kaplinghat et al. (2014, 2016)

cosmo.py: cosmology-related functions

config.py: global variables and user controls 

galhalo.py: galaxy-halo connections, including the adiabatic contraction calculations of Gnedin et al. (2004, 2011)

aux.py: auxiliary functions

- Dependent libraries and packages

numpy, scipy, cosmolopy, fast_histogram

We recommend using python installations from Enthought or Conda. 

- Usage example

`import config as cfg`

`import profiles as pr`

`import galhalo as gh`

Note that the first time importing config or profiles, it can take a few seconds as the cosmological module cosmo prepares a few interpolation tables. Subsequent usage should be very fast without noticeable delay. 

User inputs:

`tage = 10. # [Gyr] halo age, i.e., lookback time to the formation epoch of the halo`

`sigmamx = 0.5 # [cm^2/g] cross section per unit mass`

`Mv = 2e12 # [M_sun] virial mass`

`c = 9.7 # NFW concentration`

`Mb = 6.2e10 # [M_sun] galaxy mass`

`r0 = 3. # [kpc] galaxy Hernquist scale radius`


Define the target CDM halo to operate on:

`halo_init = pr.NFW(Mv,c,Delta=100.,z=0.)`

Define the inhabitant galaxy profile: 

`disk = pr.Hernquist(Mb,r0)`

Compute the contracted halo:

`r_grid = np.logspace(-3,3,200) # [kpc] radius grid for computing halo contraction`

`halo_contra = gh.contra(r_grid,halo_init,disk)[0] `


Compute the effective scattering radius <img src="https://render.githubusercontent.com/render/math?math=r_1">:

`r1 = pr.r1(halo_contra,sigmamx=sigmamx,tage=tage)`

Compute the SIDM profile:

`rhodm0,v0,rho,Vc,r = pr.stitchSIDMcore(r1,halo_contra,disk)`

where the rhodm0 is the central DM density, v0 the central velocity dispersion, rho and Vc the density and circular velocity profiles from the center to r1, and r the radii at which we register the profiles. 

- Example scripts:

`test_SolveSIDMprofile_GivenCDMandBaryon.py` -- a script containing the above example, and plot the SIDM profiles.

`test_IsothermalSolnTimeSeries.py` -- compute the solutions on halo age time series, emulating what would be achieved with a gravothermal fluid evolution

`compare_GCthreshold_DifferentCrossSectionAndConcentration.py` -- a plotting program that takes the outputs of `test_IsothermalSolnTimeSeries.py` as inputs, and plots the threshold of gravothermal-core-collapse in the space of galaxy mass fraction versus galaxy compactness <img src="https://render.githubusercontent.com/render/math?math=M_\mathrm{b}/M_\mathrm{vir}-r_\mathrm{1/2}/M_\mathrm{vir}">

The modules have detailed docstrings, and the example programs are designed 
to be self-explanatory. Please feel free to contact the author Fangzhou Jiang (fzjiang@caltech.edu, fjiang@carnegiescience.edu) for questions.
