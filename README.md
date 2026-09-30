# Dark Matter from a MaNGA Rotation Curve

Educational computational-astrophysics project using real **MaNGA 7443-6103**
integral-field spectroscopy.

## Goal

The project follows the chain

```text
MaNGA spectral cube
        ↓
H-alpha wavelength field
        ↓
line-of-sight velocity field
        ↓
inclination-corrected rotation curve
        ↓
stellar-light profile
        ↓
stellar gravitational contribution
        ↓
dynamical discrepancy
        ↓
NFW halo + Bayesian MCMC
```

## Main result

The independently estimated stellar mass is too small for the stellar disk to
reproduce the measured rotation curve by itself. An NFW halo can provide the
additional gravitational contribution required by the observed velocities.

The MCMC fit found approximately:

```text
log10(rho_s) = 6.17 +0.13 -0.07
log10(r_s)   = 1.90 +0.07 -0.13
rho_s ~ 1.5e6 Msun/kpc^3
r_s   ~ 80 kpc
```

The scale radius lies close to the upper prior boundary, so `rho_s` and `r_s`
should **not** be interpreted as precise independent measurements. The MaNGA
data only reach roughly 4 kpc, producing a strong halo-parameter degeneracy.

## Repository structure

```text
manga_dark_matter_project/
├── dark_matter_manga.py
├── Materia_oscura_clean.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── figures/
```

## Main sections of the code

1. Download and load the MaNGA LOGCUBE.
2. Measure photometric centre, major axis and inclination.
3. Build an H-alpha wavelength map.
4. Convert H-alpha shifts into a 2D LOS velocity field.
5. Extract and bin the rotation curve.
6. Fit the stellar-light profile with an exponential disk.
7. Calculate the stellar contribution using an independent stellar mass.
8. Measure the dynamical component not explained by stars.
9. Add an NFW dark-matter halo.
10. Define prior, likelihood and posterior.
11. Explore the posterior with `emcee` MCMC.
12. Plot stars, halo and total model against the observations.

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python dark_matter_manga.py
```

The script downloads the MaNGA cube on the first run and creates the final
figures automatically.

## Bayesian/MCMC logic

The parameters inferred are

```text
theta = [log10(rho_s), log10(r_s)]
```

The prior defines the allowed halo parameter range.

The likelihood compares model and observed velocities through

```text
log L = -chi^2 / 2
```

The posterior is

```text
log posterior = log prior + log likelihood
```

`emcee` uses several walkers to sample this posterior. After discarding the
initial burn-in, the retained positions of the walkers are samples from the
posterior probability distribution.

## Important limitations

This is not a precision dark-matter measurement.

- The gas contribution is not modelled.
- The stellar mass carries systematic uncertainty.
- H-alpha velocities are extracted using a simplified centroid method.
- The photometric geometry is estimated directly from the reconstructed image.
- Only spaxels near the major axis are used.
- The field of view reaches only about 4 kpc.
- The inferred NFW parameters are therefore strongly degenerate.
- A more complete analysis would use the full 2D velocity field, gas mass,
  stellar-population modelling and a larger radial baseline.

## Figures

The `figures/` directory contains the intermediate and final figures generated
during the analysis, including the H-alpha map, LOS velocity field, rotation
curve, stellar-light profile, dynamical discrepancy and final stars+NFW model.
