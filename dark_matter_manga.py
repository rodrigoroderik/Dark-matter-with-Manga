"""
MaNGA dark-matter project
=========================

Galaxy: MaNGA 7443-6103

Goal
----
1. Download a real MaNGA spectral cube.
2. Measure the H-alpha Doppler field.
3. Derive an approximate rotation curve.
4. Measure the exponential scale length of the stellar light.
5. Compare the stellar gravitational contribution with the observed rotation.
6. Fit an NFW halo using Bayesian inference + MCMC.

Important limitations
---------------------
This is an educational analysis, not a precision dark-matter measurement.
The gas mass is not modelled, the velocity field is extracted with a simplified
H-alpha centroid method, and the data only extend to about 4 kpc.
"""

# ============================================================
# 0. IMPORTS AND CONSTANTS
# ============================================================

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astroquery.mast import Observations
from astropy.io import fits
from astropy.constants import c, G
import astropy.units as u
from astropy.cosmology import Planck18

from scipy.optimize import curve_fit
from scipy.special import iv, kv
import emcee


PLATEIFU = "7443-6103"

# H-alpha rest wavelength in Angstrom
LAMBDA_HA_REST = 6562.8

# Approximate MaNGA spaxel size
SPAXEL_SCALE_ARCSEC = 0.5

# Intrinsic disk thickness used to estimate inclination
Q0 = 0.2

# Independent photometric stellar-mass estimate (Msun)
M_STAR = 1.9841247e9

# MCMC configuration
N_WALKERS = 32
N_DIM = 2
N_STEPS = 5000
BURN_IN = 1000
THIN = 10
RANDOM_SEED = 42

DATA_DIR = Path("data")
FIG_DIR = Path("figures")
DATA_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. DOWNLOAD AND LOAD THE MaNGA CUBE
# ============================================================

def download_manga_cube(plateifu=PLATEIFU):
    """
    Download the MaNGA LOGCUBE if it is not already stored locally.

    The LOGCUBE contains:
      FLUX[lambda, y, x] : spectrum at every spaxel
      IVAR                : inverse variance
      MASK                : quality mask
      WAVE                : wavelength axis
      RIMG                : reconstructed r-band image
    """

    local_path = DATA_DIR / f"manga-{plateifu}-LOGCUBE.fits.gz"

    if local_path.exists():
        return local_path

    obs = Observations.query_criteria(
        obs_collection="SDSS",
        provenance_name="MaNGA",
        obs_id=f"sdss_manga_{plateifu}",
    )

    products = Observations.get_product_list(obs)

    cube_name = f"manga-{plateifu}-LOGCUBE.fits.gz"

    cube_product = products[
        products["productFilename"] == cube_name
    ]

    manifest = Observations.download_products(
        cube_product,
        download_dir=str(DATA_DIR),
    )

    downloaded = Path(manifest["Local Path"][0])

    # Copy the downloaded file to a simpler reproducible path.
    local_path.write_bytes(downloaded.read_bytes())

    return local_path


def load_cube(cube_path):
    """Read the extensions needed in this project."""

    hdul = fits.open(cube_path)

    flux = hdul["FLUX"].data
    wave = hdul["WAVE"].data
    ivar = hdul["IVAR"].data
    mask = hdul["MASK"].data
    rimg = hdul["RIMG"].data

    return hdul, flux, wave, ivar, mask, rimg


# ============================================================
# 2. PHOTOMETRIC GEOMETRY: CENTRE, MAJOR AXIS, INCLINATION
# ============================================================

def photometric_geometry(rimg):
    """
    Estimate the centre and orientation from the r-band image.

    The centre is a flux-weighted mean position.

    The second-moment matrix measures how extended the light is in different
    directions. Its eigenvector with the largest eigenvalue defines the major
    axis; the eigenvalue ratio gives an approximate axis ratio b/a.
    """

    ny, nx = rimg.shape
    y, x = np.indices(rimg.shape)

    image = np.asarray(rimg, dtype=float)

    valid = np.isfinite(image) & (image > 0)
    weights = np.where(valid, image, 0.0)

    # Flux-weighted centre.
    flux_x = np.sum(weights, axis=0)
    flux_y = np.sum(weights, axis=1)

    xc = np.average(np.arange(nx), weights=flux_x)
    yc = np.average(np.arange(ny), weights=flux_y)

    dx = x - xc
    dy = y - yc

    total_flux = np.sum(weights)

    # Second moments.
    Mxx = np.sum(weights * dx**2) / total_flux
    Myy = np.sum(weights * dy**2) / total_flux
    Mxy = np.sum(weights * dx * dy) / total_flux

    moment_matrix = np.array([
        [Mxx, Mxy],
        [Mxy, Myy],
    ])

    # eigh returns eigenvalues from smallest to largest.
    eigenvalues, eigenvectors = np.linalg.eigh(moment_matrix)

    minor_vector = eigenvectors[:, 0]
    major_vector = eigenvectors[:, 1]

    lambda_minor = eigenvalues[0]
    lambda_major = eigenvalues[1]

    q = np.sqrt(lambda_minor / lambda_major)

    # Inclination correction for a disk with intrinsic thickness q0.
    cos2_i = (q**2 - Q0**2) / (1 - Q0**2)
    cos2_i = np.clip(cos2_i, 0, 1)

    inclination = np.degrees(
        np.arccos(np.sqrt(cos2_i))
    )

    pa_deg = np.degrees(
        np.arctan2(
            major_vector[1],
            major_vector[0],
        )
    )

    return (
        xc,
        yc,
        major_vector,
        q,
        inclination,
        pa_deg,
        x,
        y,
    )


# ============================================================
# 3. H-ALPHA DOPPLER MAP
# ============================================================

def halpha_centroid_map(wave, flux, rimg):
    """
    Estimate the H-alpha wavelength at every useful spaxel.

    We use a flux-weighted wavelength centroid inside the H-alpha window:
        lambda_Ha = sum(lambda * F_line) / sum(F_line)

    This is simpler than fitting a Gaussian at every spaxel.
    """

    ha_region = (
        (wave > 6677)
        & (wave < 6693)
    )

    wave_ha = wave[ha_region]
    flux_ha = flux[ha_region, :, :]

    # Approximate continuum from both ends of the wavelength window.
    continuum = (
        np.mean(flux_ha[:2], axis=0)
        + np.mean(flux_ha[-2:], axis=0)
    ) / 2

    line_flux = flux_ha - continuum

    # Negative residuals are noise, not H-alpha emission.
    line_flux[line_flux < 0] = 0

    numerator = np.sum(
        wave_ha[:, None, None] * line_flux,
        axis=0,
    )

    denominator = np.sum(
        line_flux,
        axis=0,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        lambda_map = numerator / denominator

    # Keep only the reasonably bright part of the galaxy.
    good = rimg > 0.15 * np.nanmax(rimg)

    lambda_map[~good] = np.nan

    return lambda_map, good


def wavelength_to_vlos(lambda_map):
    """
    Convert local H-alpha wavelength shifts into line-of-sight velocity.

    The median H-alpha wavelength is used as an approximate systemic
    wavelength, so the map measures internal motion relative to the galaxy.
    """

    lambda_sys = np.nanmedian(lambda_map)

    c_kms = c.to_value(u.km / u.s)

    vlos_map = (
        c_kms
        * (lambda_map - lambda_sys)
        / lambda_sys
    )

    z_gal = (
        lambda_sys / LAMBDA_HA_REST
        - 1
    )

    return lambda_sys, z_gal, vlos_map


# ============================================================
# 4. ROTATION CURVE ALONG THE MAJOR AXIS
# ============================================================

def rotation_curve_from_map(
    vlos_map,
    good,
    x,
    y,
    xc,
    yc,
    major_vector,
    inclination,
    z_gal,
):
    """
    Select spaxels close to the photometric major axis.

    Along the major axis:
        v_LOS ~= v_rot * sin(i)

    so we correct by sin(i). We then fold both galaxy sides using |R| and |v|.
    """

    vx = major_vector[0]
    vy = major_vector[1]

    dx = x - xc
    dy = y - yc

    # Signed position projected along the major axis.
    R_axis = (
        dx * vx
        + dy * vy
    )

    # Perpendicular distance from the major axis.
    distance_to_axis = (
        -dx * vy
        + dy * vx
    )

    # Use a narrow strip around the major axis.
    near_axis = (
        good
        & (np.abs(distance_to_axis) < 2)
    )

    i_rad = np.radians(inclination)

    vrot_signed = (
        vlos_map[near_axis]
        / np.sin(i_rad)
    )

    R_spaxel = R_axis[near_axis]

    # Angular scale -> physical scale.
    kpc_per_arcsec = (
        Planck18
        .kpc_proper_per_arcmin(z_gal)
        .to(u.kpc / u.arcsec)
        .value
    )

    R_kpc = np.abs(
        R_spaxel
        * SPAXEL_SCALE_ARCSEC
        * kpc_per_arcsec
    )

    vrot = np.abs(vrot_signed)

    order = np.argsort(R_kpc)

    return (
        R_kpc[order],
        vrot[order],
        R_axis,
        distance_to_axis,
        near_axis,
        kpc_per_arcsec,
    )


def bin_rotation_curve(R_kpc, vrot):
    """
    Average both galaxy sides in radial bins.

    v_err is the standard error of the mean velocity in each bin.
    """

    bins = np.arange(0, 4.3, 0.3)

    R_bin = []
    v_bin = []
    v_err = []

    for r1, r2 in zip(bins[:-1], bins[1:]):

        selected = (
            (R_kpc >= r1)
            & (R_kpc < r2)
        )

        if np.sum(selected) >= 3:

            velocities = vrot[selected]

            R_bin.append((r1 + r2) / 2)

            v_bin.append(
                np.mean(velocities)
            )

            v_err.append(
                np.std(velocities, ddof=1)
                / np.sqrt(len(velocities))
            )

    return (
        np.asarray(R_bin),
        np.asarray(v_bin),
        np.asarray(v_err),
    )


# ============================================================
# 5. STELLAR-LIGHT PROFILE AND EXPONENTIAL SCALE LENGTH
# ============================================================

def exponential_disk_light(R, I0, h):
    """Exponential surface-brightness profile I(R)=I0 exp(-R/h)."""

    return I0 * np.exp(-R / h)


def fit_light_profile(
    rimg,
    R_axis,
    distance_to_axis,
    kpc_per_arcsec,
):
    """
    Measure the light profile close to the major axis and fit its scale length h.

    h determines how the stellar disk is distributed radially.
    It does NOT determine the total stellar mass.
    """

    near_axis_img = (
        np.abs(distance_to_axis) < 2
    )

    R_light = (
        np.abs(R_axis[near_axis_img])
        * SPAXEL_SCALE_ARCSEC
        * kpc_per_arcsec
    )

    I_light = rimg[near_axis_img]

    disk_region = (
        (R_light > 0.5)
        & (I_light > 0)
    )

    popt, pcov = curve_fit(
        exponential_disk_light,
        R_light[disk_region],
        I_light[disk_region],
        p0=[
            np.nanmax(I_light),
            1.5,
        ],
    )

    I0_fit, h_fit = popt

    return (
        R_light,
        I_light,
        I0_fit,
        h_fit,
    )


# ============================================================
# 6. GRAVITATIONAL MODEL OF THE STELLAR DISK
# ============================================================

def v_disk(R, M_disk, h):
    """
    Circular speed of an infinitely thin exponential disk.

    Inputs:
      R       : kpc
      M_disk  : Msun
      h       : kpc

    Output:
      v_disk  : km/s

    The modified Bessel functions arise from solving the gravitational
    potential of an exponential disk.
    """

    y = R / (2 * h)

    Sigma0 = (
        M_disk
        / (2 * np.pi * h**2)
    )

    term = (
        iv(0, y) * kv(0, y)
        - iv(1, y) * kv(1, y)
    )

    G_ast = G.to_value(
        u.kpc
        * (u.km / u.s)**2
        / u.M_sun
    )

    v2 = (
        4
        * np.pi
        * G_ast
        * Sigma0
        * h
        * y**2
        * term
    )

    return np.sqrt(v2)


# ============================================================
# 7. DYNAMICAL DISCREPANCY
# ============================================================

def unexplained_velocity(v_observed, v_stars):
    """
    Velocity contribution not explained by the stellar disk alone.

    Since gravitational contributions add as v^2:
        v_obs^2 = v_stars^2 + v_extra^2
    """

    return np.sqrt(
        np.maximum(
            v_observed**2 - v_stars**2,
            0,
        )
    )


# ============================================================
# 8. NFW DARK-MATTER HALO
# ============================================================

def v_nfw(R, rho_s, r_s):
    """
    NFW halo circular speed.

    rho_s : characteristic density in Msun/kpc^3
    r_s   : scale radius in kpc
    """

    x = R / r_s

    enclosed_mass = (
        4
        * np.pi
        * rho_s
        * r_s**3
        * (
            np.log(1 + x)
            - x / (1 + x)
        )
    )

    G_ast = G.to_value(
        u.kpc
        * (u.km / u.s)**2
        / u.M_sun
    )

    return np.sqrt(
        G_ast * enclosed_mass / R
    )


def model_velocity(R, rho_s, r_s, h_fit):
    """
    Total model:
        v_total^2 = v_stars^2 + v_halo^2

    The stellar mass is fixed by the independent photometric estimate.
    """

    v_stars = v_disk(
        R,
        M_STAR,
        h_fit,
    )

    v_halo = v_nfw(
        R,
        rho_s,
        r_s,
    )

    return np.sqrt(
        v_stars**2
        + v_halo**2
    )


# ============================================================
# 9. BAYESIAN MODEL: PRIOR + LIKELIHOOD + POSTERIOR
# ============================================================

def build_log_probability(R_bin, v_bin, v_err, h_fit):
    """
    Build the posterior function used by emcee.

    We sample logarithmic halo parameters:
        theta = [log10(rho_s), log10(r_s)]

    Why logarithms?
    rho_s and r_s can span orders of magnitude. Log-space makes the MCMC
    exploration numerically easier.
    """

    def model_from_theta(theta):
        log_rho, log_rs = theta

        rho_s = 10**log_rho
        r_s = 10**log_rs

        return model_velocity(
            R_bin,
            rho_s,
            r_s,
            h_fit,
        )

    def log_prior(theta):
        """
        Prior = allowed parameter region before considering the velocity data.

        rho_s: 10^5 -- 10^10 Msun/kpc^3
        r_s  : 0.5 -- 100 kpc
        """

        log_rho, log_rs = theta

        if (
            5 < log_rho < 10
            and np.log10(0.5) < log_rs < 2
        ):
            return 0.0

        return -np.inf

    def log_likelihood(theta):
        """
        Gaussian likelihood.

        Smaller chi^2 means the model reproduces the observed rotation curve
        better, so log(L) = -chi^2/2 is larger.
        """

        model = model_from_theta(theta)

        chi2 = np.sum(
            ((v_bin - model) / v_err)**2
        )

        return -0.5 * chi2

    def log_probability(theta):
        """
        Posterior in logarithmic form:
            log posterior = log prior + log likelihood
        """

        lp = log_prior(theta)

        if not np.isfinite(lp):
            return -np.inf

        return (
            lp
            + log_likelihood(theta)
        )

    return log_probability


# ============================================================
# 10. MCMC SAMPLING
# ============================================================

def run_mcmc(log_probability):
    """
    Explore the posterior with emcee.

    A walker is one chain exploring parameter space.
    We use many walkers so the ensemble explores the posterior efficiently.
    """

    rng = np.random.default_rng(RANDOM_SEED)

    initial = np.array([
        7.0,  # log10(rho_s)
        1.0,  # log10(r_s)
    ])

    positions = (
        initial
        + 0.01
        * rng.normal(
            size=(N_WALKERS, N_DIM)
        )
    )

    sampler = emcee.EnsembleSampler(
        N_WALKERS,
        N_DIM,
        log_probability,
    )

    sampler.run_mcmc(
        positions,
        N_STEPS,
        progress=True,
    )

    # Remove the initial burn-in and keep one sample every THIN steps.
    samples = sampler.get_chain(
        discard=BURN_IN,
        thin=THIN,
        flat=True,
    )

    return sampler, samples


def posterior_summary(samples):
    """
    Return P16, P50 and P84 for both logarithmic halo parameters.
    """

    rho_percentiles = np.percentile(
        samples[:, 0],
        [16, 50, 84],
    )

    rs_percentiles = np.percentile(
        samples[:, 1],
        [16, 50, 84],
    )

    rho_best = 10**rho_percentiles[1]
    rs_best = 10**rs_percentiles[1]

    return (
        rho_percentiles,
        rs_percentiles,
        rho_best,
        rs_best,
    )


# ============================================================
# 11. MAIN ANALYSIS
# ============================================================

def main():

    # --------------------------------------------------------
    # A. Data
    # --------------------------------------------------------

    cube_path = download_manga_cube()

    (
        hdul,
        flux,
        wave,
        ivar,
        mask,
        rimg,
    ) = load_cube(cube_path)

    print("FLUX shape:", flux.shape)
    print("WAVE shape:", wave.shape)
    print("RIMG shape:", rimg.shape)

    # --------------------------------------------------------
    # B. Geometry
    # --------------------------------------------------------

    (
        xc,
        yc,
        major_vector,
        q,
        inclination,
        pa_deg,
        x,
        y,
    ) = photometric_geometry(rimg)

    print(f"Photometric centre = ({xc:.2f}, {yc:.2f})")
    print(f"Image major-axis angle = {pa_deg:.2f} deg")
    print(f"b/a = {q:.3f}")
    print(f"Inclination = {inclination:.2f} deg")

    # Geometry figure
    length = 25
    vx, vy = major_vector

    x_line = np.array([
        xc - length * vx,
        xc + length * vx,
    ])

    y_line = np.array([
        yc - length * vy,
        yc + length * vy,
    ])

    plt.figure(figsize=(7, 6))
    plt.imshow(rimg, origin="lower", cmap="gray")
    plt.scatter(xc, yc, marker="+", s=150, label="Centre")
    plt.plot(x_line, y_line, label="Major axis")
    plt.xlabel("x (spaxel)")
    plt.ylabel("y (spaxel)")
    plt.title(f"Photometric geometry - MaNGA {PLATEIFU}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "photometric_geometry.png", dpi=250)
    plt.close()

    # --------------------------------------------------------
    # C. H-alpha wavelength map and LOS velocity field
    # --------------------------------------------------------

    lambda_map, good = halpha_centroid_map(
        wave,
        flux,
        rimg,
    )

    (
        lambda_sys,
        z_gal,
        vlos_map,
    ) = wavelength_to_vlos(lambda_map)

    print(f"Systemic H-alpha wavelength = {lambda_sys:.3f} A")
    print(f"Approximate redshift = {z_gal:.6f}")

    plt.figure(figsize=(7, 6))
    plt.imshow(
        lambda_map,
        origin="lower",
        cmap="coolwarm",
    )
    plt.colorbar(label=r"$\lambda_{H\alpha}$ ($\AA$)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("H-alpha wavelength map")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "halpha_wavelength_map.png", dpi=250)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.imshow(
        vlos_map,
        origin="lower",
        cmap="coolwarm",
    )
    plt.colorbar(label="v_LOS (km/s)")
    plt.scatter(xc, yc, marker="+")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("H-alpha line-of-sight velocity field")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "vlos_map.png", dpi=250)
    plt.close()

    # --------------------------------------------------------
    # D. Rotation curve
    # --------------------------------------------------------

    (
        R_kpc,
        vrot,
        R_axis,
        distance_to_axis,
        near_axis,
        kpc_per_arcsec,
    ) = rotation_curve_from_map(
        vlos_map,
        good,
        x,
        y,
        xc,
        yc,
        major_vector,
        inclination,
        z_gal,
    )

    (
        R_bin,
        v_bin,
        v_err,
    ) = bin_rotation_curve(
        R_kpc,
        vrot,
    )

    print(
        f"Physical scale = "
        f"{kpc_per_arcsec:.3f} kpc/arcsec"
    )

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        R_bin,
        v_bin,
        yerr=v_err,
        fmt="o",
        capsize=3,
    )
    plt.xlabel("R (kpc)")
    plt.ylabel("v_rot (km/s)")
    plt.title("Binned rotation curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rotation_curve.png", dpi=250)
    plt.close()

    # --------------------------------------------------------
    # E. Stellar-light profile
    # --------------------------------------------------------

    (
        R_light,
        I_light,
        I0_fit,
        h_fit,
    ) = fit_light_profile(
        rimg,
        R_axis,
        distance_to_axis,
        kpc_per_arcsec,
    )

    print(f"Exponential scale length h = {h_fit:.3f} kpc")

    R_model = np.linspace(
        0,
        np.nanmax(R_light),
        300,
    )

    plt.figure(figsize=(8, 5))
    plt.scatter(
        R_light,
        I_light,
        s=10,
        label="RIMG",
    )
    plt.plot(
        R_model,
        exponential_disk_light(
            R_model,
            I0_fit,
            h_fit,
        ),
        label="Exponential disk",
    )
    plt.xlabel("R (kpc)")
    plt.ylabel("Brightness")
    plt.title("Stellar-light profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "stellar_light_profile.png", dpi=250)
    plt.close()

    # --------------------------------------------------------
    # F. Stars vs observed dynamics
    # --------------------------------------------------------

    v_stars = v_disk(
        R_bin,
        M_STAR,
        h_fit,
    )

    v_extra = unexplained_velocity(
        v_bin,
        v_stars,
    )

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        R_bin,
        v_bin,
        yerr=v_err,
        fmt="o",
        capsize=3,
        label="Observed",
    )
    plt.plot(
        R_bin,
        v_stars,
        label="Stars",
    )
    plt.xlabel("R (kpc)")
    plt.ylabel("v (km/s)")
    plt.title("Observed rotation vs stellar contribution")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "stars_vs_observed.png", dpi=250)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        R_bin,
        v_extra,
        "o-",
    )
    plt.xlabel("R (kpc)")
    plt.ylabel("v_extra (km/s)")
    plt.title("Dynamical component not explained by stars")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "unexplained_component.png", dpi=250)
    plt.close()

    # --------------------------------------------------------
    # G. Bayes + MCMC halo inference
    # --------------------------------------------------------

    log_probability = build_log_probability(
        R_bin,
        v_bin,
        v_err,
        h_fit,
    )

    sampler, samples = run_mcmc(
        log_probability
    )

    (
        rho_percentiles,
        rs_percentiles,
        rho_best,
        rs_best,
    ) = posterior_summary(samples)

    print("\nPosterior summary")
    print("-----------------")

    print(
        "log10(rho_s) = "
        f"{rho_percentiles[1]:.3f} "
        f"+{rho_percentiles[2] - rho_percentiles[1]:.3f} "
        f"-{rho_percentiles[1] - rho_percentiles[0]:.3f}"
    )

    print(
        "log10(r_s) = "
        f"{rs_percentiles[1]:.3f} "
        f"+{rs_percentiles[2] - rs_percentiles[1]:.3f} "
        f"-{rs_percentiles[1] - rs_percentiles[0]:.3f}"
    )

    print(
        f"rho_s median = {rho_best:.3e} Msun/kpc^3"
    )

    print(
        f"r_s median = {rs_best:.3f} kpc"
    )

    # --------------------------------------------------------
    # H. Final physical model
    # --------------------------------------------------------

    v_star_best = v_disk(
        R_bin,
        M_STAR,
        h_fit,
    )

    v_halo_best = v_nfw(
        R_bin,
        rho_best,
        rs_best,
    )

    v_total_best = np.sqrt(
        v_star_best**2
        + v_halo_best**2
    )

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        R_bin,
        v_bin,
        yerr=v_err,
        fmt="o",
        capsize=3,
        label="Observed",
    )
    plt.plot(
        R_bin,
        v_star_best,
        label="Stars",
    )
    plt.plot(
        R_bin,
        v_halo_best,
        label="NFW halo",
    )
    plt.plot(
        R_bin,
        v_total_best,
        label="Stars + halo",
    )
    plt.xlabel("R (kpc)")
    plt.ylabel("v (km/s)")
    plt.title("Final dynamical model")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "final_stars_nfw_mcmc.png", dpi=250)
    plt.close()

    hdul.close()


if __name__ == "__main__":
    main()
