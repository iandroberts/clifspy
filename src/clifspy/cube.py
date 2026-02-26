from clifspy import utils
from clifspy import astrometry

import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np

from astropy.wcs import WCS
from astropy import units
from astropy import coordinates
from astropy import convolution
from astropy import stats
from astropy.io import fits

from photutils import aperture
from photutils import background
from scipy.spatial import KDTree
from spectral_cube import SpectralCube

import sys
import logging
import subprocess
import argparse
import time

logger = logging.getLogger("CLIFS_Pipeline")

def _do_nothing():
    pass

def _ivar_sum(arr, axis):
    N = 4
    alpha = 1.7
    noise_correction = 1.0 + alpha * np.log10(N)
    var = 1 / arr
    new_var = noise_correction * np.nansum(var, axis=axis)
    return 1 / new_var

def _standard_err_mean(arr, axis):
    return np.nanmean(arr, axis = axis) / np.sqrt(2)

def crop_cube(galaxy, data, ivar):
    xmin_py = galaxy.config["cube"]["xmin"]
    xmax_py = galaxy.config["cube"]["xmax"]
    ymin_py = galaxy.config["cube"]["ymin"]
    ymax_py = galaxy.config["cube"]["ymax"]
    data = data[:, ymin_py:ymax_py+1, xmin_py:xmax_py+1]
    ivar = ivar[:, ymin_py:ymax_py+1, xmin_py:xmax_py+1]
    return data, ivar

def fibre_to_pixel_flux_conversion(data, ivar, cdelt):
    Afibre = np.pi * (2.6 / 2) ** 2
    Apx = (3600 * cdelt) ** 2
    return data / ((Afibre / Apx) * 1e-17), ivar / ((Apx / Afibre) ** 2 * 1e34)

def format_preprocess_output_header(galaxy, head_in, shape, wcs, w0, dw, fullfield, maptype):
    head_new = wcs.to_header()
    head_new["NAXIS"] = (3, "Number of array dimensions")
    head_new["NAXIS1"] = shape[2]
    head_new["NAXIS2"] = shape[1]
    head_new["NAXIS3"] = shape[0]
    head_new["PC3_3"] = (dw * 1e-10, "Coordinate transformation matrix element")
    head_new["CDELT1"] = (1., "[deg] Coordinate increment at reference point")
    head_new["CDELT2"] = (1., "[deg] Coordinate increment at reference point")
    head_new["CDELT3"] = (1., "[m] Coordinate increment at reference point")
    head_new["CRVAL3"] = (w0 * 1e-10, "[m] Coordinate value at reference point")
    head_new["CTYPE3"] = ("AWAV", "Air wavelength")
    if not fullfield:
        head_new["CRPIX1"] = (head_in["CRPIX1"] - galaxy.config["cube"]["xmin"], "Pixel coordinate of reference point")
        head_new["CRPIX2"] = (head_in["CRPIX2"] - galaxy.config["cube"]["ymin"], "Pixel coordinate of reference point")
    else:
        head_new["CRPIX1"] = (head_in["CRPIX1"], "Pixel coordinate of reference point")
        head_new["CRPIX2"] = (head_in["CRPIX2"], "Pixel coordinate of reference point")
    head_new["CRPIX3"] = (1.0, "Pixel coordinate of reference point")
    if maptype == "ivar":
        head_new["BUNIT"] = ("1E34 (s2 cm4 Ang2)/erg2", "units of image")
    elif maptype == "flux":
        head_new["BUNIT"] = ("1E-17 erg/(s cm2 Ang)", "units of image")
    else:
        raise ValueError("Invalid maptype provided")
    return head_new

def write_preprocessed_cube(fname, hdul, data, ivar, data_h, ivar_h):
    prim_hdu = fits.PrimaryHDU(header = hdul[0].header)
    data_hdu = fits.ImageHDU(data = data, header = data_h, name = "FLUX")
    ivar_hdu = fits.ImageHDU(data = ivar, header = ivar_h, name = "IVAR")
    hdul_out = fits.HDUList([prim_hdu, data_hdu, ivar_hdu])
    name_split = fname.split(".fit")[0]
    hdul_out.writeto(name_split + "_cal.fit", overwrite = True)
    hdul_out.close()
    return name_split + "_cal.fit"

def preprocess_cube(galaxy, fname, hdul, arm, ext_data=1, ext_ivar=2, ext_fluxcal=5):
    cal_data = hdul[ext_data].data * hdul[ext_fluxcal].data[:, None, None]
    cal_ivar = hdul[ext_ivar].data * (1 / hdul[ext_fluxcal].data[:, None, None] ** 2)
    wcs = WCS(hdul[ext_data].header)
    cdelt = hdul[ext_data].header["CD2_2"]
    wave_orig = utils.spectral_axis_from_wcs(wcs, cal_data.shape[0])
    cal_data, cal_ivar = fibre_to_pixel_flux_conversion(cal_data, cal_ivar, cdelt)
    if galaxy.config["cube"]["xmin"] == -99:
        fullfield = True
    elif galaxy.config["cube"]["xmin"] >= 0:
        fullfield = False
    else:
        raise ValueError("xmin should either = -99 or be >= 0")
    if galaxy.config["pipeline"]["bkgsub"]:
        cal_data = bkg_sub(galaxy, cal_data, wave_orig, wcs)
    if not fullfield:
        cal_data, cal_ivar = crop_cube(galaxy, cal_data, cal_ivar)
    if galaxy.config["pipeline"]["downsample_wav"]:
        wave, cal_data = downsample_wav_axis(wave_orig, cal_data, "flux", return_wave = True)
        cal_ivar = downsample_wav_axis(wave_orig, cal_ivar, "ivar")
    else:
        wave = wave_orig.copy()
    dwave = np.median(np.diff(wave))
    head_new_flux = format_preprocess_output_header(galaxy, hdul[ext_data].header, cal_data.shape, wcs, wave[0], dwave, fullfield, "flux")
    head_new_ivar = format_preprocess_output_header(galaxy, hdul[ext_data].header, cal_data.shape, wcs, wave[0], dwave, fullfield, "ivar")
    fname_cal = write_preprocessed_cube(fname, hdul, cal_data, cal_ivar, head_new_flux, head_new_ivar)
    return fname_cal

def galaxy_mask(galaxy, wcs, shape):
    reff = galaxy.config["galaxy"]["reff"] * units.arcsec
    ba = 1 - galaxy.config["galaxy"]["ell"]
    pa = galaxy.config["galaxy"]["pa"] * units.deg
    aper = aperture.SkyEllipticalAperture(galaxy.c, 2 * reff, 2 * ba * reff, theta = pa)
    aper_px = aper.to_pixel(wcs.celestial)
    mask = aper_px.to_mask().to_image(shape)
    return mask

def smooth_cube_spectral(data, sig):
    data_smooth = np.zeros(data.shape)
    gauss_kernel = convolution.Gaussian1DKernel(sig)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            spec = data_smooth[:, i, j]
            data_smooth[:, i, j] = convolution.convolve_fft(spec, gauss_kernel)
    return data_smooth

def bkg_sub(galaxy, data, wave, wcs):
    if galaxy.config["pipeline"]["bkgsub_galmask"]:
        mask = galaxy_mask(galaxy, wcs, data.shape[1:])
    else:
        mask = np.zeros(data.shape[1:])
    line_mask = utils.eline_mask(wave, galaxy.z)
    bkg_cube = np.zeros(data.shape)
    for ch in range(data.shape[0]):
        sigma_clip = stats.SigmaClip(sigma = 3.0)
        bkg_estimator = background.MedianBackground()
        bkg = background.Background2D(data[ch, :, :], (10, 10), filter_size = (5, 5), mask = mask.astype(bool),
                            sigma_clip = sigma_clip, bkg_estimator = bkg_estimator)
        if line_mask[ch]:
            bkg_cube[ch, :, :] = bkg.background
        else:
            bkg_cube[ch, :, :] = bkg_cube[ch - 1, :, :]
    #bkg_cube = smooth_cube_spectral(bkg_cube, 5)
    data = data - bkg_cube
    return data

def downsample_wav_axis(wave, data, method, return_wave=False, cov=0):
    new_data = np.zeros((data.shape[0] // 2, data.shape[1], data.shape[2]))
    new_wave = np.zeros(new_data.shape[0])
    k = 0
    for i in range(new_data.shape[0]):
        if method == "flux":
            new_data[i, :, :] = 0.5 * (data[k, :, :] + data[k + 1, :, :])
        elif method == "ivar":
            var = (1/4) * (1/data[k, :, :] + 1/data[k + 1, :, :] + 2*cov)
            new_data[i, :, :] = 1 / var
        else:
            raise ValueError("Invalid method keyword provided to 'downsample_wav_axis'")
        new_wave[i] = 0.5 * (wave[k] + wave[k + 1])
        k += 2
    if return_wave: return new_wave, new_data
    return new_data

def downsample_cube_spatial(cube, axes, cube_type, factor=2):
    cube.allow_huge_operations = True
    for a in axes:
        if cube_type == "flux":
            cube = cube.downsample_axis(factor, a, estimator = np.nansum)
        elif cube_type == "ivar":
            cube = cube.downsample_axis(factor, a, estimator = _ivar_sum)
        else:
            raise ValueError
    return cube

def reproject_spectral_axis(cube, l_low, l_high, dl, fill_value=0):
    dl_ang = dl.to(units.AA).value
    new_spectral_axis = np.arange(l_low.to(units.AA).value, l_high.to(units.AA).value + dl_ang / 2, dl_ang) * units.AA
    cube_newspec = cube.spectral_interpolate(new_spectral_axis, fill_value = fill_value, update_function=None)
    return cube_newspec

def stitch_cubes(cube_blue, cube_red, ivar_blue, ivar_red):
    cube_blue.allow_huge_operations = True
    cube_red.allow_huge_operations = True
    ivar_blue.allow_huge_operations = True
    ivar_red.allow_huge_operations = True
    cube_full = (cube_blue * ivar_blue + cube_red * ivar_red) / (ivar_blue + ivar_red)
    ivar_full = (ivar_blue * ivar_blue + ivar_red * ivar_red) / (ivar_blue + ivar_red)
    return cube_full, ivar_full

def write_fullcube(galaxy, fname_out, fname_blue, fname_red, config, flux, ivar, hdr):
    head = fits.Header()
    head["TELESCOP"] = ("WHT", "4.2m William Herschel Telescope")
    head["DETECTOR"] = ("WEAVELIFU", "WEAVE Large IFU")
    head["INFILE_B"] = fname_blue
    head["INFILE_R"] = fname_red
    head["OBJRA"] = config["galaxy"]["ra"]
    head["OBJDEC"] = config["galaxy"]["dec"]
    ivar[ivar < 0] = 0
    mask = np.isnan(flux) | np.isnan(ivar)
    if galaxy.config["pipeline"]["fix_astrometry"]:
        hdr = astrometry.find_astrometry_solution(flux, WCS(hdr))
        logger.info("Fixed WCS solution")
    prim_hdu = fits.PrimaryHDU(header=head)
    img_hdu = fits.ImageHDU(data=flux, header=hdr, name="FLUX")
    ivar_hdu = fits.ImageHDU(data=ivar, header=hdr, name="IVAR")
    mask_hdu = fits.ImageHDU(data=mask.astype(int), header=hdr, name="MASK")
    hdul = fits.HDUList([prim_hdu, img_hdu, ivar_hdu, mask_hdu])
    hdul.writeto(fname_out, overwrite = True)
    fname_flux = fname_out.split(".fits")[0] + "_only-flux.fits"
    hdu_flux = fits.PrimaryHDU(data = flux.astype("float32"), header=hdr)
    hdu_flux.writeto(fname_flux, overwrite = True)
    if galaxy.config["pipeline"]["hdf5"]:
        subprocess.run(["fits2idia", fname_flux])
        logger.info("Converted cube to HDF5")

def fill_holes(data, ivar, N=3, dmax=3):
    #data = cube.unmasked_data[:, :,  :].value.copy()
    mask = np.all(np.isnan(data), axis=0)
    null = np.argwhere(mask)
    nonnull = np.argwhere(~mask)

    tree = KDTree(nonnull)
    dist, ind = tree.query(null, k=N)
    for i, c in enumerate(null):
        if dist[i].max() > dmax:
            continue
        slab = np.array([data[:, cc[0], cc[1]] for cc in nonnull[ind][i]])
        slab_ivar = np.array([ivar[:, cc[0], cc[1]] for cc in nonnull[ind][i]])
        data[:, c[0], c[1]] = np.average(slab, weights=1/dist[i], axis=0)
        ivar[:, c[0], c[1]] = np.average(slab_ivar, weights=1/dist[i], axis=0)
    return data, ivar

def generate_cube(galaxy, fullfield=False):
    if galaxy.config["pipeline"]["downsample_spatial"] and (galaxy.config["pipeline"]["factor_spatial"] is None):
        raise ValueError("If 'downsample_spatial = True', 'factor' cannot be None")
    outdir = galaxy.config["files"]["outdir"]
    fname_blue = galaxy.config["files"]["cube_blue"]
    fname_red = galaxy.config["files"]["cube_red"]
    hdul_blue = fits.open(fname_blue)
    hdul_red = fits.open(fname_red)
    cal_fname_red = preprocess_cube(galaxy, fname_red, hdul_red, "red")
    cal_fname_blue = preprocess_cube(galaxy, fname_blue, hdul_blue, "blue")
    logger.info("Done preprocessing")
    hdul_blue.close()
    hdul_red.close()
    cube_blue = SpectralCube.read(cal_fname_blue, hdu = 1)
    cube_red = SpectralCube.read(cal_fname_red, hdu = 1)
    ivar_blue = SpectralCube.read(cal_fname_blue, hdu = 2)
    ivar_red = SpectralCube.read(cal_fname_red, hdu = 2)
    logger.info("Read flux-calibrated cubes")
    if galaxy.config["pipeline"]["downsample_spatial"]:
        cube_blue = downsample_cube_spatial(cube_blue, [1, 2], "flux", factor = galaxy.config["pipeline"]["factor_spatial"])
        cube_red = downsample_cube_spatial(cube_red, [1, 2], "flux", factor = galaxy.config["pipeline"]["factor_spatial"])
        ivar_blue = downsample_cube_spatial(ivar_blue, [1, 2], "ivar", factor = galaxy.config["pipeline"]["factor_spatial"])
        ivar_red = downsample_cube_spatial(ivar_red, [1, 2], "ivar", factor = galaxy.config["pipeline"]["factor_spatial"])
        logger.info("Done spatial binning")
    wavblue = cube_blue.spectral_axis.to(units.AA).value
    min = np.abs(wavblue - 3700).argmin()
    wavred = cube_red.spectral_axis.to(units.AA).value
    max = np.abs(wavred - 9400).argmin()
    minlam = wavblue[min]
    maxlam = wavred[max]
    cube_blue = reproject_spectral_axis(cube_blue, minlam*units.AA, maxlam*units.AA, cube_blue.header["CDELT3"]*units.m)
    cube_red = reproject_spectral_axis(cube_red, minlam*units.AA, maxlam*units.AA, cube_blue.header["CDELT3"]*units.m)
    ivar_blue = reproject_spectral_axis(ivar_blue, minlam*units.AA, maxlam*units.AA, cube_blue.header["CDELT3"]*units.m)
    ivar_red = reproject_spectral_axis(ivar_red, minlam*units.AA, maxlam*units.AA, cube_blue.header["CDELT3"]*units.m)
    logger.info("Reprojected red and blue cubes onto common spectral axis")
    cube_full, ivar_full = stitch_cubes(cube_blue, cube_red, ivar_blue, ivar_red)
    data = cube_full.unmasked_data[:, :,  :].value.copy()
    ivar = ivar_full.unmasked_data[:, :,  :].value.copy()
    hdr = cube_full.header
    logger.info("Combined red and blue cubes")
    if galaxy.config["pipeline"]["fill_holes"]:
        data, ivar = fill_holes(data, ivar)
    if galaxy.config["pipeline"]["downsample_spatial"]:
        if fullfield:
            outfile = outdir + "/calibrated_cube_full.fits"
        else:
            outfile = outdir + "/calibrated_cube.fits"
        write_fullcube(galaxy, outfile, fname_blue, fname_red, galaxy.config, data, ivar, hdr)
        logger.info(f"Wrote combined, flux-calibrated cube: {outfile}")
    else:
        outfile = outdir + "/calibrated_cube_p5.fits"
        write_fullcube(galaxy, outfile, fname_blue, fname_red, galaxy.config, cube, ivar, hdr)
        logger.info(f"Wrote combined, flux-calibrated cube: {outfile}")
