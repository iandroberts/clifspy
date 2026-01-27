import  os
from astropy.stats import sigma_clipped_stats
from photutils.aperture import SkyEllipticalAperture, EllipticalAperture
from astropy.coordinates import SkyCoord
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.collections import PatchCollection
from reproject import reproject_exact, reproject_interp
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from clifspy.galaxy import Galaxy
from astropy.wcs import WCS
import matplotlib.gridspec as gs
from astropy.visualization import (AsymmetricPercentileInterval, PercentileInterval, SqrtStretch,
                                 ImageNormalize, LinearStretch, AsinhStretch)
import astropy.units as u
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"

class panel_image:
    def __init__(self, clifs_id, panels = ["snr", "v_star", "vdisp_star", "flux_ha", "v_ha", "dn4000"], figsize = (9.0, 2.9)):
        self.galaxy = Galaxy(clifs_id)
        self.fig = plt.Figure(figsize = figsize)
        self.axis_grid = gs.GridSpec(2, 5)
        self.axis_grid.update(wspace = 0.05, hspace = 0.1)
        self.panels = panels
        self.maps = self.galaxy.get_maps()

    def _offset_axis(self, ax, labels = False, grid = False):
         # Remove the absolute coordinates
        ra = ax.coords["ra"]
        dec = ax.coords["dec"]
        ra.set_ticks_visible(False)
        ra.set_ticklabel_visible(False)
        dec.set_ticks_visible(False)
        dec.set_ticklabel_visible(False)
        ra.set_axislabel("")
        dec.set_axislabel("")
        # Create an overlay with relative coordinates
        aframe = self.galaxy.c.skyoffset_frame()
        overlay = ax.get_coords_overlay(aframe)
        ra_offset = overlay["lon"]
        dec_offset = overlay["lat"]
        if labels:
            ra_offset.set_axislabel(r"$\Delta\,\mathrm{RA}$")
            dec_offset.set_axislabel(r"$\Delta\,\mathrm{Dec}$")
            ra_offset.set_major_formatter("s")
            dec_offset.set_major_formatter("s")
        else:
            ra_offset.set_axislabel(" ", minpad = -5)
            dec_offset.set_axislabel(" ", minpad = -5)
        ra_offset.set_ticks_visible(labels)
        dec_offset.set_ticks_visible(labels)
        ra_offset.set_ticklabel_visible(labels)
        dec_offset.set_ticklabel_visible(labels)
        ra_offset.set_ticks_position("b")
        dec_offset.set_ticks_position("l")
        ra_offset.set_axislabel_position("b")
        dec_offset.set_axislabel_position("l")
        ra_offset.set_ticklabel_position("b")
        dec_offset.set_ticklabel_position("l")
        if grid:
            overlay.grid(color = "k", alpha = 0.1, lw = 0.5)

    def _get_rmask(self):
        x0, y0 = WCS(self.maps[1].header).celestial.world_to_pixel(self.galaxy.c)
        cd = self.maps[1].header["PC2_2"]
        imshape = self.maps[1].data.shape[1:]
        #yy, xx = np.mgrid[:imshape[0], :imshape[1]]
        r90_px = self.galaxy.config["galaxy"]["r90"] / 3600 / cd
        ba = 1 - self.galaxy.config["galaxy"]["ell"]
        th = np.radians(self.galaxy.config["galaxy"]["pa"] - 90)
        #r = np.sqrt((x0 - xx)**2 + (y0 - yy)**2)
        aper = EllipticalAperture((x0, y0), r90_px, ba*r90_px, theta=th)
        apermask = aper.to_mask(method="center")
        mask_im = apermask.to_image(imshape).astype(bool)
        return ~mask_im

    def optical(self, gax, rgb = False, xlim = None, ylim = None):
        img, img_h = self.galaxy.get_cutout_image("cfht", "G", header = True)
        x0, y0 = WCS(img_h).celestial.world_to_pixel(self.galaxy.c)
        cd = img_h["PC2_2"]
        r90 = self.galaxy.config["galaxy"]["r90"]
        Nr = self.galaxy.config["plotting"]["panel"]["Nr90"]
        xlim = [int(x0 - Nr * (r90 / 3600 / cd)), int(x0 + Nr * (r90 / 3600 / cd))]
        ylim = [int(y0 - Nr * (r90 / 3600 / cd)), int(y0 + Nr * (r90 / 3600 / cd))]
        if rgb:
            imgU, imgU_h = self.galaxy.get_cutout_image("cfht", "U", header = True)
            imgI, imgI_h = self.galaxy.get_cutout_image("cfht", "I2", header = True)
            normU = ImageNormalize(imgU, interval = PercentileInterval(self.galaxy.config["plotting"]["fov"]["b_pct"]), stretch = AsinhStretch(a = 0.05))
            normG = ImageNormalize(img, interval = PercentileInterval(self.galaxy.config["plotting"]["fov"]["g_pct"]), stretch = AsinhStretch(a = 0.05))
            normI = ImageNormalize(imgI, interval = PercentileInterval(self.galaxy.config["plotting"]["fov"]["r_pct"]), stretch = AsinhStretch(a = 0.05))
            rgb_array = np.array([normI(imgI), normG(img), normU(imgU)])
            print(np.max(rgb_array))
            print(np.min(rgb_array))
            ax = self.fig.add_subplot(gax, projection = WCS(img_h).celestial)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.imshow(np.moveaxis(rgb_array, 0, -1))
            self._offset_axis(ax, labels = True)
            ax.text(0.03, 0.97, "CLIFS {}  ({})".format(self.galaxy.clifs_id, self.galaxy.name), color = "w", fontsize = 10,
                    ha = "left", va = "top", transform = ax.transAxes)
        else:
            norm = ImageNormalize(img, interval = PercentileInterval(99.7), stretch = AsinhStretch(a = 0.05))
            ax = self.fig.add_subplot(gax, projection = WCS(img_h).celestial)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.imshow(img, norm = norm, cmap = "binary")
            self._offset_axis(ax, labels = True)
            #ax.tick_params(direction = "in", length = 3.0, width = 0.5)

    def snr(self, gax, xlim = None, ylim = None, yticks = False, xticks = False):
        snr_map = np.copy(self.maps["SPX_SNR"].data)
        snr_map[snr_map < 1] = np.nan
        ax = self.fig.add_subplot(gax, projection = WCS(self.maps[1].header).celestial)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        im = ax.imshow(snr_map,
                       cmap = "viridis",
                       vmin = self.galaxy.config["plotting"]["panel"]["sn_min"][0],
                       vmax = self.galaxy.config["plotting"]["panel"]["sn_max"][0])
        ticks = [self.galaxy.config["plotting"]["panel"]["sn_min"][1], self.galaxy.config["plotting"]["panel"]["sn_max"][1]]
        ticks.insert(1, int(np.average(ticks)))
        cbar = self.fig.colorbar(im, location = "right", pad = 0.04, ticks = ticks)
        cbar.ax.tick_params(direction = "in", labelsize = 7, pad = 3, rotation = 90, length = 1.5, width = 0.3)
        ax.text(0.03, 0.97, r"$(\mathrm{S\,/\,N})_{\,g}$", fontsize = 8, color = "k", ha = "left", va = "top", transform = ax.transAxes)
        ax.set_aspect("equal")
        self._offset_axis(ax, labels = False, grid = True)
        ax.set_facecolor("#dddddd")

    def v_star(self, gax, mask = None, xlim = None, ylim = None, vel_min = -100, vel_max = 100, xticks = False, yticks = False):
        vel = self.maps["STELLAR_VEL"].data
        vel[self.maps["BINID"].data[0] == -1] = np.nan
        vel[self.maps["BIN_SNR"].data < 8] = np.nan
        #vel[self._get_rmask()] = np.nan
        if mask is not None:
            vel[~mask] = np.nan
        ax = self.fig.add_subplot(gax, projection = WCS(self.maps[1].header, naxis = 2))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        im = ax.imshow(
                       vel,
                       cmap = "RdBu_r",
                       vmin = self.galaxy.config["plotting"]["panel"]["v_star_min"][0],
                       vmax = self.galaxy.config["plotting"]["panel"]["v_star_max"][0],
                      )
        ticks = [self.galaxy.config["plotting"]["panel"]["v_star_min"][1], self.galaxy.config["plotting"]["panel"]["v_star_max"][1]]
        ticks.insert(1, int(np.average(ticks)))
        cbar = self.fig.colorbar(im, location = "right", pad = 0.04, ticks = ticks)
        cbar.ax.tick_params(direction = "in", labelsize = 7, pad = 3, rotation = 90, length = 1.5, width = 0.3)
        ax.text(0.03, 0.97, r"$V_\bigstar$", fontsize = 8, ha = "left", va = "top", transform = ax.transAxes)
        self._offset_axis(ax, labels = False, grid = True)
        ax.set_facecolor("#dddddd")

    def vdisp_star(self, gax, mask=None, xlim=None, ylim=None, xticks=False, yticks=False):
        vel = self.maps["STELLAR_SIGMA"].data
        vel[self.maps["BINID"].data[0] == -1] = np.nan
        vel[self.maps["BIN_SNR"].data < 8] = np.nan
        #vel[self._get_rmask()] = np.nan
        if mask is not None:
            vel[~mask] = np.nan
        ax = self.fig.add_subplot(gax, projection = WCS(self.maps[1].header, naxis = 2))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        im = ax.imshow(
                       vel,
                       cmap = "inferno_r",
                       vmin = self.galaxy.config["plotting"]["panel"]["vdisp_star_min"][0],
                       vmax = self.galaxy.config["plotting"]["panel"]["vdisp_star_max"][0],
                      )
        ticks = [self.galaxy.config["plotting"]["panel"]["vdisp_star_min"][1], self.galaxy.config["plotting"]["panel"]["vdisp_star_max"][1]]
        ticks.insert(1, int(np.average(ticks)))
        cbar = self.fig.colorbar(im, location = "right", pad = 0.04, ticks = ticks)
        cbar.ax.tick_params(direction = "in", labelsize = 7, pad = 3, rotation = 90, length = 1.5, width = 0.3)
        ax.text(0.03, 0.97, r"$\sigma_\bigstar$", fontsize = 8, ha = "left", va = "top", transform = ax.transAxes)
        self._offset_axis(ax, labels = False, grid = True)
        ax.set_facecolor("#dddddd")

    def dn4000(self, gax, xlim = None, ylim = None, mask = None, yticks = False, xticks = False):
        d4 = self.maps["SPECINDEX"].data[44]
        snr_map = np.copy(self.maps["SPX_SNR"].data)
        d4[snr_map < 3] = np.nan
        if mask is not None:
            d4[~mask] = np.nan
        ax = self.fig.add_subplot(gax, projection = WCS(self.maps[1].header, naxis = 2))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        im = ax.imshow(d4,
                       vmin = self.galaxy.config["plotting"]["panel"]["dn4000_min"][0],
                       vmax = self.galaxy.config["plotting"]["panel"]["dn4000_max"][0],
                       cmap = "RdBu_r")
        ticks = [self.galaxy.config["plotting"]["panel"]["dn4000_min"][1], self.galaxy.config["plotting"]["panel"]["dn4000_max"][1]]
        ticks.insert(1, float(np.average(ticks)))
        cbar = self.fig.colorbar(im, location = "right", pad = 0.04, ticks = ticks)
        cbar.ax.tick_params(direction = "in", labelsize = 7, pad = 3, rotation = 90, length = 1.5, width = 0.3)
        ax.text(0.03, 0.97, r"$\mathrm{D_n4000}$", fontsize = 8, ha = "left", va = "top", transform = ax.transAxes)
        self._offset_axis(ax, labels = False, grid = True)
        ax.set_facecolor("#dddddd")

    def flux_ha(self, gax, xlim = None, ylim = None, return_mask = False, yticks = False, xticks = False):
        flux = self.galaxy.get_eline_map("Ha-6564")
        flux_sn = flux * np.sqrt(self.galaxy.get_eline_map("Ha-6564", map = "GFLUX_IVAR"))
        mask = flux_sn < 4
        flux[mask] = np.nan
        ax = self.fig.add_subplot(gax, projection = WCS(self.maps[1].header, naxis = 2))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        norm = ImageNormalize(flux, vmin = self.galaxy.config["plotting"]["panel"]["flux_ha_min"][0], vmax = self.galaxy.config["plotting"]["panel"]["flux_ha_max"][0], stretch = SqrtStretch())
        im = ax.imshow(
                       flux,
                       cmap = "viridis",
                       #vmin = self.galaxy.config["plotting"]["panel"]["flux_ha_min"][0],
                       #vmax = self.galaxy.config["plotting"]["panel"]["flux_ha_max"][0],
                       norm = norm,
                      )
        ticks = [self.galaxy.config["plotting"]["panel"]["flux_ha_min"][1], self.galaxy.config["plotting"]["panel"]["flux_ha_max"][1]]
        ticks.insert(1, int(np.average(ticks)))
        cbar = self.fig.colorbar(im, location = "right", pad = 0.04, ticks = ticks)
        cbar.ax.tick_params(direction = "in", labelsize = 7, pad = 3, rotation = 90, length = 1.5, width = 0.3)
        ax.text(0.03, 0.97, r"$F_\mathrm{H\alpha}$", fontsize = 8, ha = "left", va = "top", transform = ax.transAxes)
        self._offset_axis(ax, labels = False, grid = True)
        ax.set_facecolor("#dddddd")
        if return_mask:
            return mask

    def v_ha(self, gax, mask = None, xlim = None, ylim = None, yticks = False, xticks = False):
        vel = self.galaxy.get_eline_map("Ha-6564", map = "GVEL")
        flux = self.galaxy.get_eline_map("Ha-6564")
        flux_sn = flux * np.sqrt(self.galaxy.get_eline_map("Ha-6564", map = "GFLUX_IVAR"))
        mask = flux_sn < 4
        vel[mask] = np.nan
        ax = self.fig.add_subplot(gax, projection = WCS(self.maps[1].header, naxis = 2))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        im = ax.imshow(
                       vel,
                       cmap = "RdBu_r",
                       vmin = self.galaxy.config["plotting"]["panel"]["v_ha_min"][0],
                       vmax = self.galaxy.config["plotting"]["panel"]["v_ha_max"][0],
                      )
        ticks = [self.galaxy.config["plotting"]["panel"]["v_ha_min"][1], self.galaxy.config["plotting"]["panel"]["v_ha_max"][1]]
        ticks.insert(1, int(np.average(ticks)))
        cbar = self.fig.colorbar(im, location = "right", pad = 0.04, ticks = ticks)
        cbar.ax.tick_params(direction = "in", labelsize = 7, pad = 3, rotation = 90, length = 1.5, width = 0.3)
        ax.text(0.03, 0.97, r"$V_\mathrm{gas}$", fontsize = 8, ha = "left", va = "top", transform = ax.transAxes)
        self._offset_axis(ax, labels = False, grid = True)
        ax.set_facecolor("#dddddd")

    def vdisp_ha(self, gax, mask = None, xlim = None, ylim = None, yticks = False, xticks = False, vel_min = 0, vel_max = 140):
        vel = self.galaxy.get_eline_map("Ha-6564", map = "GSIGMA")
        flux = self.galaxy.get_eline_map("Ha-6564")
        flux_sn = flux * self.galaxy.get_eline_map("Ha-6564", map = "GFLUX_IVAR")
        mask = flux_sn < 4
        vel[mask] = np.nan
        ax = self.fig.add_subplot(gax, projection = WCS(self.maps[1].header, naxis = 2))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        im = ax.imshow(
                       vel,
                       cmap = "inferno_r",
                       vmin = vel_min,
                       vmax = vel_max,
                      )
        cbar = self.fig.colorbar(im, location = "right", pad = 0.04, ticks = [20, 120])
        cbar.ax.tick_params(direction = "in", labelsize = 7, pad = 3, rotation = 90, length = 1.5, width = 0.3)
        ax.text(0.03, 0.97, r"$\sigma_\mathrm{gas}$", fontsize = 8, ha = "left", va = "top", transform = ax.transAxes)
        self._offset_axis(ax, labels = False, grid = True)
        ax.set_facecolor("#dddddd")

    def make(self, filepath, rgb = False):
        if len(self.panels) != 6:
            raise IndexError("List of panels should have precisely six elements")
        self.optical(self.axis_grid[0:2, 0:2], rgb = rgb)
        x0, y0 = WCS(self.maps[1].header).celestial.world_to_pixel(self.galaxy.c)
        cd = self.maps[1].header["PC2_2"]
        r90 = self.galaxy.config["galaxy"]["r90"]
        Nr = self.galaxy.config["plotting"]["panel"]["Nr90"]
        xlim = [int(x0 - Nr * (r90 / 3600 / cd)), int(x0 + Nr * (r90 / 3600 / cd))]
        ylim = [int(y0 - Nr * (r90 / 3600 / cd)), int(y0 + Nr * (r90 / 3600 / cd))]
        for i in range(len(self.panels)):
            r = i // 3
            c = (i % 3) + 2
            getattr(self, self.panels[i])(self.axis_grid[r, c], xlim=xlim, ylim=ylim)
        self.fig.savefig(filepath + ".png", bbox_inches = "tight", pad_inches = 0.03)
        self.fig.savefig(filepath + ".pdf", bbox_inches = "tight", pad_inches = 0.03)

def specfit(galaxy):
    eline_labels = galaxy.config["plotting"]["specfit"]["eline_labels"]
    absline_labels = galaxy.config["plotting"]["specfit"]["absline_labels"]
    if not eline_labels ^ absline_labels:
        raise ValueError("One of 'eline' or 'abs' labels must be true (but not both)")
    zgal = galaxy.z
    c = 2.998e+5
    data_cube = fits.open(galaxy.config["files"]["cube_sci"])
    model_cube = galaxy.get_modelcube()
    maps = galaxy.get_maps()
    flux = data_cube["FLUX"].data
    ivar = data_cube["IVAR"].data
    wcs = WCS(data_cube["FLUX"].header)
    model_flux = model_cube["MODEL"].data
    wcs_model = WCS(model_cube["MODEL"].header)
    # Wavelength vector
    nwave = flux.shape[0]
    coo = np.array([np.ones(nwave), np.ones(nwave), np.arange(nwave) + 1]).T
    wave_model = wcs_model.all_pix2world(coo, 1)[:, 2] * wcs.wcs.cunit[2].to("angstrom")
    wave = (wcs.all_pix2world(coo, 1)[:,2] * wcs.wcs.cunit[2].to("angstrom")) * u.AA
    wlum = wave.to(u.um).value
    wave = ((1+1e-6*(287.6155+1.62887/wlum**2+0.01360/wlum**4)) * wave).to(u.AA).value
    # Mask for ccd gaps
    lgap_red = [7570, 7695]
    lgap_blue = [5480, 5580]
    mask_gap_red = np.greater(wave, lgap_red[0]) & np.less(wave, lgap_red[1])
    mask_gap_blue = np.greater(wave, lgap_blue[0]) & np.less(wave, lgap_blue[1])
    x, y = np.round(wcs.celestial.world_to_pixel(galaxy.c)).astype(int)
    vel = galaxy.get_eline_map("Ha-6564", map = "GVEL")[y, x]
    spec = flux[:, y, x]
    spec_err = 1 / np.sqrt(ivar[:, y, x])
    err = 1 / np.sqrt(ivar[:, y, x])
    model_spec = model_flux[:, y, x]
    spec[mask_gap_blue] = np.nan
    spec[mask_gap_red] = np.nan
    err[mask_gap_blue] = np.nan
    err[mask_gap_red] = np.nan

    plt.rcParams["text.usetex"] = True
    fig = plt.figure(figsize = (9.0, 6.0))
    grid = gs.GridSpec(3, 2)
    grid.update(hspace = 0.3)
    ax = fig.add_subplot(grid[0, 0:2])
    #fig, ax = plt.subplots(2, 2, figsize = (9.0, 4.5))
    ## Full spectrum ##
    ax.axvspan(lgap_blue[0], lgap_blue[1], color = "k", lw = 0, alpha = 0.2)
    ax.axvspan(lgap_red[0], lgap_red[1], color = "k", lw = 0, alpha = 0.2)
    ax.plot(wave, spec, color = "k", lw = 1.0, drawstyle = "steps")
    ax.plot(wave_model, model_spec, color = "C0", lw = 0.8, drawstyle = "steps")
    ax.set_xlim(3725, 9400)
    ax.text(0.015, 0.97, r"CLIFS {}:$\;\;$({}, {})".format(galaxy.clifs_id, x, y), fontsize = 9,
               ha = "left", va = "top", transform = ax.transAxes)

    ## OII, Hdelta, Hgamma ##
    ax = fig.add_subplot(grid[1, 0])
    #Including an inset axis in order to zoom-in on the OII doublet
    if eline_labels:
        mask = np.greater(wave, 3701 * (1 + zgal) * (1 + vel / c)) & np.less(wave, 4020 * (1 + zgal) * (1 + vel / c))
        mask_model = np.greater(wave_model, 3701 * (1 + zgal) * (1 + vel / c)) & np.less(wave_model, 4020 * (1 + zgal) * (1 + vel / c))
    else:
        mask = np.greater(wave, 3850 * (1 + zgal) * (1 + vel / c)) & np.less(wave, 4350 * (1 + zgal) * (1 + vel / c))
        mask_model = np.greater(wave_model, 3850 * (1 + zgal) * (1 + vel / c)) & np.less(wave_model, 4350 * (1 + zgal) * (1 + vel / c))
    ax.fill_between(wave[mask], spec[mask] - spec_err[mask], spec[mask] + spec_err[mask],
        color="k", lw=0, alpha=0.25, step="pre")
    ax.plot(wave[mask], spec[mask], color = "k", lw = 1.0, drawstyle = "steps")
    ax.plot(wave_model[mask_model], model_spec[mask_model], color = "C0", lw = 0.8, drawstyle = "steps")
    ax.set_xlim(wave[mask].min(), wave[mask].max())
    if galaxy.config["plotting"]["specfit"]["inset"]:
        axin = ax.inset_axes([0.25, 0.55, 0.6, 0.4],
            xlim = (3732 * (1 + zgal) * (1 + vel / c), 4000 * (1 + zgal) * (1 + vel / c)),
            ylim = galaxy.config["plotting"]["specfit"]["inset_ylim"], yticklabels = [])
        axin.fill_between(wave[mask], spec[mask] - spec_err[mask], spec[mask] + spec_err[mask],
            color="k", lw=0, alpha=0.25, step="pre")
        axin.plot(wave[mask], spec[mask], color = "k", lw = 1.0, drawstyle = "steps")
        axin.plot(wave_model[mask_model], model_spec[mask_model], color = "C0", lw = 0.8, drawstyle = "steps")
        axin.set_yticks([])
        axin.tick_params(labelsize = 8)
    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.set_ylim(ymin, 1.25 * ymax)
    # Adjust the ylim in order to fit in the line labels, probably far from the best way to do this...
    if eline_labels:
        ax.text(3728 * (1 + zgal) * (1 + vel / c), 0.88, r"$\textsc{Oii}$",
            fontsize=8, ha="center", va="bottom", transform=trans)
        ax.vlines([3727 * (1 + zgal) * (1 + vel / c), 3729 * (1 + zgal) * (1 + vel / c)],
            0.77, 0.85, color = "k", lw = 0.75, transform=trans)
    else:
        ax.text(3934.8 * (1 + zgal) * (1 + vel / c), 0.88, r"K", fontsize = 8, ha = "center", va = "bottom", transform=trans)
        ax.vlines(3934.8 * (1 + zgal) * (1 + vel / c), 0.77, 0.85, color = "k", lw = 0.75, transform=trans)
        ax.text(3969.6 * (1 + zgal) * (1 + vel / c), 0.88, r"H", fontsize = 8, ha = "center", va = "bottom", transform=trans)
        ax.vlines(3969.6 * (1 + zgal) * (1 + vel / c), 0.77, 0.85, color = "k", lw = 0.75, transform=trans)
        ax.text(4305.6 * (1 + zgal) * (1 + vel / c), 0.88, r"G", fontsize = 8, ha = "center", va = "bottom", transform=trans)
        ax.vlines(4305.6 * (1 + zgal) * (1 + vel / c), 0.77, 0.85, color = "k", lw = 0.75, transform=trans)

    ## Hgamma, Hdelta ##
    #ax = fig.add_subplot(grid[1, 1])
    #mask = np.greater(wave, 4080 * (1 + zgal) * (1 + vel / c)) & np.less(wave, 4360 * (1 + zgal) * (1 + vel / c))
    #mask_model = np.greater(wave_model, 4080 * (1 + zgal) * (1 + vel / c)) & np.less(wave_model, 4360 * (1 + zgal) * (1 + vel / c))
    #ax.plot(wave[mask], spec[mask], color = "k", lw = 1.0, drawstyle = "steps")
    #ax.plot(wave_model[mask_model], model_spec[mask_model], color = "C0", lw = 0.8, drawstyle = "steps")
    #ax.set_xlim(wave[mask].min(), wave[mask].max())
    #ymin = ax.get_ylim()[0]
    #ymax = ax.get_ylim()[1]
    #if eline_labels:
        #ax.text(4341 * (1 + zgal) * (1 + vel / c), 1.09 * ymax, r"$\mathrm{H\gamma}$", fontsize = 8, ha = "center", va = "bottom")
        #ax.vlines(4341 * (1 + zgal) * (1 + vel / c), 1.0 * ymax, 1.08 * ymax, color = "k", lw = 0.75)
        #ax.text(4102 * (1 + zgal) * (1 + vel / c), 1.09 * ymax, r"$\mathrm{H\delta}$", fontsize = 8, ha = "center", va = "bottom")
        #ax.vlines(4102 * (1 + zgal) * (1 + vel / c), 1.0 * ymax, 1.08 * ymax, color = "k", lw = 0.75)
        #ax.set_ylim(ymin, 1.2 * ymax)

    ## Calcium triplet ##
    ax = fig.add_subplot(grid[2, 1])
    mask = np.greater(wave, 8440 * (1 + zgal) * (1 + vel / c)) & np.less(wave, 8720 * (1 + zgal) * (1 + vel / c))
    mask_model = (np.greater(wave_model, 8440 * (1 + zgal) * (1 + vel / c)) &
        np.less(wave_model, 8720 * (1 + zgal) * (1 + vel / c)))
    ax.fill_between(wave[mask], spec[mask] - spec_err[mask], spec[mask] + spec_err[mask],
        color="k", lw=0, alpha=0.25, step="pre")
    ax.plot(wave[mask], spec[mask], color = "k", lw = 1.0, drawstyle = "steps")
    ax.plot(wave_model[mask_model], model_spec[mask_model], color = "C0", lw = 0.8, drawstyle = "steps")
    ax.set_xlim(wave[mask].min(), wave[mask].max())
    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.set_ylim(ymin, 1.25 * ymax)
    if eline_labels or absline_labels:
        ax.text(8500 * (1 + zgal) * (1 + vel / c), 0.88, r"$\mathrm{Ca}\,\textsc{ii}$",
            fontsize = 8, ha = "center", va = "bottom", transform=trans)
        ax.vlines(8500 * (1 + zgal) * (1 + vel / c), 0.77, 0.85, color = "k", lw = 0.75, transform=trans)
        ax.text(8544 * (1 + zgal) * (1 + vel / c), 0.88, r"$\mathrm{Ca}\,\textsc{ii}$",
            fontsize = 8, ha = "center", va = "bottom", transform=trans)
        ax.vlines(8544 * (1 + zgal) * (1 + vel / c), 0.77, 0.85, color = "k", lw = 0.75, transform=trans)
        ax.text(8664 * (1 + zgal) * (1 + vel / c), 0.88, r"$\mathrm{Ca}\,\textsc{ii}$",
            fontsize = 8, ha = "center", va = "bottom", transform=trans)
        ax.vlines(8664 * (1 + zgal) * (1 + vel / c), 0.77, 0.85, color = "k", lw = 0.75, transform=trans)

    ## Hbeta, OIII ##
    ax = fig.add_subplot(grid[1, 1])
    if eline_labels:
        mask = np.greater(wave, 4840 * (1 + zgal) * (1 + vel / c)) & np.less(wave, 5026 * (1 + zgal) * (1 + vel / c))
        mask_model = (np.greater(wave_model, 4840 * (1 + zgal) * (1 + vel / c)) &
            np.less(wave_model, 5026 * (1 + zgal) * (1 + vel / c)))
    else:
        mask = np.greater(wave, 5150 * (1 + zgal) * (1 + vel / c)) & np.less(wave, 5225 * (1 + zgal) * (1 + vel / c))
        mask_model = (np.greater(wave_model, 5150 * (1 + zgal) * (1 + vel / c)) &
            np.less(wave_model, 5225 * (1 + zgal) * (1 + vel / c)))
    ax.fill_between(wave[mask], spec[mask] - spec_err[mask], spec[mask] + spec_err[mask],
        color="k", lw=0, alpha=0.25, step="pre")
    ax.plot(wave[mask], spec[mask], color = "k", lw = 1.0, drawstyle = "steps")
    ax.plot(wave_model[mask_model], model_spec[mask_model], color = "C0", lw = 0.8, drawstyle = "steps")
    ax.set_xlim(wave[mask].min(), wave[mask].max())
    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.set_ylim(ymin, 1.25 * ymax)
    if eline_labels:
        ax.text(4862 * (1 + zgal) * (1 + vel / c), 0.88, r"$\mathrm{H\beta}$", fontsize = 8,
            ha = "center", va = "bottom", transform=trans)
        ax.vlines(4862 * (1 + zgal) * (1 + vel / c), 0.77, 0.85, color = "k", lw = 0.75, transform=trans)
        ax.text(5008 * (1 + zgal) * (1 + vel / c), 0.88, r"$\textsc{Oiii}$", fontsize = 8,
            ha = "center", va = "bottom", transform=trans)
        ax.text(4960 * (1 + zgal) * (1 + vel / c), 0.88, r"$\textsc{Oiii}$", fontsize = 8,
            ha = "center", va = "bottom", transform=trans)
        ax.vlines([4960 * (1 + zgal) * (1 + vel / c), 5008 * (1 + zgal) * (1 + vel / c)], 0.77, 0.85,
            color = "k", lw = 0.75, transform=trans)
    else:
        ax.text(5167.3 * (1 + zgal) * (1 + vel / c), 0.88, r"$\mathrm{Mg}\,\textsc{i}$", fontsize = 8,
            ha = "center", va = "bottom", transform=trans)
        ax.vlines(5167.3 * (1 + zgal) * (1 + vel / c), 0.77, 0.85, color = "k", lw = 0.75, transform=trans)
        ax.text(5172.7 * (1 + zgal) * (1 + vel / c), 0.88, r"$\mathrm{Mg}\,\textsc{i}$", fontsize = 8,
            ha = "center", va = "bottom", transform=trans)
        ax.vlines(5172.7 * (1 + zgal) * (1 + vel / c), 0.77, 0.85, color = "k", lw = 0.75, transform=trans)
        ax.text(5183.6 * (1 + zgal) * (1 + vel / c), 0.88, r"$\mathrm{Mg}\,\textsc{i}$", fontsize = 8,
            ha = "center", va = "bottom", transform=trans)
        ax.vlines(5183.6 * (1 + zgal) * (1 + vel / c), 0.77, 0.85, color = "k", lw = 0.75, transform=trans)

    ## NII, Halpha, SII ##
    ax = fig.add_subplot(grid[2, 0])
    if eline_labels:
        mask = np.greater(wave, 6538 * (1 + zgal) * (1 + vel / c)) & np.less(wave, 6744 * (1 + zgal) * (1 + vel / c))
        mask_model = (np.greater(wave_model, 6538 * (1 + zgal) * (1 + vel / c)) &
            np.less(wave_model, 6744 * (1 + zgal) * (1 + vel / c)))
    else:
        mask = np.greater(wave, 5800 * (1 + zgal) * (1 + vel / c)) & np.less(wave, 6000 * (1 + zgal) * (1 + vel / c))
        mask_model = (np.greater(wave_model, 5800 * (1 + zgal) * (1 + vel / c)) &
            np.less(wave_model, 6000 * (1 + zgal) * (1 + vel / c)))
    ax.fill_between(wave[mask], spec[mask] - spec_err[mask], spec[mask] + spec_err[mask],
        color="k", lw=0, alpha=0.25, step="pre")
    ax.plot(wave[mask], spec[mask], color = "k", lw = 1.0, drawstyle = "steps")
    ax.plot(wave_model[mask_model], model_spec[mask_model], color = "C0", lw = 0.8, drawstyle = "steps")
    ax.set_xlim(wave[mask].min(), wave[mask].max())
    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.set_ylim(ymin, 1.25 * ymax)
    if eline_labels:
        ax.text(6564 * (1 + zgal) * (1 + vel / c), 0.88, r"$\mathrm{H\alpha}$", fontsize = 8,
            ha = "center", va = "bottom", transform=trans)
        ax.vlines(6564 * (1 + zgal) * (1 + vel / c), 0.77, 0.85, color = "k", lw = 0.75, transform=trans)
        ax.text(6549 * (1 + zgal) * (1 + vel / c), 0.88, r"$\textsc{Nii}$", fontsize = 8,
            ha = "center", va = "bottom", transform=trans)
        ax.text(6585 * (1 + zgal) * (1 + vel / c), 0.88, r"$\textsc{Nii}$", fontsize = 8,
            ha = "center", va = "bottom", transform=trans)
        ax.vlines([6549 * (1 + zgal) * (1 + vel / c), 6585 * (1 + zgal) * (1 + vel / c)], 0.77, 0.85,
            color = "k", lw = 0.75, transform=trans)
        ax.text(6718 * (1 + zgal) * (1 + vel / c), 0.88, r"$\textsc{Sii}$", fontsize = 8,
            ha = "center", va = "bottom", transform=trans)
        ax.text(6732 * (1 + zgal) * (1 + vel / c), 0.88, r"$\textsc{Sii}$", fontsize = 8,
            ha = "center", va = "bottom", transform=trans)
        ax.vlines([6718 * (1 + zgal) * (1 + vel / c), 6732 * (1 + zgal) * (1 + vel / c)], 0.77, 0.85,
            color = "k", lw = 0.75, transform=trans)
    else:
        ax.text(5895.6 * (1 + zgal) * (1 + vel / c), 0.88, r"$\mathrm{Na}\,\textsc{i}$", fontsize = 8,
            ha = "center", va = "bottom", transform=trans)
        ax.vlines(5895.6 * (1 + zgal) * (1 + vel / c), 0.75, 0.85, color = "k", lw = 0.75, transform=trans)

    # Figure labels
    fig.supxlabel(r"Wavelength$\;\;\mathrm{[\AA]}$", fontsize = 10, y = 0.03)
    fig.supylabel(r"Flux density$\;\;\mathrm{[10^{-17}\;erg\,s^{1}\,cm^{-2}\,\AA^{-1}}]$", fontsize = 10, x = 0.07)
    # Save
    fig.savefig("/arc/projects/CLIFS/plots/specfits/specfit_clifs{}_{}_{}.pdf".format(galaxy.clifs_id, x, y), bbox_inches = "tight", pad_inches = 0.03)
    fig.savefig("/arc/projects/CLIFS/plots/specfits/specfit_clifs{}_{}_{}.png".format(galaxy.clifs_id, x, y), bbox_inches = "tight", pad_inches = 0.03)

def reproject_manga_map(data, snr, wcs, wcs_out, shape_out, method = "interp"):
    # Factor of 0.25 is to convert MaNGA maps from pixel^-1 to arcsec^-2
    if method == "interp":
        map_reproj, footprint = reproject_interp((data / 0.25, wcs), wcs_out, shape_out = shape_out)
        snr_reproj, footprint = reproject_interp((snr /  0.25, wcs), wcs_out, shape_out = shape_out)
        return map_reproj, snr_reproj
    elif method == "exact":
        map_reproj, footprint = reproject_exact((data / 0.25, wcs), wcs_out, shape_out = shape_out)
        snr_reproj, footprint = reproject_exact((snr /  0.25, wcs), wcs_out, shape_out = shape_out)
        return map_reproj, snr_reproj
    else:
        raise ValueError("'method' must either be 'interp' or 'exact'")

def compare_weave_manga(galaxy, line, fig = None, gax = None, j = None, sn_cut = 4, return_arrays = False):
    weave_map, weave_wcs = galaxy.get_eline_map(line, return_wcs = True)
    weave_ivar = galaxy.get_eline_map(line, map = "GFLUX_IVAR")
    weave_sn = weave_map * np.sqrt(weave_ivar)
    manga_map, manga_wcs = galaxy.get_eline_map(line, return_wcs=True, force_manga=True)
    manga_ivar = galaxy.get_eline_map(line, map="GFLUX_IVAR", force_manga=True)
    manga_sn = manga_map * np.sqrt(manga_ivar)
    manga_reproj, manga_sn = reproject_manga_map(manga_map, manga_sn, manga_wcs, weave_wcs,
        weave_map.shape, method = "exact")
    mask_det = np.greater(weave_sn, sn_cut) & np.greater(manga_sn, sn_cut) & np.greater(weave_map, 0) & np.greater(manga_reproj, 0)
    weave_map[~mask_det] = np.nan
    manga_reproj[~mask_det] = np.nan
    if return_arrays:
        return list(manga_reproj[np.isfinite(manga_reproj)]), list(weave_map[np.isfinite(weave_map)])
    weave_err = weave_map / weave_sn
    manga_err = manga_reproj / manga_sn
    delta = np.log10(weave_map.ravel() / manga_reproj.ravel())
    mu = np.nanmedian(delta)
    sig = np.nanstd(delta)
    ax = fig.add_subplot(gax)
    ax.errorbar(manga_reproj.ravel(), weave_map.ravel(), xerr = 0, yerr = 0, #xerr = manga_err.ravel(), yerr = weave_err.ravel(),
                color = "k", ls = "none", marker = "o", mew = 0, ms = 3, alpha = 0.75, elinewidth = 0.75)
    ax.plot([0.01, 1000], [0.01, 1000], color = "C1", ls = "--")
    ax.set_aspect("equal")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.01, 1000)
    ax.set_ylim(0.01, 1000)
    ax.set_xticks([0.01, 1.0, 100])
    ax.set_yticks([0.01, 1.0, 100])
    ax.text(0.02, 0.98, line, fontsize = 10, ha = "left", va = "top", transform = ax.transAxes)
    ax.text(0.98, 0.11, r"$\Delta = {:.2f}$".format(mu), fontsize = 10, ha = "right", va = "bottom", transform = ax.transAxes)
    ax.text(0.98, 0.02, r"$\sigma = {:.2f}$".format(sig), fontsize = 10, ha = "right", va = "bottom", transform = ax.transAxes)
    if j % 5 > 0:
        ax.set_yticklabels([])
    if j < 10:
        ax.set_xticklabels([])
    plt.minorticks_off()

def weave_manga_line_fluxes(galaxy):
    lines = np.array([
                  "OII-3727",
                  "OII-3729",
                  "Heps-3971",
                  "Hdel-4102",
                  "Hgam-4341",
                  "Hb-4862",
                  "OIII-4960",
                  "OIII-5008",
                  "OI-6302",
                  "OI-6365",
                  "NII-6549",
                  "Ha-6564",
                  "NII-6585",
                  "SII-6718",
                  "SII-6732",
                 ])
    fig = plt.figure(figsize = (9.0, 5.8))
    grid = gs.GridSpec(3, 5)
    grid.update(wspace = 0.1, hspace = 0.1)
    for j in range(lines.size):
        compare_weave_manga(galaxy, lines[j], fig = fig, gax = grid[j], j = j)
    fig.supxlabel(r"MaNGA:$\quad F_\lambda \;\; \mathrm{[erg\,s^{-1}\,cm^{-2}]}$", fontsize = 10, y = 0.04)
    fig.supylabel(r"WEAVE:$\quad F_\lambda \;\; \mathrm{[erg\,s^{-1}\,cm^{-2}]}$", fontsize = 10, x = 0.05)
    fig.savefig("/arc/projects/CLIFS/plots/manga_comparison/weave_manga_line_fluxes_clifs{}.pdf".format(galaxy.clifs_id), bbox_inches = "tight", pad_inches = 0.03)
    fig.savefig("/arc/projects/CLIFS/plots/manga_comparison/weave_manga_line_fluxes_clifs{}.png".format(galaxy.clifs_id), bbox_inches = "tight", pad_inches = 0.03)

def fiber_map(x0, y0, header):
    y_spacing = 3.4 / 3600 / header["PC2_2"]
    x_spacing = y_spacing * np.cos(np.radians(30))
    diameter = 2.6 / 3600 / header["PC2_2"]
    Ny = np.arange(-13, 14)
    Nx = np.arange(-13, 14)
    patches = []
    vertices = ((0., 13.), (-13., 6.5), (-13., -6.5), (0., -13.), (13., -6.5), (13., 6.5)) # CW from top
    m1 = (vertices[0][1] - vertices[1][1]) / (vertices[0][0] - vertices[1][0])
    b1 = vertices[0][1] - m1 * vertices[0][0]
    m2 = (vertices[2][1] - vertices[3][1]) / (vertices[2][0] - vertices[3][0])
    b2 = vertices[2][1] - m2 * vertices[2][0]
    m3 = (vertices[3][1] - vertices[4][1]) / (vertices[3][0] - vertices[4][0])
    b3 = vertices[3][1] - m3 * vertices[3][0]
    m4 = (vertices[5][1] - vertices[0][1]) / (vertices[5][0] - vertices[0][0])
    b4 = vertices[5][1] - m4 * vertices[5][0]
    for i in Nx:
        for j in Ny:
            if (j > i * m1 + b1) or (j + 1 <= i * m2 + b2) or (j + 1 <= i * m3 + b3) or (j > i * m4 + b4):
                continue
            else:
                if i % 2 == 0:
                    p = Circle((x0 + i * x_spacing, y0 + j * y_spacing), radius = diameter / 2)
                else:
                    p = Circle((x0 + i * x_spacing, y0 + j * y_spacing + y_spacing / 2), radius = diameter / 2)
                patches.append(p)
    return patches

def LIFU_boundary(galaxy):
    img, img_h = galaxy.get_cutout_image("cfht", "G", header = True)
    cd = img_h["PC2_2"]
    if galaxy.ra_pnt == -99:
        coord_pnt = SkyCoord(galaxy.ra, galaxy.dec, unit = "deg")
    else:
        coord_pnt = SkyCoord(galaxy.ra_pnt, galaxy.dec_pnt, unit = "deg")
    x_pnt, y_pnt = WCS(img_h).celestial.world_to_pixel(coord_pnt)
    patch = RegularPolygon((x_pnt, y_pnt), numVertices = 6, radius = 45 / 3600 / cd, ec = "w", fc = "none")
    return patch

def MaNGA_boundary(galaxy):
    img, img_h = galaxy.get_cutout_image("cfht", "G", header = True)
    cd = img_h["PC2_2"]
    coord_pnt = SkyCoord(galaxy.ra, galaxy.dec, unit = "deg")
    x_pnt, y_pnt = WCS(img_h).celestial.world_to_pixel(coord_pnt)
    rad_as = np.sqrt(galaxy.manga_Nfib) / 2 / 0.355
    patch = RegularPolygon((x_pnt, y_pnt), numVertices=6, radius=rad_as/3600/cd, ec="w", fc="none", ls="--", orientation=np.pi/2)
    return patch

def fiber_overlay_plot(galaxy, rgb = False, xlim = None, ylim = None, Nr = 2):
    img, img_h = galaxy.get_cutout_image("cfht", "G", header = True)
    x0, y0 = WCS(img_h).celestial.world_to_pixel(galaxy.c)
    cd = img_h["PC2_2"]
    fig = plt.figure(figsize = (3.0, 3.0))
    ax = fig.add_subplot(1, 1, 1, projection = WCS(img_h).celestial)
    if galaxy.ra_pnt == -99:
        coord_pnt = SkyCoord(galaxy.ra, galaxy.dec, unit = "deg")
    else:
        coord_pnt = SkyCoord(galaxy.ra_pnt, galaxy.dec_pnt, unit = "deg")
    x_pnt, y_pnt = WCS(img_h).celestial.world_to_pixel(coord_pnt)
    xlim = [int(x_pnt - 0.75 / 60 / cd), int(x_pnt + 0.75 / 60 / cd)]
    ylim = [int(y_pnt - 0.75 / 60 / cd), int(y_pnt + 0.75 / 60 / cd)]
    if rgb:
        imgU, imgU_h = galaxy.get_cutout_image("cfht", "U", header = True)
        imgI, imgI_h = galaxy.get_cutout_image("cfht", "I2", header = True)
        normU = ImageNormalize(imgU, interval = PercentileInterval(galaxy.config["plotting"]["fov"]["b_pct"]), stretch = AsinhStretch(a = galaxy.config["plotting"]["fov"]["asinh_a"]))
        normG = ImageNormalize(img, interval = PercentileInterval(galaxy.config["plotting"]["fov"]["g_pct"]), stretch = AsinhStretch(a = galaxy.config["plotting"]["fov"]["asinh_a"]))
        normI = ImageNormalize(imgI, interval = PercentileInterval(galaxy.config["plotting"]["fov"]["r_pct"]), stretch = AsinhStretch(a = galaxy.config["plotting"]["fov"]["asinh_a"]))
        rgb_array = np.array([normI(imgI), normG(img), normU(imgU)])
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.imshow(np.moveaxis(rgb_array, 0, -1))
        if galaxy.tail:
            hba, hba_h = galaxy.get_cutout_image("lofar", "hba", header = True)
            lev_hba = sigma_clipped_stats(hba)[2] * np.array([2.0])
            ax.contour(hba, levels = lev_hba, linewidths = 0.75, colors = "w", alpha = 0.8, transform = ax.get_transform(WCS(hba_h)))
        if galaxy.manga:
            patch = MaNGA_boundary(galaxy)
            ax.add_patch(patch)
        if galaxy.weave:
            patch = LIFU_boundary(galaxy)
            ax.add_patch(patch)
        ax.text(0.02, 0.98, "CLIFS {}".format(galaxy.clifs_id), color="w", ha="left", va="top", transform=ax.transAxes)
        ax.coords["ra"].set_axislabel("RA")
        ax.coords["dec"].set_axislabel("Dec")
        ax.coords["ra"].set_ticks_position("b")
        ax.coords["dec"].set_ticks_position("l")
        fig.savefig("/arc/projects/CLIFS/plots/fiber_overlay/fiber_overlay_image_clifs{}.pdf".format(galaxy.clifs_id), bbox_inches = "tight", pad_inches = 0.03)
        fig.savefig("/arc/projects/CLIFS/plots/fiber_overlay/fiber_overlay_image_clifs{}.png".format(galaxy.clifs_id), bbox_inches = "tight", pad_inches = 0.03)
    else:
        norm = ImageNormalize(img, interval = PercentileInterval(99.7), stretch = AsinhStretch(a = 0.05))
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.imshow(img, norm = norm, cmap = "binary")

def _plot_r90(galaxy, ax, wcs):
    sky_aper = SkyEllipticalAperture(galaxy.c, galaxy.r90, galaxy.r90 * (1 - galaxy.ell), theta = galaxy.pa)
    px_aper = sky_aper.to_pixel(wcs)
    px_aper.plot(ax = ax, fc = "none", ec = "k", ls = "--")

def ha_tail_plot(galaxy):
    flux, wcs = galaxy.get_eline_map("Ha-6564", return_wcs = True)
    flux_sn = flux * np.sqrt(galaxy.get_eline_map("Ha-6564", map = "GFLUX_IVAR"))
    flux[flux_sn < 3] = np.nan
    hba, hba_h = galaxy.get_cutout_image("lofar", "hba", header = True)
    lev_hba = sigma_clipped_stats(hba)[2] * np.array([2.5])
    fig = plt.figure(figsize = (4.5, 4.5))
    ax = fig.add_subplot(1, 1, 1, projection = wcs)
    norm = ImageNormalize(flux, vmin = galaxy.config["plotting"]["tail"]["vmin"], vmax = galaxy.config["plotting"]["tail"]["vmax"], stretch = SqrtStretch())
    ax.imshow(flux, norm = norm, cmap = "viridis")
    ax.contour(hba, levels = lev_hba, colors = "C1", transform = ax.get_transform(WCS(hba_h)))
    _plot_r90(galaxy, ax, wcs)
    ax.coords["ra"].set_axislabel("RA")
    ax.coords["dec"].set_axislabel("Dec")
    ax.coords["ra"].set_ticks_position("b")
    ax.coords["dec"].set_ticks_position("l")
    fig.savefig("/arc/projects/CLIFS/plots/ha_tail_plots/ha_tail_clifs{}.pdf".format(galaxy.clifs_id), bbox_inches = "tight", pad_inches = 0.03)
    fig.savefig("/arc/projects/CLIFS/plots/ha_tail_plots/ha_tail_clifs{}.png".format(galaxy.clifs_id), bbox_inches = "tight", pad_inches = 0.03)

def plots_for_clifspipe(galaxy):
    fiber_overlay_plot(galaxy, rgb = True)
    if os.path.isdir(galaxy.config["files"]["outdir_dap"]):
        panel_image(galaxy.clifs_id).make("/arc/projects/CLIFS/plots/panel_images/panel_img_clifs{}".format(galaxy.clifs_id), rgb = True)
        specfit(galaxy)
        if galaxy.manga: weave_manga_line_fluxes(galaxy)
        if galaxy.tail: ha_tail_plot(galaxy)
