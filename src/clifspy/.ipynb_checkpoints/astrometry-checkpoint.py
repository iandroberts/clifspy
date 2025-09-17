from photutils.segmentation import detect_sources, make_2dgaussian_kernel, SourceCatalog
from astropy.stats import sigma_clipped_stats
import matplotlib.pyplot as plt
from astropy.convolution import convolve
import numpy as np

def white_image(cube, wave = None, wave_min = None, wave_max = None):
    if (wave_min is not None) or (wave_max is not None):
        assert wave is not None
        wave_mask = np.greater_equal(wave, wave_min) & np.less_equal(wave, wave_max)
        cube = cube[wave_mask]
    return np.nansum(cube, axis = 0)

def segment_image(data, convolve_bool = False, fwhm = 3.0):
    if convolve_bool:
        kernel = make_2dgaussian_kernel(fwhm, size = 5)  # FWHM = 3.0
        convolved_data = convolve(data, kernel)
        stats = sigma_clipped_stats(convolved_data)
        convolved_data -= stats[1]
        threshold = 2.0 * stats[2]
        segm = detect_sources(convolved_data, threshold, npixels = 10)
        return segm, convolved_data
    else:
        stats = sigma_clipped_stats(data)
        data -= stats[1]
        threshold = 2.0 * stats[2]
        segm = detect_sources(data, threshold, npixels = 10)
        return segm

def source_catalog(data, segm, convolved_data = None):
    if convolved_data is not None:
        cat = SourceCatalog(data, segm, convolved_data = convolved_data)
    else:
        cat = SourceCatalog(data, segm)
    return cat.to_table(columns = ["xcentroid", "ycentroid", "xcentroid_quad", "ycentroid_quad", "segment_flux"])        

def run_astrometry_dot_net(tbl, shape, ra_cent, dec_cent, rad):
    from astroquery.astrometry_net import AstrometryNet
    ast = AstrometryNet()
    ast.api_key = "lgqersexiwpimwhi"

    tbl.sort("segment_flux")
    print(tbl["xcentroid"])
    print(tbl["ycentroid"])
    wcs_header = ast.solve_from_source_list(tbl["xcentroid"], tbl["ycentroid"], shape[1], shape[0],
                                            center_ra = ra_cent, center_dec = dec_cent, radius = rad,
                                            solve_timeout = 120)
    print(wcs_header)
    assert 1==2

def center_coords(data, wcs):
    x0, y0 = 0.5 * (data.shape[1] - 1), 0.5 * (data.shape[0] - 1)
    coord = wcs.pixel_to_world(x0, y0)
    return coord

def find_astrometry_solution(cube, wcs, method = "galaxy", wave = None, wave_min = None, wave_max = None, convolve_bool = True):
    data = white_image(cube, wave = wave, wave_min = wave_min, wave_max = wave_max)
    segm, convolved_data = segment_image(data, convolve_bool = convolve_bool)
    src_tbl = source_catalog(data, segm, convolved_data = convolved_data)

    if method == "astrometry_dot_net":
        coord_cent = center_coords(data, wcs.celestial)
        rad = 0.5 * data.shape[0] / 3600
        hdr = run_astrometry_dot_net(src_tbl, data.shape, coord_cent.ra.value, coord_cent.dec.value, rad)
        return hdr

    elif method == "galaxy":
        from astroquery.sdss import SDSS
        from astropy.coordinates import SkyCoord
        maxind = np.argmax(segm.areas)
        x = src_tbl["xcentroid_quad"][maxind]
        y = src_tbl["ycentroid_quad"][maxind]
        c = wcs.celestial.pixel_to_world(x, y)
        result = SDSS.query_region(c, radius = "5 arcsec", spectro = True)
        assert len(result) == 1
        c_sdss = SkyCoord(ra = result["ra"][0], dec = result["dec"][0], unit = "deg")
        x_sdss, y_sdss = wcs.celestial.world_to_pixel(c_sdss)
        print(x, y)
        print(x_sdss, y_sdss)
        dx = x - x_sdss
        dy = y - y_sdss
        new_hdr = wcs.to_header()
        new_hdr["CRPIX1"] += dx
        new_hdr["CRPIX2"] += dy
        return new_hdr

##
