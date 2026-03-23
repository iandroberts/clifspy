# clifsPy

This is a data analysis and processing package for the Coma Legacy IFU Survey. This package permits the combination and processing of *Level 1* data cubes from the WEAVE spectrograph.

The standard pipeline run includes the following analysis steps:

### Cube pre-processing and combination
- Apply flux calibration
- Spectral and spatial binning (optional)
- Background subtraction (optional)
- Combine blue and red level 1 cubes
- Fix WCS (optional)

### Multi-wavelength
- Make multi-wavelength cutout images from available ancillary data
- Currently supported: GALEX, DESI Legacy, CFHT, Herschel, LOFAR
- Make integrated CO spectrum (optional, IRAM or ALMA)

### MaNGA Data Analysis Pipeline
- Push the calibrated WEAVE data cubes through the MaNGA data analysis pipeline (DAP)
- Custom class in order to format the WEAVE data suitably for entry into the MaNGA DAP
- DAP performs full spectral fitting, outputs maps of: stellar kinematics, emission line flux, gas kinematics, absorption line indices, etc.

### Products
- Creates value-added products outside of what is produced by the MaNGA DAP
- Currently includes star formation rate and stellar mass maps
- More to be added...

### Plotting
- Creates diagnostic plots to evaluate the performance of the MaNGA DAP run

## Usage
The package is centered around the `clifsipe` command, which takes a CLIFS ID value as input.  All of the standard pipeline steps can be run as:
```
clifspipe $clifs_id --default_run
```
It can also be used in a modular fashion, for example to only run the initial data cube processing:
```
clifspipe $clifs_id --process_cube
```
