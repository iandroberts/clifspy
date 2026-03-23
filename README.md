# clifsPy

This is a data analysis and processing package for the Coma Legacy IFU Survey. This package permits the combination and processing of *Level 1* data cubes from the WEAVE spectrograph.

The standard pipeline run includes the following analysis steps:

## Cube pre-processing and combination
- Apply flux calibration
- Spectral and spatial binning (optional)
- Background subtraction (optional)
- Combine blue and red level 1 cubes
- Fix WCS (optional)
