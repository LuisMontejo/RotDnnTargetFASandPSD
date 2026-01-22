# Target FAS and PSD Generator: Single-Component & RotDnn

**A unified Python library for generating Duration-Dependent Target Power Spectral Density (PSD) and Fourier Amplitude Spectra (FAS) functions compatible with design response spectra.**

> **⚠️ NOTE:** This repository supersedes the functionality of [TargetPSD](https://github.com/LuisMontejo/TargetPSD). It integrates the original single-component methodology (Montejo, 2024) with the new orientation-independent RotDnn methodology (Montejo, 2026).

>  Requires `reqpy_M` (Core spectral processing library) `pip install "reqpy-M[smoothing]"`

>  Cite the code https://doi.org/10.5281/zenodo.18331475

## Overview

In seismic safety assessments, particularly for nuclear facilities (US-NRC SRP 3.7.1), input motions must often match a design response spectrum (DRS) while satisfying a minimum PSD requirement to ensure sufficient energy content.

This repository provides an iterative numerical algorithm to generate target PSDs and FAS that are **duration-dependent**—addressing a critical gap in current guidelines where target PSDs are often implicitly fixed to specific durations (e.g., ~9s in SRP 3.7.1).

The module supports two modes of operation:
1.  **Single-Component Mode:** Generates a target PSD compatible with a standard design spectrum, explicitly accounting for strong motion duration ($SD_{5-75}$).
2.  **RotDnn Mode (Orientation-Independent):** Generates target PSD and FAS functions compatible with **RotDnn** spectra (e.g., RotD50, RotD100) for bidirectional analysis. This mode handles the azimuthal dependence of strong motion duration.

## Methodologies & References

If you use this code, please cite the paper corresponding to the methodology used:

### 1. Single-Component Analysis
* **Method:** Matches a single-component target PSD to a design spectrum using a specified $SD_{5-75}$.
* **Reference:** Montejo, L.A. (2024). "Strong-Motion-Duration-Dependent Power Spectral Density Functions Compatible with Design Response Spectra." *Geotechnics*, 4(4), 1048-1064.  
    [https://doi.org/10.3390/geotechnics4040053](https://doi.org/10.3390/geotechnics4040053)

### 2. RotDnn (Bidirectional) Analysis
* **Method:** Generates orientation-independent target spectra by iteratively adjusting two orthogonal components until their combined RotDnn response matches the target.
* **Reference:** Montejo, L.A. (2026). "Generation of Fourier Amplitude Spectra and Power Spectral Density Functions Compatible with Orientation-Independent Design Spectra for Bidirectional Seismic Analyses of Nuclear Facilities." *Nuclear Engineering and Technology*, 58(5).  
    [https://doi.org/10.1016/j.net.2026.104136](https://doi.org/10.1016/j.net.2026.104136)

## Dependencies

* `numpy`
* `scipy`
* `matplotlib`
* `numba`
* `reqpy_M` (Core spectral processing library, `pip install "reqpy-M[smoothing]"`)

## Repository Structure

TargetFASandPSDGenerator.py: The main library containing the algorithms for both methodologies.

Example_SingleComp_TargetPSD.py: Full example script for the single-component method (2024 paper).

Example_RotDnn_TargetPSD.py: Full example script for the RotDnn method (2026 paper).

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
Luis A. Montejo 
luis.montejo@upr.edu
