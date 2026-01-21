"""
TargetFASandPSDGenerator.py
=====================

**Author:** Luis A. Montejo (luis.montejo@upr.edu)
**Description:** A unified module for generating Duration-Dependent Target Power Spectral Density (PSD) 
    and Fourier Amplitude Spectra (FAS) functions compatible with design response spectra.

    This module implements two distinct methodologies:
    1. **Single-Component**: Generates target PSDs compatible with a standard design spectrum,
       explicitly accounting for strong motion duration.
    2. **RotDnn (Orientation-Independent)**: Generates target PSDs and FAS compatible with 
       RotDnn spectra (e.g., RotD50, RotD100) for bidirectional analysis.

**References:**
    * **Single-Component**: Montejo, L.A. (2024). "Strong-Motion-Duration-Dependent Power 
        Spectral Density Functions Compatible with Design Response Spectra." *Geotechnics*, 4(4), 1048-1064.
    * **RotDnn**: Montejo, L. A. (2026). "Generation of Fourier Amplitude Spectra and Power Spectral Density 
        Functions Compatible with Orientation-Independent Design Spectra for Bidirectional Seismic Analyses 
        of Nuclear Facilities." *Nuclear Engineering and Technology*, 58(5).

**Dependencies:**
    * numpy, scipy, matplotlib
    * reqpy_M (Core spectral library)
    * pykooh (Optional, recommended for fast smoothing)
"""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import logging
import concurrent.futures
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.sparse
from numba import jit
from typing import Tuple, Dict, Any, Optional

# --- Import from core library (reqpy_M) ---
try:
    from reqpy_M import (
        # PSA calculators
        compute_spectrum_fd, 
        rotdnn,
        # PSD/FAS calculators
        calculate_earthquake_psd, 
        calculate_fas_rotDnn, 
        calculate_psd_rotDnn,
        # Utility functions
        log_interp, 
        get_log_freqs, 
        SignificantDuration,
        # Smoothing functions
        _konno_ohmachi_1998_downsample,
        _konno_ohmachi_1998_sparse_matrix, 
        _smooth_boxcar_variable,
        PYKOOH_AVAILABLE,
        warnings
    )
    # Conditionally import pykooh if available
    if PYKOOH_AVAILABLE:
        import pykooh
except ImportError:
    log = logging.getLogger(__name__)
    log.error("Could not import 'reqpy_M' library. Please ensure it is in your Python path.")
    raise

# Set up module logger
log = logging.getLogger(__name__)


# =============================================================================
# PUBLIC API: SINGLE-COMPONENT GENERATOR
# =============================================================================

def generate_single_comp_target_psd(
    target_freqs: np.ndarray,
    target_psa: np.ndarray,
    sd575: float,
    freqs_des: np.ndarray,
    workname: str = 'SingleComp_TargetPSD',
    zi: float = 0.05,
    F1: float = 0.2,
    F2: float = 50.0,
    allow_err: float = 2.5,
    neqsPSD: int = 1000,
    use_multiprocessing: bool = True,
    fas_smoothing_method: Optional[str] = 'konno_ohmachi',
    fas_smoothing_coeff: float = 20.0,
    psd_smoothing_method: Optional[str] = 'konno_ohmachi',
    psd_smoothing_coeff: float = 20.0,
    prefer_pykooh: bool = True
) -> Dict[str, Any]:
    """
    Generates a single-component target PSD compatible with a design response spectrum,
    accounting for strong motion duration.

    Based on the methodology described in Montejo (2024).

    Parameters
    ----------
    target_freqs : np.ndarray
        Frequencies of the target design response spectrum [Hz].
    target_psa : np.ndarray
        Amplitudes of the target design response spectrum (usually in g).
    sd575 : float
        Target significant duration (5-75% Arias Intensity) [s].
    freqs_des : np.ndarray
        Desired output frequency vector for the final smoothed PSD/FAS [Hz].
    workname : str, optional
        Label for the analysis (default: 'SingleComp_TargetPSD').
    zi : float, optional
        Damping ratio for response spectrum matching (default: 0.05).
    F1 : float, optional
        Lower frequency limit for error calculation [Hz] (default: 0.2).
    F2 : float, optional
        Upper frequency limit for error calculation [Hz] (default: 50.0).
    allow_err : float, optional
        Maximum allowable mean percentage error between target and calculated PSA (default: 2.5).
    neqsPSD : int, optional
        Number of synthetic records to generate for final PSD estimation (default: 1000).
    use_multiprocessing : bool, optional
        If True, uses parallel processing for PSA/PSD calculations (default: True).
    fas_smoothing_method : str, optional
        Method to smooth the final FAS ('konno_ohmachi', 'variable_window', or 'none').
    fas_smoothing_coeff : float, optional
        Smoothing coefficient (b-value for Konno-Ohmachi, percentage for variable window).
    psd_smoothing_method : str, optional
        Method to smooth the final PSD ('konno_ohmachi', 'variable_window', or 'none').
    psd_smoothing_coeff : float, optional
        Smoothing coefficient for PSD.
    prefer_pykooh : bool, optional
        If True and pykooh is installed, uses the optimized C++ implementation for smoothing.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing results:
        - 'target_psd': Smoothed target PSD (geometric mean).
        - 'target_fas': Smoothed target FAS.
        - 'target_psd_low_smooth': Smoothed -1 sigma PSD bound.
        - 'target_psd_high_smooth': Smoothed +1 sigma PSD bound.
        - 'sd575_mean_actual': The actual mean duration of the synthetic suite.
        - 'iteration_errors': List of PSA matching errors per iteration.
        - ... (other intermediate data).
    """
    
    neqsPSD = int(neqsPSD)

    # --- 1. Load and Resample Target Spectrum ---
    f_or = target_freqs
    ds_or = target_psa

    # --- 2. Create Time Envelope & Freq Vector ---
    # Total duration is estimated as ~3.54 * SD5-75 to fit the Saragoni-Hart window
    tf = 3.54 * sd575
    fs = 200.0
    dt = 1.0 / fs
    nyquist_freq = fs / 2

    nt = int(tf / dt) + 1
    if nt % 2 != 0:
        tf += dt
        nt += 1

    envelope = _saragoni_hart_w(nt, eps=0.2, n=0.2, tn=0.6)
    sets = np.linspace(10, 100, 10, dtype=int)
    nsets = np.size(sets)
    m = np.arange(0, np.ceil(nt / 2) + 1, dtype=int)
    freqs_raw = m * fs / nt
    
    # Define calculation frequencies for PSA matching
    f_psa = np.hstack((np.array([0.01, 0.02, 0.04, 0.06, 0.08]),
                       np.geomspace(0.1, 50, 100),
                       np.array([55, 60, 70, 80, 90, 100])))
    f_psa = f_psa[f_psa <= nyquist_freq]
    f_psa[-1] = nyquist_freq
    if f_psa[0] < freqs_raw[1]: f_psa[0] = 0.99 * freqs_raw[1]

    psa_target_calc = log_interp(f_psa, f_or, ds_or)
    locs = np.where((f_psa >= F1) & (f_psa <= F2))[0]
    T_psa = 1.0 / f_psa

    # --- 3. Initial FAS Guess ---
    # Uses empirical relationship from Montejo & Vidot-Vega (2017)
    ratio = _get_fas_psa_ratio(f_psa, sd575)
    TFAS = log_interp(f_psa, f_or, ds_or) * ratio
    ds_freqs = log_interp(freqs_raw[1:], f_psa, psa_target_calc)
    ds_freqs = np.concatenate(([0], ds_freqs))
    TFAS_freqs = log_interp(freqs_raw[1:], f_psa, TFAS)
    TFAS_freqs = np.concatenate(([0], TFAS_freqs))

    PSAavg = np.zeros((len(T_psa), nsets))
    FAStarget = np.zeros((len(freqs_raw), nsets))
    FAStarget[:, 0] = TFAS_freqs
    calc_errs = np.zeros(nsets)

    log.info('*' * 20)
    log.info('Generating single-component spectrum compatible FAS')
    log.info(f'Target error: {allow_err:.2f}%, max # of iters.: {nsets}')
    log.info('*' * 20)

    # --- 4. Iterative FAS-to-PSA Matching ---
    k_final = 0
    for k in range(nsets):
        n_records_in_set = sets[k]
        log.info(f"Starting iteration {k+1}/{nsets}...")
        
        TaFAS = FAStarget[:, k]
        
        if use_multiprocessing:
            PSA_results_list = []
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(_single_comp_psa_worker, nt, envelope, m, dt, TaFAS, T_psa, zi) 
                    for _ in range(n_records_in_set)
                ]
                for f in concurrent.futures.as_completed(futures):
                    PSA_results_list.append(f.result())
            PSA_set = np.array(PSA_results_list).T
        else:
            PSA_set = np.zeros((len(T_psa), n_records_in_set))
            for q in range(n_records_in_set):
                PSA_set[:, q] = _single_comp_psa_worker(nt, envelope, m, dt, TaFAS, T_psa, zi)

        # GEOMETRIC MEAN (Median) used for convergence check
        PSAavg[:, k] = np.exp(np.mean(np.log(PSA_set + 1e-30), axis=1))
        
        diflimits = np.abs(psa_target_calc[locs] - PSAavg[locs, k]) / psa_target_calc[locs]
        calc_errs[k] = np.mean(diflimits) * 100
        k_final = k

        log.info(f'Iteration {k+1}: error = {calc_errs[k]:.2f}%')
        if calc_errs[k] < allow_err:
            log.info(f'Error satisfied at iteration {k+1}.')
            break
        elif k != nsets - 1:
            # Update FAS based on PSA mismatch ratios
            PSAavg_interp = log_interp(freqs_raw[1:], f_psa, PSAavg[:, k])
            with np.errstate(divide='ignore', invalid='ignore'):
                factor = ds_freqs[1:] / PSAavg_interp
            factor[np.isnan(factor) | np.isinf(factor)] = 1.0
            FAStarget[1:, k + 1] = factor * FAStarget[1:, k]

    PSAavg = PSAavg[:, :k_final + 1]
    FAStarget = FAStarget[:, :k_final + 1]
    calc_errs = calc_errs[:k_final + 1]
    
    # --- 5. Generate Suite and Compute Raw PSDs ---
    log.info('*' * 20)
    log.info(f'Now generating target PSD using {neqsPSD} records')
    log.info('*' * 20)

    minerrloc = np.argmin(calc_errs)
    FAS_target_raw_final = FAStarget[:, minerrloc]
    
    PSD_raw_all = np.zeros((len(freqs_raw), neqsPSD))
    sdfin = np.zeros(neqsPSD)

    if use_multiprocessing:
        psd_results_list = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(_single_comp_psd_worker, nt, envelope, m, dt, FAS_target_raw_final, fs) 
                for _ in range(neqsPSD)
            ]
            for f in concurrent.futures.as_completed(futures):
                psd_results_list.append(f.result())
        
        for q, (psd_raw, sd) in enumerate(psd_results_list):
            PSD_raw_all[:, q] = psd_raw
            sdfin[q] = sd
    else:
        for q in range(neqsPSD):
            psd_raw, sd = _single_comp_psd_worker(nt, envelope, m, dt, FAS_target_raw_final, fs)
            PSD_raw_all[:, q] = psd_raw
            sdfin[q] = sd

    sdfinmean = np.mean(sdfin)
    log.info(f'Target SD5-75: {sd575:.2f}s, Actual Mean SD5-75: {sdfinmean:.2f}s')

    # --- 6. Efficient "Smooth First" Statistics Strategy ---
    log.info("Applying efficient 'Smooth First' matrix strategy for ensemble statistics...")
    
    PSD_smooth_all = None
    
    if psd_smoothing_method == 'konno_ohmachi':
        if prefer_pykooh and PYKOOH_AVAILABLE:
            # Use PyKooh C++ implementation
            smoother_obj = pykooh.CachedSmoother(freqs_raw, freqs_des, bandwidth=psd_smoothing_coeff)
            W_smooth = smoother_obj._weights 
            PSD_smooth_all = W_smooth.T @ np.nan_to_num(PSD_raw_all)
            
        else:
            # Use sparse matrix implementation
            W_smooth = _konno_ohmachi_1998_sparse_matrix(freqs_des, freqs_raw, b=psd_smoothing_coeff)
            PSD_smooth_all = W_smooth @ np.nan_to_num(PSD_raw_all)
            
    elif psd_smoothing_method == 'variable_window':
        PSD_smooth_all = np.zeros((len(freqs_des), neqsPSD))
        for i in range(neqsPSD):
            PSD_smooth_all[:, i] = _smooth_boxcar_variable(freqs_des, freqs_raw, PSD_raw_all[:, i], percentage=psd_smoothing_coeff)
    else:
        PSD_smooth_all = log_interp(freqs_des, freqs_raw, PSD_raw_all)

    # Geometric Mean on Smoothed Data (This is the Target PSD)
    PSD_target_smooth_mean = np.exp(np.mean(np.log(PSD_smooth_all + 1e-30), axis=1))
    
    # Bounds on Smoothed Data
    PSD_low_smooth, PSD_high_smooth, sigma_ln = _calc_lognormal_stats(PSD_smooth_all, PSD_target_smooth_mean)
    
    # FAS Smoothing (Simple deterministic smoothing of the final target FAS)
    FAS_target_smooth = _apply_smoothing(freqs_des, freqs_raw, FAS_target_raw_final, 
                                         fas_smoothing_method, fas_smoothing_coeff, prefer_pykooh, "FAS")

    # --- 7. Package Results ---
    results = {
        'type': 'single_comp',
        'workname': workname,
        'freqs_smooth': freqs_des,
        
        # Main Results (Smoothed)
        'target_psd': PSD_target_smooth_mean,
        'target_fas': FAS_target_smooth,
        'target_psd_low_smooth': PSD_low_smooth,
        'target_psd_high_smooth': PSD_high_smooth,
        
        # Full Smoothed Ensemble
        'all_psd_smooth': PSD_smooth_all,

        # Raw Data
        'freqs_raw': freqs_raw,
        'target_psd_raw_mean': np.mean(PSD_raw_all, axis=1), 
        'target_fas_raw_mean': FAS_target_raw_final,
        'psd_raw_low': np.zeros_like(freqs_raw), 
        'psd_raw_high': np.zeros_like(freqs_raw),
        
        # Iteration/Info
        'sd575_target': sd575,
        'sd575_mean_actual': sdfinmean,
        'target_freqs': f_or,
        'target_psa': ds_or,
        'calc_freqs': f_psa,
        'calc_psa_mean_initial': PSAavg[:, 0],
        'calc_psa_mean_final': PSAavg[:, minerrloc],
        'fas_initial_raw': FAStarget[:, 0],
        'iteration_errors': calc_errs,
        'n_iterations': k_final + 1,
        'allow_err': allow_err
    }
    
    return results


# =============================================================================
# PUBLIC API: RotDnn (ORIENTATION-INDEPENDENT) GENERATOR
# =============================================================================

def generate_rotdnn_target_psd(
    target_freqs: np.ndarray,
    target_psa: np.ndarray,
    sd575_gm: float,
    freqs_des: np.ndarray,
    workname: str = 'RotDnnTargetFASPSD',
    nn_psa: int = 100,
    nn_analysis: int = 100,
    sd_ratio: float = 1.3,
    smoothing_method: Optional[str] = 'konno_ohmachi',
    smoothing_coeff: float = 20.0,
    zi: float = 0.05,
    F1: float = 0.2,
    F2: float = 50.0,
    allow_err: float = 2.5,
    neqsPSD: int = 1000,
    use_multiprocessing: bool = True,
    prefer_pykooh: bool = True
) -> Dict[str, Any]:
    """
    Generates orientation-independent (RotDnn) Target PSD and FAS functions compatible
    with a RotDnn design spectrum.

    Based on the methodology described in Montejo (2026). This function generates two
    orthogonal horizontal components (H1, H2) with different durations such that their 
    combined RotDnn response matches the target.

    Parameters
    ----------
    target_freqs : np.ndarray
        Frequencies of the target RotDnn response spectrum [Hz].
    target_psa : np.ndarray
        Amplitudes of the target RotDnn response spectrum (usually in g).
    sd575_gm : float
        Target Geometric Mean Significant Duration (SD5-75) [s].
    freqs_des : np.ndarray
        Desired output frequency vector for the final smoothed PSD/FAS [Hz].
    workname : str, optional
        Label for the analysis (default: 'RotDnnTargetFASPSD').
    nn_psa : int, optional
        Percentile of the input target PSA (e.g., 50 for RotD50, 100 for RotD100).
    nn_analysis : int, optional
        Percentile for the output FAS/PSD calculation (default: 100).
    sd_ratio : float, optional
        Ratio of durations between the two orthogonal components (default: 1.3).
    smoothing_method : str, optional
        Smoothing method for FAS/PSD ('konno_ohmachi', 'variable_window').
    smoothing_coeff : float, optional
        Smoothing coefficient (b-value or percentage).
    zi : float, optional
        Damping ratio (default: 0.05).
    F1, F2 : float, optional
        Frequency range for error calculation [Hz] (defaults: 0.2, 50.0).
    allow_err : float, optional
        Maximum allowable error for PSA matching [%] (default: 2.5).
    neqsPSD : int, optional
        Number of synthetic record pairs for final statistics (default: 1000).
    use_multiprocessing : bool, optional
        Enable parallel processing (default: True).
    prefer_pykooh : bool, optional
        Enable optimized C++ smoothing if available (default: True).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing results:
        - 'target_fas_rotdnn': Smoothed Target FAS (RotDnn).
        - 'target_psd_rotdnn': Smoothed Target PSD (RotDnn).
        - 'sd575_gm_mean_actual': Actual geometric mean duration of the suite.
        - 'all_psd_smooth': Full matrix of smoothed PSDs for variability analysis.
        - ... (other intermediate data).
    """
    
    fs = 200.0
    dt = 1.0 / fs
    f_or = target_freqs
    ds_or = target_psa

    # --- 1. Create Time Envelopes ---
    # Duration for each component is adjusted so that sqrt(H1 * H2) = sd575_gm
    if sd_ratio < 1: sd_ratio = 1.0 / sd_ratio
    sd90 = (sd575_gm**2 / sd_ratio)**0.5
    sd0 = sd_ratio * sd90
    
    log.info(f"Target GM SD5-75: {sd575_gm:.2f}s. H1={sd0:.2f}s, H2={sd90:.2f}s")

    tf0 = 3.54 * sd0
    nt0 = int(tf0 / dt) + 1
    if nt0 % 2 != 0: nt0 += 1
    envelope0 = _saragoni_hart_w(nt0, eps=0.2, n=0.2, tn=0.6)

    tf90 = 3.54 * sd90
    nt90 = int(tf90 / dt) + 1
    if nt90 % 2 != 0: nt90 += 1
    envelope90 = _saragoni_hart_w(nt90, eps=0.2, n=0.2, tn=0.6)

    nt = max(nt0, nt90)
    # Pad shorter envelope to match length
    if nt0 < nt: envelope0 = np.pad(envelope0, (0, nt - nt0), 'constant')
    if nt90 < nt: envelope90 = np.pad(envelope90, (0, nt - nt90), 'constant')

    # --- 2. Create Frequency Vectors ---
    sets = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    nsets = len(sets)
    m = np.arange(0, np.ceil(nt / 2) + 1, dtype=int)
    freqs_raw = m * fs / nt
    nyquist_freq = fs / 2

    # Calculation frequencies for PSA matching
    f_psa = np.hstack((np.array([0.04, 0.06, 0.08]), 
                       np.geomspace(0.1, 50, 100),
                       np.array([55, 60, 70, 80, 90, 100])))
    f_psa = f_psa[f_psa <= nyquist_freq]
    f_psa[-1] = nyquist_freq
    if f_psa[0] < freqs_raw[1]: f_psa[0] = 0.99 * freqs_raw[1]
    T_psa = 1.0 / f_psa

    # --- 3. Validate and Resample Target Spectrum ---
    psa_target_calc = log_interp(f_psa, f_or, ds_or)
    locs = np.where((f_psa >= F1) & (f_psa <= F2))[0]
    
    # --- 4. Initial Component FAS Guess ---
    # Apply Montejo & Vidot-Vega (2017) to each component based on its duration
    psa_target_raw = log_interp(freqs_raw[1:], f_psa, psa_target_calc)
    psa_target_raw = np.concatenate(([0], psa_target_raw))

    ratio0 = _get_fas_psa_ratio(freqs_raw[1:], sd0)
    TFAS0 = psa_target_raw[1:] * ratio0
    TFAS0 = np.concatenate(([0], TFAS0))
    
    ratio90 = _get_fas_psa_ratio(freqs_raw[1:], sd90)
    TFAS90 = psa_target_raw[1:] * ratio90
    TFAS90 = np.concatenate(([0], TFAS90))

    PSAavg = np.zeros((len(T_psa), nsets))
    FAS0target = np.zeros((len(freqs_raw), nsets))
    FAS90target = np.zeros((len(freqs_raw), nsets))
    FAS0target[:, 0] = TFAS0
    FAS90target[:, 0] = TFAS90
    calc_errs = np.zeros(nsets)

    log.info('*' * 20)
    log.info(f'Generating RotD{nn_psa} spectrum compatible FAS')
    log.info(f'Target error: {allow_err:.2f}%, max # of iters.: {nsets}')
    log.info('*' * 20)

    # --- 5. Iterative FAS-to-PSA Matching ---
    k_final = 0
    for k in range(nsets):
        n_records_in_set = sets[k]
        log.info(f"Starting iteration {k+1}/{nsets}...")
        TaFAS0 = FAS0target[:, k]
        TaFAS90 = FAS90target[:, k]

        if use_multiprocessing:
            PSA_results_list = []
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(_rotdnn_psa_worker, nt, dt, envelope0, envelope90, TaFAS0, TaFAS90, T_psa, zi, nn_psa) 
                    for _ in range(n_records_in_set)
                ]
                for f in concurrent.futures.as_completed(futures):
                    PSA_results_list.append(f.result())
            PSA_set = np.array(PSA_results_list).T
        else:
            PSA_set = np.zeros((len(T_psa), n_records_in_set))
            for q in range(n_records_in_set):
                PSA_set[:, q] = _rotdnn_psa_worker(nt, dt, envelope0, envelope90, TaFAS0, TaFAS90, T_psa, zi, nn_psa)

        # GEOMETRIC MEAN (Median)
        PSAavg[:, k] = np.exp(np.mean(np.log(PSA_set + 1e-30), axis=1))
        
        diflimits = np.abs(psa_target_calc[locs] - PSAavg[locs, k]) / psa_target_calc[locs]
        calc_errs[k] = np.mean(diflimits) * 100
        k_final = k
        log.info(f'Iteration {k+1}: error = {calc_errs[k]:.2f}%')
        if calc_errs[k] < allow_err:
            log.info(f'Error satisfied at iteration {k+1}.')
            break
        elif k != nsets - 1:
            # Update both component FAS using the ratio of Target PSA / Current PSA
            PSAavg_interp = log_interp(freqs_raw[1:], f_psa, PSAavg[:, k])
            with np.errstate(divide='ignore', invalid='ignore'):
                factor = psa_target_raw[1:] / PSAavg_interp
            factor[np.isnan(factor) | np.isinf(factor)] = 1.0
            FAS0target[1:, k + 1] = factor * FAS0target[1:, k]
            FAS90target[1:, k + 1] = factor * FAS90target[1:, k]

    PSAavg = PSAavg[:, :k_final + 1]
    FAS0target = FAS0target[:, :k_final + 1]
    FAS90target = FAS90target[:, :k_final + 1]
    calc_errs = calc_errs[:k_final + 1]

    # --- 6. Find Target RotDnn PSD/FAS from final component FAS ---
    log.info('*' * 20)
    log.info(f'Now generating target RotD{nn_analysis} PSD/FAS using {neqsPSD} records')
    log.info('*' * 20)
    
    minerrloc = np.argmin(calc_errs)
    TFAS0fin = FAS0target[:, minerrloc]
    TFAS90fin = FAS90target[:, minerrloc]
    PSAfin_calc = PSAavg[:, minerrloc]

    SD_rot_all = np.zeros((180, neqsPSD))
    PSD_rotdnn_raw_all = np.zeros((len(freqs_raw), neqsPSD))
    FAS_rotdnn_raw_all = np.zeros((len(freqs_raw), neqsPSD))

    if use_multiprocessing:
        results_list = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(_rotdnn_psd_fas_worker, nt, dt, envelope0, envelope90, TFAS0fin, TFAS90fin, nn_analysis) 
                for _ in range(neqsPSD)
            ]
            for f in concurrent.futures.as_completed(futures):
                results_list.append(f.result())
        for q, (sd_rot, psd_rotdnn, fas_rotdnn) in enumerate(results_list):
            SD_rot_all[:, q] = sd_rot
            PSD_rotdnn_raw_all[:, q] = psd_rotdnn
            FAS_rotdnn_raw_all[:, q] = fas_rotdnn
    else:
        for q in range(neqsPSD):
            sd_rot, psd_rotdnn, fas_rotdnn = _rotdnn_psd_fas_worker(nt, dt, envelope0, envelope90, TFAS0fin, TFAS90fin, nn_analysis)
            SD_rot_all[:, q] = sd_rot
            PSD_rotdnn_raw_all[:, q] = psd_rotdnn
            FAS_rotdnn_raw_all[:, q] = fas_rotdnn

    # --- 6b. Efficient "Smooth First" Statistics Strategy ---
    log.info("Applying efficient 'Smooth First' matrix strategy for RotDnn ensemble statistics...")
    
    PSD_rotdnn_smooth_all = None
    FAS_rotdnn_smooth_all = None
    
    # 1. Build Operator (Once) and Apply
    if smoothing_method == 'konno_ohmachi':
        if prefer_pykooh and PYKOOH_AVAILABLE:
            smoother_obj = pykooh.CachedSmoother(freqs_raw, freqs_des, bandwidth=smoothing_coeff)
            W_smooth = smoother_obj._weights
            # Apply to PSD and FAS
            PSD_rotdnn_smooth_all = W_smooth.T @ np.nan_to_num(PSD_rotdnn_raw_all)
            FAS_rotdnn_smooth_all = W_smooth.T @ np.nan_to_num(FAS_rotdnn_raw_all)
        else:
            W_smooth = _konno_ohmachi_1998_sparse_matrix(freqs_des, freqs_raw, b=smoothing_coeff)
            PSD_rotdnn_smooth_all = W_smooth @ np.nan_to_num(PSD_rotdnn_raw_all)
            FAS_rotdnn_smooth_all = W_smooth @ np.nan_to_num(FAS_rotdnn_raw_all)
            
    elif smoothing_method == 'variable_window':
        PSD_rotdnn_smooth_all = np.zeros((len(freqs_des), neqsPSD))
        FAS_rotdnn_smooth_all = np.zeros((len(freqs_des), neqsPSD))
        for i in range(neqsPSD):
            PSD_rotdnn_smooth_all[:, i] = _smooth_boxcar_variable(freqs_des, freqs_raw, PSD_rotdnn_raw_all[:, i], percentage=smoothing_coeff)
            FAS_rotdnn_smooth_all[:, i] = _smooth_boxcar_variable(freqs_des, freqs_raw, FAS_rotdnn_raw_all[:, i], percentage=smoothing_coeff)
    else:
        PSD_rotdnn_smooth_all = log_interp(freqs_des, freqs_raw, PSD_rotdnn_raw_all)
        FAS_rotdnn_smooth_all = log_interp(freqs_des, freqs_raw, FAS_rotdnn_raw_all)

    # 2. Compute Geometric Means and Bounds on SMOOTHED Data
    PSD_target_smooth_mean = np.exp(np.mean(np.log(PSD_rotdnn_smooth_all + 1e-30), axis=1))
    FAS_target_smooth_mean = np.exp(np.mean(np.log(FAS_rotdnn_smooth_all + 1e-30), axis=1))

    # Bounds
    PSD_low_smooth, PSD_high_smooth, _ = _calc_lognormal_stats(PSD_rotdnn_smooth_all, PSD_target_smooth_mean)
    FAS_low_smooth, FAS_high_smooth, _ = _calc_lognormal_stats(FAS_rotdnn_smooth_all, FAS_target_smooth_mean)

    # GM Duration Stats
    with np.errstate(invalid='ignore'):
        sd_gm_mean = np.mean(np.sqrt(SD_rot_all[0, :] * SD_rot_all[90, :]))
    
    log.info(f'Target GM SD5-75: {sd575_gm:.2f}s. Actual records GM SD5-75: {sd_gm_mean:.2f}s')

    # --- 8. Package Results ---
    results = {
        'type': 'rotdnn',
        'workname': workname,
        'freqs_smooth': freqs_des,
        
        # Main Results (Smoothed)
        'target_fas_rotdnn': FAS_target_smooth_mean,
        'target_psd_rotdnn': PSD_target_smooth_mean,
        
        # Smooth Bounds
        'target_fas_low_smooth': FAS_low_smooth,
        'target_fas_high_smooth': FAS_high_smooth,
        'target_psd_low_smooth': PSD_low_smooth,
        'target_psd_high_smooth': PSD_high_smooth,
        
        # Full Smoothed Ensemble (for plotting)
        'all_psd_smooth': PSD_rotdnn_smooth_all,
        'all_fas_smooth': FAS_rotdnn_smooth_all,

        # Raw Data & Bounds
        'freqs_raw': freqs_raw,
        'target_fas_rotdnn_raw_mean': np.mean(FAS_rotdnn_raw_all, axis=1),
        'target_psd_rotdnn_raw_mean': np.mean(PSD_rotdnn_raw_all, axis=1),
        
        # Full Raw Data (for plotting ind lines)
        'all_fas_rotdnn_raw': FAS_rotdnn_raw_all,
        'all_psd_rotdnn_raw': PSD_rotdnn_raw_all,
        
        # Iteration/Info
        'sd575_gm_target': sd575_gm,
        'sd575_gm_mean_actual': sd_gm_mean,
        'sd_all_records': SD_rot_all,
        'nn_psa': nn_psa,
        'nn_analysis': nn_analysis,
        'target_freqs': f_or,
        'target_psa': ds_or,
        'calc_freqs': f_psa,
        'calc_psa_mean_final': PSAfin_calc,
        'fas_h1_final_raw': TFAS0fin,
        'fas_h2_final_raw': TFAS90fin,
        'iteration_errors': calc_errs,
        'n_iterations': k_final + 1,
        'allow_err': allow_err
    }
    
    return results


# =============================================================================
# PUBLIC API: PLOTTING & FILE I/O
# =============================================================================

def save_target_psd_file(results: Dict[str, Any], filepath: str) -> None:
    """
    Saves the generated Target FAS and PSD results to a formatted text file.

    Parameters
    ----------
    results : Dict[str, Any]
        The results dictionary returned by `generate_single_comp_target_psd` or 
        `generate_rotdnn_target_psd`.
    filepath : str
        The path where the output text file will be saved.
    """
    try:
        freqs = results['freqs_smooth']
        
        if results['type'] == 'rotdnn':
            fas_mean = results['target_fas_rotdnn']
            psd_mean = results['target_psd_rotdnn']
            fas_low = results['target_fas_low_smooth']
            fas_high = results['target_fas_high_smooth']
            psd_low = results['target_psd_low_smooth']
            psd_high = results['target_psd_high_smooth']
            nn = results.get('nn_analysis', 'nn')
            header = (f'freq[Hz]  -  FASRotD{nn}_mean  -  FAS_minus_1std  -  FAS_plus_1std  -  '
                      f'PSDRotD{nn}_mean  -  PSD_minus_1std  -  PSD_plus_1std')
            outputfile = np.vstack((freqs, fas_mean, fas_low, fas_high, psd_mean, psd_low, psd_high)).T
            
        else: # single_comp
            fas_mean = results['target_fas']
            psd_mean = results['target_psd']
            psd_low = results['target_psd_low_smooth']
            psd_high = results['target_psd_high_smooth']
            # Only save PSD bounds (FAS is deterministic/single-path in this method)
            header = ('freq[Hz]  -  FAS_mean  -  PSD_mean  -  PSD_minus_1std  -  PSD_plus_1std')
            outputfile = np.vstack((freqs, fas_mean, psd_mean, psd_low, psd_high)).T

        np.savetxt(filepath, outputfile, header=header, fmt='%.8e')
        log.info(f"Successfully saved target PSD/FAS file to: {filepath}")
        
    except Exception as e:
        log.error(f"Error saving file to {filepath}: {e}")
        raise

def plot_target_psd_results(
    results: Dict[str, Any],
    target_spectrum_name: str = "",
    units: str = 'g',
    max_plot_records: Optional[int] = None
) -> plt.Figure:
    """
    Generates a summary plot of the analysis results.

    Parameters
    ----------
    results : Dict[str, Any]
        The results dictionary returned by the generator functions.
    target_spectrum_name : str, optional
        Name of the spectrum to display in the plot title.
    units : str, optional
        Units of acceleration (e.g., 'g', 'm/s2').
    max_plot_records : int, optional
        Maximum number of individual background traces to plot (to reduce file size/lag).
        If None, plots all.

    Returns
    -------
    plt.Figure
        The Matplotlib figure object containing the subplots.
    """
    
    if results['type'] == 'single_comp':
        return _plot_single_comp_summary(results, target_spectrum_name, units, max_plot_records)
    elif results['type'] == 'rotdnn':
        return _plot_rotdnn_summary(results, target_spectrum_name, units, max_plot_records)
    else:
        raise ValueError(f"Unknown results type: {results['type']}")


# =============================================================================
# INTERNAL: PLOTTING HELPERS
# =============================================================================

def _plot_single_comp_summary(results, target_spectrum_name, units, max_plot_records=None):
    """Internal helper to plot single-component results."""
    mpl.rcParams['font.size'] = 9
    mpl.rcParams['legend.frameon'] = False
    
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 8))
    fig.suptitle(f'Design Spectrum: {target_spectrum_name}\n'
                 f'Target SD5-75: {results["sd575_target"]:.1f}s - '
                 f'Average SD5-75: {results["sd575_mean_actual"]:.1f}s')

    aux = np.arange(1, results['n_iterations'] + 1)

    # --- Subplot 1: Error ---
    ax = axs[0, 0]
    ax.plot(aux, results['iteration_errors'], '--o', mfc='white')
    ax.hlines(results['allow_err'], aux[0], aux[-1], colors='red')
    ax.text(1, results['allow_err'] + 0.1, f'target: {results["allow_err"]:.2f}%')
    ax.set_xticks(aux)
    ax.set_xlabel('Iteration #')
    ax.set_ylabel('PSA Error [%]')

    # --- Subplot 2: PSA ---
    ax = axs[0, 1]
    ax.semilogx(results['target_freqs'], results['target_psa'], color='silver', lw=3, label='Target')
    ax.semilogx(results['calc_freqs'], results['calc_psa_mean_initial'], color='darkred', label='First It.')
    # Updated label to emphasize Geometric Mean
    ax.semilogx(results['calc_freqs'], results['calc_psa_mean_final'], color='black', label='Final (Geom. Mean)')
    ax.set_xlim(0.1, 100)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('F [Hz]')
    ax.set_ylabel(f'PSA [{units}]')
    ax.legend()

    # --- Common Data for FAS/PSD ---
    freqs_raw = results['freqs_raw']
    freqs_smooth = results['freqs_smooth']
    
    # --- Subplot 3: FAS (Deterministic - No Cloud, No Bounds) ---
    ax = axs[1, 0]
    # For single comp, FAS is effectively deterministic/single path, but plot mean for consistency
    ax.loglog(freqs_raw, results['target_fas_raw_mean'], color='black', lw=1, 
              label='Geom. Mean', zorder=3)
    # Changed label to Geom. Mean
    ax.loglog(freqs_smooth, results['target_fas'], color='salmon', lw=1.5, 
              label='Smooth', zorder=3)
    
    ax.set_xlim(freqs_smooth.min(), freqs_smooth.max())
    if np.any(results['target_fas'] > 0):
        ax.set_ylim(bottom=np.nanmin(results['target_fas'][results['target_fas']>0])*0.5)
    ax.set_xlabel('F [Hz]')
    ax.set_ylabel(f'FAS [{units} * s]')
    ax.legend(loc='lower left')

    # --- Subplot 4: PSD (Smoothed Bounds + Individual Clouds) ---
    ax = axs[1, 1]
    
    # 0. Individual Smoothed Records (Background Cloud)
    if 'all_psd_smooth' in results:
        n_total = results['all_psd_smooth'].shape[1]
        n_plot = n_total if max_plot_records is None else min(n_total, max_plot_records)
        
        # Plot Cloud (Transparent)
        ax.loglog(freqs_smooth, results['all_psd_smooth'][:, 0:n_plot], color='gray', lw=0.2, alpha=0.3, zorder=1)
        # Proxy Legend (Opaque)
        ax.plot([], [], color='gray', lw=1, alpha=1.0, label='ind. motions (smooth)')

    # 1. Smoothed Bounds (Shaded Blue) - zorder=2
    # These now represent the bounds of the smoothed trends (aleatory variability)
    ax.fill_between(freqs_smooth, results['target_psd_low_smooth'], results['target_psd_high_smooth'], 
                    color='cornflowerblue', alpha=0.6, label=r'$\pm 1\sigma$', zorder=2)
    
    # 2. Geometric Mean (Target) - zorder=3
    # Changed label to Geom. Mean
    ax.loglog(freqs_smooth, results['target_psd'], color='salmon', lw=1.5, 
              label='Geom. Mean', zorder=3)

    ax.set_xlim(freqs_smooth.min(), freqs_smooth.max())
    
    # Y-Limits based on Smoothed Bounds (Robust to outliers)
    if np.any(results['target_psd_high_smooth']):
        max_bound = np.nanmax(results['target_psd_high_smooth'])
        min_bound = np.nanmin(results['target_psd_low_smooth'])
        # Handle zeros in low bound if any
        if min_bound <= 0: min_bound = np.nanmin(results['target_psd'][results['target_psd']>0]) * 0.1
        
        ax.set_ylim(bottom=min_bound*0.5, top=max_bound*1.5)
    
    ax.set_xlabel('F [Hz]')
    ax.set_ylabel(f'PSD [{units}² * s]')
    ax.legend(loc='lower left')

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig

def _plot_rotdnn_summary(results, target_spectrum_name, units, max_plot_records=None):
    """Internal helper to plot RotDnn results."""
    mpl.rcParams['font.size'] = 9
    mpl.rcParams['legend.frameon'] = False
    
    # --- FIGURE 1: Summary ---
    fig, axs = plt.subplots(2, 3, figsize=(15, 8)) # Use 'fig' for single figure
    fig.suptitle(f'Target: {target_spectrum_name}\n'
                 f'Target GM SD5-75: {results["sd575_gm_target"]:.1f}s - '
                 f'Actual GM SD5-75: {results["sd575_gm_mean_actual"]:.1f}s')

    # Plot 1: Error
    ax = axs[0, 0]
    aux = np.arange(1, results['n_iterations'] + 1)
    ax.plot(aux, results['iteration_errors'], '--o', color='black', ms=4, lw=1, mfc='pink')
    ax.hlines(results['allow_err'], aux[0], aux[-1], colors='red')
    ax.text(1, results['allow_err'] + 0.1, f'target: {results["allow_err"]:.2f}%')
    ax.set_xticks(aux)
    ax.set_xlabel('Iteration #')
    ax.set_ylabel(f'PSA RotD{results["nn_psa"]} Error [%]')

    # Plot 2: PSA
    ax = axs[0, 1]
    ax.semilogx(results['target_freqs'], results['target_psa'], color='black', lw=1.5, label='Target')
    ax.semilogx(results['calc_freqs'], results['calc_psa_mean_final'], '-', color='blueviolet', lw=1, label='Set Geom. Mean')
    ax.set_xlim(0.1, 100)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(f'PSA RotD{results["nn_psa"]} [{units}]')
    ax.set_xlabel('F [Hz]')
    ax.legend(handlelength=1)

    # Plot 3: Component FAS
    ax = axs[0, 2]
    ax.semilogx(results['freqs_raw'], results['fas_h1_final_raw'], color='cornflowerblue', lw=1, label='Final FAS H1')
    ax.semilogx(results['freqs_raw'], results['fas_h2_final_raw'], color='salmon', lw=1, label='Final FAS H2')
    ax.set_xlim(0.1, 100)
    ax.set_ylim(bottom=0)
    ax.set_ylabel(f'Target Component FAS [{units} * s]')
    ax.set_xlabel('F [Hz]')
    ax.legend()

    # --- ROW 2: VALIDATION ---
    nn_an = results['nn_analysis']
    freqs_raw = results['freqs_raw']
    freqs_smooth = results['freqs_smooth']
    
    # Helper to clean up plotting individual lines with limit + Proxy Legend
    def plot_cloud_rotdnn(ax, x, Y_matrix):
        n_total = Y_matrix.shape[1]
        n_plot = n_total if max_plot_records is None else min(n_total, max_plot_records)

        # Plot Cloud (Transparent)
        ax.loglog(x, Y_matrix[:, 0:n_plot], color='gray', lw=0.2, alpha=0.3, zorder=1)
        # Proxy Legend (Opaque)
        ax.plot([], [], color='gray', lw=1, alpha=1.0, label='ind. motions (smooth)')

    # Plot 4: FAS RotDnn
    ax = axs[1, 0]
    if 'all_fas_smooth' in results:
        plot_cloud_rotdnn(ax, freqs_smooth, results['all_fas_smooth'])
    else:
        plot_cloud_rotdnn(ax, freqs_raw, results['all_fas_rotdnn_raw'])
    
    ax.fill_between(freqs_smooth, results['target_fas_low_smooth'], results['target_fas_high_smooth'], 
                    color='cornflowerblue', alpha=0.6, label=r'$\pm 1\sigma$', zorder=2)
    
    # Changed Label to 'Geom. Mean'
    ax.loglog(freqs_smooth, results['target_fas_rotdnn'], '-', color='salmon', lw=1.5, label='Geom. Mean', zorder=3)
    
    ax.set_xlim(freqs_smooth.min(), freqs_smooth.max())
    
    # Y-Limits based on Smoothed Bounds (Robust to outliers)
    if np.any(results['target_fas_high_smooth']):
        max_bound = np.nanmax(results['target_fas_high_smooth'])
        min_bound = np.nanmin(results['target_fas_low_smooth'])
        if min_bound <= 0: min_bound = np.nanmin(results['target_fas_rotdnn'][results['target_fas_rotdnn']>0]) * 0.1
        ax.set_ylim(bottom=min_bound*0.5, top=max_bound*1.5)

    ax.set_xlabel('F [Hz]')
    ax.set_ylabel(f'FASRotD{nn_an} [{units} * s]')
    ax.legend(loc='lower left')

    # Plot 5: PSD RotDnn
    ax = axs[1, 1]
    if 'all_psd_smooth' in results:
        plot_cloud_rotdnn(ax, freqs_smooth, results['all_psd_smooth'])
    else:
        plot_cloud_rotdnn(ax, freqs_raw, results['all_psd_rotdnn_raw'])

    ax.fill_between(freqs_smooth, results['target_psd_low_smooth'], results['target_psd_high_smooth'], 
                    color='cornflowerblue', alpha=0.6, label=r'$\pm 1\sigma$', zorder=2)
    
    # Changed Label to 'Geom. Mean'
    ax.loglog(freqs_smooth, results['target_psd_rotdnn'], '-', color='salmon', lw=1.5, label='Geom. Mean', zorder=3)
    
    ax.set_xlim(freqs_smooth.min(), freqs_smooth.max())
    
    # Y-Limits based on Smoothed Bounds (Robust to outliers)
    if np.any(results['target_psd_high_smooth']):
        max_bound = np.nanmax(results['target_psd_high_smooth'])
        min_bound = np.nanmin(results['target_psd_low_smooth'])
        if min_bound <= 0: min_bound = np.nanmin(results['target_psd_rotdnn'][results['target_psd_rotdnn']>0]) * 0.1
        ax.set_ylim(bottom=min_bound*0.5, top=max_bound*1.5)

    ax.set_xlabel('F [Hz]')
    ax.set_ylabel(f'PSDRotD{nn_an} [{units}² * s]')
    
    # Plot 6: Duration
    ax = axs[1, 2]
    theta = np.arange(0, 180)
    
    n_total_dur = results['sd_all_records'].shape[1]
    n_plot_dur = n_total_dur if max_plot_records is None else min(n_total_dur, max_plot_records)
    
    # Plot Cloud
    ax.plot(theta, results['sd_all_records'][:, 0:n_plot_dur], color='gray', lw=0.2, alpha=0.3, zorder=1)
    # Proxy Legend
    ax.plot([], [], color='gray', lw=1, alpha=1.0, label='ind. motions')
    
    # Target
    ax.hlines(results['sd575_gm_target'], theta[0], theta[-1], color='blueviolet', lw=2, label='target', zorder=3)
    # Actual GM Mean
    ax.hlines(results['sd575_gm_mean_actual'], theta[0], theta[-1], linestyles='dashed', color='black', lw=1.5, label='avg. geom. mean', zorder=3)
    
    ax.legend()
    ax.set_xticks([0, 45, 90, 135, 179])
    ax.set_xlabel(r'Rotation Angle $\theta$')
    ax.set_ylabel(r'$SD_{5-75}$ [s]')

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    return fig

# =============================================================================
# INTERNAL: MATH & SIGNAL PROCESSING HELPERS
# =============================================================================

def _calc_lognormal_stats(data_matrix, mean_vector=None):
    """
    Computes log-normal statistics.
    Returns: Lower Bound (-1 sigma), Upper Bound (+1 sigma), Sigma_ln
    
    If mean_vector is provided, bounds are centered on it: mean * exp(+/- sigma).
    Otherwise, computes the geometric mean from data.
    """
    # Avoid log(0)
    data_safe = np.where(data_matrix > 1e-30, data_matrix, 1e-30)
    
    # 1. Sigma of the logs (variability in log space)
    sigma_ln = np.std(np.log(data_safe), axis=1, ddof=1)
    
    # 2. Determine Mean (Center)
    if mean_vector is not None:
        mean_safe = np.where(mean_vector > 1e-30, mean_vector, 1e-30)
        center = mean_safe
    else:
        # Geometric mean
        center = np.exp(np.mean(np.log(data_safe), axis=1))

    lower = center * np.exp(-sigma_ln)
    upper = center * np.exp(sigma_ln)
    
    return lower, upper, sigma_ln

@jit(nopython=True, cache=True)
def _get_fas_psa_ratio(f, sd575):
    """
    Computes empirical FAS/PSA ratio based on Montejo & Vidot-Vega (2017).
    """
    aa75, ab75, ac75 = 0.0512, 0.4920, 0.1123
    ba75, bb75, bc75 = -0.5869, -0.2650, -0.4580
    ratio = (aa75 * sd575**ab75 + ac75) * f**(ba75 * sd575**bb75 + bc75)
    return ratio

@jit(nopython=True, cache=True)
def _saragoni_hart_w(npoints, eps=0.25, n=0.4, tn=0.6):
    """
    Generates a Saragoni-Hart time modulation window.
    """
    e_val = 2.718281828459045
    if n <= 0: n = 1e-9
    if eps <= 0: eps = 1e-9
    b = -(eps * np.log(n)) / (1.0 + eps * (np.log(eps) - 1.0))
    c = b / eps
    a = (e_val / eps)**b
    t = np.linspace(0, 1, npoints)
    if tn == 0: tn = 1.0 
    t_tn = t / tn
    w = a * (t_tn)**b * np.exp(-c * (t_tn))
    return w

def _apply_smoothing(output_freqs, raw_freqs, raw_spectrum, method, coeff, prefer_pykooh, name="Spectrum"):
    """
    Applies frequency smoothing to a spectrum.
    """
    if method == 'konno_ohmachi':
        if prefer_pykooh and PYKOOH_AVAILABLE:
            smooth_spectrum = pykooh.smooth(output_freqs, raw_freqs, raw_spectrum, coeff)
        else:
            smooth_spectrum = _konno_ohmachi_1998_downsample(output_freqs, raw_freqs, raw_spectrum, b=coeff)
    elif method == 'variable_window':
        smooth_spectrum = _smooth_boxcar_variable(output_freqs, raw_freqs, raw_spectrum, percentage=coeff)
    elif method is None or method == 'none':
        smooth_spectrum = log_interp(output_freqs, raw_freqs, raw_spectrum)
    else:
        raise ValueError(f"Unknown smoothing_method: {method}")
    return smooth_spectrum


# =============================================================================
# INTERNAL: PARALLEL WORKERS
# =============================================================================

def _single_comp_psa_worker(nt, envelope, m, dt, TaFAS, T, zi):
    """Worker to compute PSA for a single synthetic motion."""
    so = np.random.randn(nt) * envelope
    FSo = np.fft.fft(so)
    FASo = np.abs(FSo[m]) * dt
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ff = TaFAS / FASo
    ff[np.isnan(ff) | np.isinf(ff)] = 0.0
    
    fsymm = np.concatenate((ff, ff[-2:0:-1]))
    FS = FSo * fsymm
    sc = np.fft.ifft(FS).real
    
    PSArecord, _, _ = compute_spectrum_fd(T, sc, zi, dt)
    return PSArecord


def _single_comp_psd_worker(nt, envelope, m, dt, TFASfin, fs):
    """Worker to compute PSD and Duration for a single synthetic motion."""
    so = np.random.randn(nt) * envelope
    FSo = np.fft.fft(so)
    FASo = np.abs(FSo[m]) * dt

    with np.errstate(divide='ignore', invalid='ignore'):
        ff = TFASfin / FASo
    ff[np.isnan(ff) | np.isinf(ff)] = 0.0

    fsymm = np.concatenate((ff, ff[-2:0:-1]))
    FS = FSo * fsymm
    s = np.fft.ifft(FS).real
    
    (freqs, psd_raw, sd, ai, _, _) = calculate_earthquake_psd(
        s, fs, tukey_alpha=0.1, duration_percent=(5, 75),
        nfft_method=nt, detrend_method='linear', smoothing_method='none', downsample_freqs=None 
    )
    
    if len(psd_raw) != len(m):
        raw_fas_freqs = m * fs / nt
        psd_raw = log_interp(raw_fas_freqs, freqs, psd_raw)

    return psd_raw, sd

def _rotdnn_psa_worker(nt, dt, envelope1, envelope2, TargetFAS0, TargetFAS90, T, zi, nn):
    """Worker to compute RotDnn PSA for a pair of synthetic motions."""
    so1 = np.random.randn(nt) * envelope1
    so2 = np.random.randn(nt) * envelope2
    
    FSo1 = np.fft.rfft(so1)
    FSo2 = np.fft.rfft(so2)
    FASo1 = dt * np.abs(FSo1)
    FASo2 = dt * np.abs(FSo2) 
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ff1 = TargetFAS0 / FASo1
        ff2 = TargetFAS90 / FASo2
    ff1[np.isnan(ff1) | np.isinf(ff1)] = 0.0
    ff2[np.isnan(ff2) | np.isinf(ff2)] = 0.0
    
    FS1 = ff1 * FSo1
    FS2 = ff2 * FSo2
    s1 = np.fft.irfft(FS1, n=nt)
    s2 = np.fft.irfft(FS2, n=nt)
    
    PSArotnn, _ = rotdnn(s1, s2, dt, zi, T, nn)
    return PSArotnn

def _rotdnn_psd_fas_worker(nt, dt, envelope1, envelope2, TargetFAS0, TargetFAS90, nn):
    """Worker to compute RotDnn PSD/FAS and Duration stats for a pair of motions."""
    fs = 1.0 / dt
    so1 = np.random.randn(nt) * envelope1
    so2 = np.random.randn(nt) * envelope2
    
    FSo1 = np.fft.rfft(so1)
    FSo2 = np.fft.rfft(so2)
    FASo1 = dt * np.abs(FSo1)
    FASo2 = dt * np.abs(FSo2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ff1 = TargetFAS0 / FASo1
        ff2 = TargetFAS90 / FASo2
    ff1[np.isnan(ff1) | np.isinf(ff1)] = 0.0
    ff2[np.isnan(ff2) | np.isinf(ff2)] = 0.0
    
    FS1 = ff1 * FSo1
    FS2 = ff2 * FSo2
    s1 = np.fft.irfft(FS1, n=nt)
    s2 = np.fft.irfft(FS2, n=nt)
    
    theta = np.arange(0, 180)
    thetarad = np.deg2rad(theta)
    t = np.linspace(0, (nt - 1) * dt, nt)
    
    SD_rot = np.zeros(180)
    for k in range(180):
        sr = s1 * np.cos(thetarad[k]) + s2 * np.sin(thetarad[k])
        try:
            sd_val, _, _, _, _ = SignificantDuration(sr, t, ival=5, fval=75)
            SD_rot[k] = sd_val
        except ValueError:
            SD_rot[k] = 0.0

    (_, _, psd_rotd_raw_interp) = calculate_psd_rotDnn(
        s1, s2, fs, percentiles=[nn], nfft_method=nt, smoothing_method='none', downsample_freqs=None)
    psd_rotdnn_raw = psd_rotd_raw_interp[nn]
    
    (freqs_raw_fas, _, fas_rotd_raw_interp) = calculate_fas_rotDnn(
        s1, s2, fs, percentiles=[nn], nfft_method=nt, smoothing_method='none', downsample_freqs=None)
    fas_rotdnn_raw = fas_rotd_raw_interp[nn]

    return SD_rot, psd_rotdnn_raw, fas_rotdnn_raw