"""
Example: Constructing a Duration-Dependent RotDnn-Compatible Target PSD
 (Two-Component)

This script demonstrates the refactored workflow for generating a RotDnn target PSD
using the unified 'TargetPSDGenerator' module.

Based on:
Montejo, L. A. (2026). "Generation of Fourier Amplitude Spectra and Power Spectral Density 
    Functions Compatible with Orientation-Independent Design Spectra for Bidirectional Seismic Analyses 
    of Nuclear Facilities." *Nuclear Engineering and Technology*, 58(5).
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import TargetFASandPSDGenerator as tpsd 
import warnings

plt.close('all')
# --- 0. Configuration ---
# Setup basic logging to see output from the module
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# Ensure user-facing warnings are always displayed by default
warnings.simplefilter("default")
# Get the logger object
log = logging.getLogger()

# --- Input Parameters ---
target_spectrum_name = 'BSSA14_M7_VS400_RJB50_RotD100'
filename = 'BSSA14_M7_VS400_RJB50_RotD100.txt'  # (frequency [Hz], PSA [g])
sd575_gm = 10.4 # Target Geometric Mean SD5-75 [s]
units_label = 'g' # Define the units for plotting

# Main function check is required for multiprocessing
if __name__ == '__main__':
    
    # --- 1. Load Data ---
    log.info(f"Loading target spectrum from {filename}...")
    dspec = np.loadtxt(filename)
        
    f_or = dspec[:, 0]  # frequencies [Hz]
    ds_or = dspec[:, 1] # amplitudes [g]

    # Frequencies for final smoothed output
    freqs_des = np.geomspace(0.1, 98, 50) 
    
    # --- 2. Perform Computation ---
    log.info(f"Generating RotDnn-Target-PSD for sd575_gm={sd575_gm}s...")
    
    results = tpsd.generate_rotdnn_target_psd(
        target_freqs=f_or,
        target_psa=ds_or, # Pass PSA in [g]
        sd575_gm=sd575_gm,
        freqs_des=freqs_des,
        workname=target_spectrum_name,
        nn_psa=100,
        nn_analysis=100,
        sd_ratio=1.3,
        smoothing_method='konno_ohmachi', # Use 'konno_ohmachi' or 'variable_window'
        smoothing_coeff=20.0,
        zi=0.05,
        F1=0.1, 
        F2=50.0, 
        allow_err=2.5,
        neqsPSD=1000,
        use_multiprocessing=True,
        prefer_pykooh=True
    )

    log.info("Computation complete.")
    log.info(f"Final PSA error: {results['iteration_errors'][-1]:.2f}%")
    log.info(f"Actual mean GM SD5-75: {results['sd575_gm_mean_actual']:.2f}s")
    
    # --- 3. Plot Results ---
    log.info("Generating plots...")
    
    fig = tpsd.plot_target_psd_results(
        results, 
        target_spectrum_name=target_spectrum_name,
        units=units_label
    )
    
    summary_plot_fn = f"{target_spectrum_name}_SD{sd575_gm:.1f}s_Summary.png"
        
    fig.savefig(summary_plot_fn, dpi=300)
   
        
    log.info(f"Plot saved to {summary_plot_fn}")
    
    plt.show()

    # --- 4. Save Results ---
    log.info("Saving target PSD file...")
    
    save_path = f'{target_spectrum_name}_SD{sd575_gm:.1f}s_TargetFAS_PSD.txt'
    
    tpsd.save_target_psd_file(results, save_path)
    
    log.info(f"Script finished. Outputs saved to {save_path} and  {summary_plot_fn}")