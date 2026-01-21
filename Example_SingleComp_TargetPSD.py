"""
Example: Constructing a Duration-Dependent Spectrum-Compatible Target PSD
 (Single-Component)

This script demonstrates the refactored workflow for generating a target PSD
using the unified 'TargetPSDGenerator' module.

Based on:
Montejo, L.A. 2024. "Strong-Motion-Duration-Dependent Power Spectral Density 
Functions Compatible with Design Response Spectra" Geotechnics 4, no. 4: 1048-1064. 
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
target_spectrum_name = 'WUS_M7.5_R75'
filename = 'WUS_M7.5_R75_Frequencies.txt' # e.g: 'CEUS_M7.5_R150_Frequencies.txt', 'WUS_M7.5_R75_Frequencies.txt'  # (frequency [Hz], PSA [g])
sd575 = 12.50  # Target SD5-75 [s]
units_label = 'g' # Define the units for plotting

# Main function check is required for multiprocessing
if __name__ == '__main__':
    
    # --- 1. Load Data ---
    log.info(f"Loading target spectrum from {filename}...")
    target_spec_data = np.loadtxt(filename)
    
    f_or = target_spec_data[:, 0]  # frequencies [Hz]
    ds_or = target_spec_data[:, 1] # amplitudes [g]
    
    # Define the desired output frequencies
    freqs_des = np.geomspace(0.1, 100, 100)
        
    # --- 2. Perform Computation ---
    log.info(f"Generating DD-Target-PSD for sd575={sd575}s...")
    
    results = tpsd.generate_single_comp_target_psd(
        target_freqs=f_or,
        target_psa=ds_or, # Pass PSA in [g]
        sd575=sd575,
        freqs_des=freqs_des,
        workname=target_spectrum_name,
        zi=0.05,
        F1=0.2,
        F2=50.0, 
        allow_err=2.5,
        neqsPSD=1000,
        use_multiprocessing=True,
        fas_smoothing_method='konno_ohmachi',
        fas_smoothing_coeff=20.0,
        psd_smoothing_method='konno_ohmachi',
        psd_smoothing_coeff=20.0,
        prefer_pykooh=True
    )
    
    log.info("Computation complete.")
    log.info(f"Final PSA error: {results['iteration_errors'][-1]:.2f}%")
    log.info(f"Actual mean SD5-75: {results['sd575_mean_actual']:.2f}s")

    # --- 3. Plot Results ---
    log.info("Generating plots...")
    
    fig = tpsd.plot_target_psd_results(
        results, 
        target_spectrum_name=target_spectrum_name,
        units=units_label
    )
    
    plot_filename = f"{target_spectrum_name}_SD575_{sd575:.2f}s.png"
    fig.savefig(plot_filename, dpi=300)
    log.info(f"Summary plot saved to {plot_filename}")
    
    plt.show()

    # --- 4. Save Results ---
    log.info("Saving target PSD file...")
    
    save_path = f'{target_spectrum_name}_SD575_{sd575:.2f}s.txt'
    
    tpsd.save_target_psd_file(results, save_path)
    
    log.info(f"Script finished. Outputs saved to {save_path} and {plot_filename}")