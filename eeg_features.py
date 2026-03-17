
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import emd

def butter_bandpass_filter(data, lowcut=0.5, highcut=30.0, fs=200, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def get_emd_cycle_features(imfs, sample_rate=200, target_imfs=3):
    """
    Extracts Instantaneous Frequency (IF) and Amplitude (IA) stats 
    from the EMD cycles.
    """
    # Initialize empty features [Mean IF, Std IF, Mean IA, Std IA] per target IMF
    # Shape: (target_imfs * 4,)
    cycle_feats = np.zeros(target_imfs * 4)
    
    # If EMD failed and returned a 1D flatline instead of 2D IMFs
    if imfs.ndim == 1 or imfs.shape[1] == 0:
        return cycle_feats
        
    try:
        # Calculate Instantaneous Phase, Frequency, and Amplitude using Hilbert Transform
        IP, IF, IA = emd.spectra.frequency_transform(imfs, sample_rate, 'hilbert')
        
        # We only look at the first `target_imfs` (usually the most active high/mid frequencies)
        num_imfs_to_process = min(target_imfs, imfs.shape[1])
        
        for j in range(num_imfs_to_process):
            base_idx = j * 4
            # Mean and Std of Instantaneous Frequency for this IMF cycle
            cycle_feats[base_idx] = np.nanmean(IF[:, j])
            cycle_feats[base_idx + 1] = np.nanstd(IF[:, j])
            
            # Mean and Std of Instantaneous Amplitude for this IMF cycle
            cycle_feats[base_idx + 2] = np.nanmean(IA[:, j])
            cycle_feats[base_idx + 3] = np.nanstd(IA[:, j])
            
    except Exception as e:
        # Failsafe for mathematical errors in Hilbert transform on bad signals
        pass
        
    # Replace any potential NaNs generated during transforms with 0
    return np.nan_to_num(cycle_feats)

def process_emd_cycles(filtered_data):
    num_samples, num_channels = filtered_data.shape
    
    # target_imfs=3 * 4 features per IMF = 12 features per channel
    feats_per_channel = 12 
    all_channel_feats = np.zeros((num_channels, feats_per_channel))
    
    for i in range(num_channels):
        channel_signal = filtered_data[:, i]
        
        # SAFETY CHECK: Bypass EMD if signal is completely flat
        if np.var(channel_signal) < 1e-8:
            pass # Leaves zeros for this channel
        else:
            try:
                # Limit IMFs to 4 to drastically speed up processing
                imfs = emd.sift.sift(channel_signal, max_imfs=4)
                
                # Extract the cycle properties directly from the IMFs
                channel_features = get_emd_cycle_features(imfs, sample_rate=200, target_imfs=3)
                all_channel_feats[i, :] = channel_features
                
            except UnboundLocalError:
                pass
            except Exception:
                pass
                
    return all_channel_feats.flatten()
import numpy as np

def auto_select_active_segment(data, fs=200, window_sec=50):
    """
    Scans the EEG data and extracts the most 'active' window 
    based on signal energy (sum of squared amplitudes).
    
    data: shape (samples, channels)
    fs: sampling frequency (200Hz)
    window_sec: How many seconds to extract (50 seconds is standard for capturing an event)
    """
    window_samples = int(fs * window_sec)
    
    # If the file is already shorter than our target window, just return the whole thing
    if len(data) <= window_samples:
        return data

    # 1. Square the data to make all values positive and emphasize large spikes (Energy)
    squared_data = data ** 2
    
    # 2. Sum the energy across all channels so we get one energy value per time step
    total_energy_per_sample = np.sum(squared_data, axis=1)
    
    # 3. Use convolution to efficiently calculate the rolling sum of energy for our window size
    window = np.ones(window_samples)
    rolling_energy = np.convolve(total_energy_per_sample, window, mode='valid')
    
    # 4. Find the start index of the window with the absolute highest energy
    best_start_idx = np.argmax(rolling_energy)
    best_end_idx = best_start_idx + window_samples
    
    # 5. Crop and return just that highly active segment
    return data[best_start_idx:best_end_idx, :]