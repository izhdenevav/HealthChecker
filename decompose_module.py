import numpy as np
from scipy.fft import dct, idct

def DCTiDCT(spatial_temporal_map, number_of_stripes=4, fps=30, freq_bands=None):
    regions = ['between_eyebrows_landmarks', 'nose_landmarks', 'left_cheek_landmarks', 'right_cheek_landmarks']
    channels = ['y', 'u', 'v']
    
    if freq_bands is None:
        freq_bands = np.linspace(0.7, 2.5, number_of_stripes + 1)
        freq_bands = [(freq_bands[i], freq_bands[i+1]) for i in range(len(freq_bands)-1)]

    N = spatial_temporal_map[regions[0]].shape[0]
    freq_resolution = fps / N 
    freq_indices = np.arange(N) * freq_resolution

    multi_band_signals = {channel: [] for channel in channels}

    for channel_idx, channel in enumerate(channels):
        # [R, T]
        signals_per_region = np.array([spatial_temporal_map[region][:, channel_idx] for region in regions])
        # [R, T]
        dct_signals = dct(signals_per_region, axis=1, norm='ortho')

        for f_min, f_max in freq_bands:
            adjusted_spectrum = np.zeros_like(dct_signals)
            mask = (freq_indices >= f_min) & (freq_indices < f_max)
            adjusted_spectrum[:, mask] = dct_signals[:, mask]
            # [R, T]
            filtered_signals = idct(adjusted_spectrum, axis=1, norm='ortho')
            
            # [T]
            combined_signal = np.mean(filtered_signals, axis=0)
            multi_band_signals[channel].append(combined_signal)
    
    # [C, K, T]
    C = len(channels)
    K = number_of_stripes
    T = N
    result = np.zeros((C, K, T))
    for c_idx, channel in enumerate(channels):
        result[c_idx, :, :] = np.array(multi_band_signals[channel])

    return result