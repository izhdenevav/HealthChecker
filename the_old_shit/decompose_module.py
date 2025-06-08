import numpy as np
from scipy.fft import dct, idct

def DCTiDCT(spatial_temporal_map, number_of_stripes=4, fps=30, freq_bands=None):
    # C=3 (Y, U, V), R=4 (регионы), T=кол-во кадров
    C, R, T = spatial_temporal_map.shape
    K = number_of_stripes

    if freq_bands is None:
        freq_bands = np.linspace(0.7, 4.0, number_of_stripes + 1)
        freq_bands = [(freq_bands[i], freq_bands[i + 1]) for i in range(len(freq_bands) - 1)]

    freq_resolution = fps / T
    freq_indices = np.arange(T) * freq_resolution

    multi_band_signals = np.zeros((C, K, R, T))

    for c in range(C):
        signals_per_region = spatial_temporal_map[c]

        dct_signals = dct(signals_per_region, axis=1, norm='ortho')

        for k, (f_min, f_max) in enumerate(freq_bands):
            mask = (freq_indices >= f_min) & (freq_indices < f_max)
            adjusted_spectrum = np.zeros_like(dct_signals)
            adjusted_spectrum[:, mask] = dct_signals[:, mask]

            filtered_signals = idct(adjusted_spectrum, axis=1, norm='ortho')

            multi_band_signals[c, k, :, :] = filtered_signals

    return multi_band_signals