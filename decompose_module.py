# import numpy as np
# from scipy.fft import dct, idct

# def DCTiDCT(spatial_temporal_map, number_of_stripes=4, fps=30, freq_bands=None):
#     regions = ['between_eyebrows_landmarks', 'nose_landmarks', 'left_cheek_landmarks', 'right_cheek_landmarks']
#     channels = ['y', 'u', 'v']
    
#     if freq_bands is None:
#         freq_bands = np.linspace(0.7, 2.5, number_of_stripes + 1)
#         freq_bands = [(freq_bands[i], freq_bands[i+1]) for i in range(len(freq_bands)-1)]

#     N = spatial_temporal_map[regions[0]].shape[0]
#     freq_resolution = fps / N 
#     freq_indices = np.arange(N) * freq_resolution

#     multi_band_signals = {channel: [] for channel in channels}

#     for channel_idx, channel in enumerate(channels):
#         # [R, T]
#         signals_per_region = np.array([spatial_temporal_map[region][:, channel_idx] for region in regions])
#         # [R, T]
#         dct_signals = dct(signals_per_region, axis=1, norm='ortho')

#         for f_min, f_max in freq_bands:
#             adjusted_spectrum = np.zeros_like(dct_signals)
#             mask = (freq_indices >= f_min) & (freq_indices < f_max)
#             adjusted_spectrum[:, mask] = dct_signals[:, mask]
#             # [R, T]
#             filtered_signals = idct(adjusted_spectrum, axis=1, norm='ortho')
            
#             # [T]
#             combined_signal = np.mean(filtered_signals, axis=0)
#             multi_band_signals[channel].append(combined_signal)
    
#     # [C, K, T]
#     C = len(channels)
#     K = number_of_stripes
#     T = N
#     result = np.zeros((C, K, T))
#     for c_idx, channel in enumerate(channels):
#         result[c_idx, :, :] = np.array(multi_band_signals[channel])

#     return result

# import numpy as np
# from scipy.fft import dct, idct

# def DCTiDCT(spatial_temporal_map, number_of_stripes=4, fps=30, freq_bands=None):
#     """
#     Выполняет многочастотное разложение сигнала с помощью DCT и IDCT.
    
#     Args:
#         spatial_temporal_map (np.ndarray): Входной тензор размером [B, C=3, R=4, T] или [C=3, R=4, T].
#         number_of_stripes (int): Количество частотных полос (K).
#         fps (int): Частота кадров видео.
#         freq_bands (list of tuples, optional): Список кортежей (f_min, f_max) для частотных полос.
    
#     Returns:
#         np.ndarray: Многочастотные сигналы размером [B, C=3, K, T] или [C=3, K, T].
#     """
#     if spatial_temporal_map.ndim == 3:  # [C, R, T]
#         spatial_temporal_map = spatial_temporal_map[None, ...]  # Добавляем batch dimension [1, C, R, T]
    
#     B, C, R, T = spatial_temporal_map.shape
    
#     if freq_bands is None:
#         freq_bands = np.linspace(0.7, 2.5, number_of_stripes + 1)  # Частоты от 0.7 до 2.5 Гц (42–150 bpm)
#         freq_bands = [(freq_bands[i], freq_bands[i+1]) for i in range(len(freq_bands)-1)]
    
#     freq_resolution = fps / T
#     freq_indices = np.arange(T) * freq_resolution
    
#     multi_band_signals = np.zeros((B, C, number_of_stripes, T))
    
#     for b in range(B):
#         for c in range(C):
#             # [R, T]
#             signals_per_region = spatial_temporal_map[b, c, :, :]  # Сигналы для канала c
#             # [R, T]
#             dct_signals = dct(signals_per_region, axis=1, norm='ortho')
            
#             for k, (f_min, f_max) in enumerate(freq_bands):
#                 adjusted_spectrum = np.zeros_like(dct_signals)
#                 mask = (freq_indices >= f_min) & (freq_indices < f_max)
#                 adjusted_spectrum[:, mask] = dct_signals[:, mask]
#                 # [R, T]
#                 filtered_signals = idct(adjusted_spectrum, axis=1, norm='ortho')
#                 # [T]
#                 combined_signal = np.mean(filtered_signals, axis=0)
#                 multi_band_signals[b, c, k, :] = combined_signal
    
#     if B == 1:
#         return multi_band_signals[0]  # [C, K, T]
#     return multi_band_signals  # [B, C, K, T]

# decompose_module.py

import numpy as np
from scipy.fft import dct, idct

def DCTiDCT(spatial_temporal_map, number_of_stripes=4, fps=30, freq_bands=None):
    """
    
    Args:
        spatial_temporal_map (dict or list of dict): Словарь или список словарей с YUV значениями для регионов.
        number_of_stripes (int): Количество частотных полос (K).
        fps (int): Частота кадров видео.
        freq_bands (list of tuples, optional): Список кортежей (f_min, f_max) для частотных полос.
    
    Returns:
        np.ndarray: Многочастотные сигналы размером [B, C=3, K, T] или [C=3, K, T].
    """
    regions = ['between_eyebrows_landmarks', 'nose_landmarks', 'left_cheek_landmarks', 'right_cheek_landmarks']
    channels = ['y', 'u', 'v']
    
    if isinstance(spatial_temporal_map, dict):
        spatial_temporal_map = [spatial_temporal_map]
    
    B = len(spatial_temporal_map) 
    T = len(spatial_temporal_map[0][regions[0]])
    C = len(channels)
    K = number_of_stripes
    
    if freq_bands is None:
        freq_bands = np.linspace(0.7, 2.5, number_of_stripes + 1)  # Частоты от 0.7 до 2.5 Гц
        freq_bands = [(freq_bands[i], freq_bands[i+1]) for i in range(len(freq_bands)-1)]
    
    freq_resolution = fps / T
    freq_indices = np.arange(T) * freq_resolution
    
    multi_band_signals = np.zeros((B, C, K, T))
    
    for b in range(B):
        for region in regions:
            if region not in spatial_temporal_map[b] or len(spatial_temporal_map[b][region]) != T:
                print(f"Предупреждение: Некорректные данные для региона {region} в батче {b}")
                spatial_temporal_map[b][region] = [[0, 0, 0]] * T
        
        for c_idx, channel in enumerate(channels):
            # [R, T]
            signals_per_region = np.array([
                [frame[c_idx] for frame in spatial_temporal_map[b][region]]
                for region in regions
            ])
            
            # [R, T]
            dct_signals = dct(signals_per_region, axis=1, norm='ortho')
            
            for k, (f_min, f_max) in enumerate(freq_bands):
                adjusted_spectrum = np.zeros_like(dct_signals)
                mask = (freq_indices >= f_min) & (freq_indices < f_max)
                adjusted_spectrum[:, mask] = dct_signals[:, mask]
                # [R, T]
                filtered_signals = idct(adjusted_spectrum, axis=1, norm='ortho')
                # [T]
                combined_signal = np.mean(filtered_signals, axis=0)
                multi_band_signals[b, c_idx, k, :] = combined_signal
    
    if B == 1:
        return multi_band_signals[0]  # [C, K, T]
    return multi_band_signals  # [B, C, K, T]