import numpy as np

from . import facepoints

def extract_face_regions(frame):
    regions = {}

    landmarks = facepoints.extract_all_landmarks(frame)

    regions['between_eyebrows_landmarks'] = np.array(facepoints.get_eyebrows_landmarks(landmarks), dtype=np.int32)
    regions['nose_landmarks'] = np.array(facepoints.get_nose_landmarks(landmarks), dtype=np.int32)
    regions['left_cheek_landmarks'] = np.array(facepoints.get_cheekL_landmarks(landmarks), dtype=np.int32)
    regions['right_cheek_landmarks'] = np.array(facepoints.get_cheekR_landmarks(landmarks), dtype=np.int32)

    return regions

def get_facial_regions_means(frame, regions):
    if not regions:
        return None
    
    means = {}

    for region, coords in regions.items():
        if coords is None or (isinstance(coords, (list, np.ndarray)) and len(coords) == 0):
            means[region] = [0, 0, 0]
            continue

        min_x = min(coords, key=lambda p: p[0])[0]
        max_x = max(coords, key=lambda p: p[0])[0]
        min_y = min(coords, key=lambda p: p[1])[1]
        max_y = max(coords, key=lambda p: p[1])[1]

        cropped = frame[min_y:max_y, min_x:max_x]

        mean = np.mean(cropped, axis=(0, 1))

        means[region] = mean

    return means

# перевод в другое цветовое пространство
def rgb_to_yuv(means_rgb):
    means_yuv = {}
    for region, rgb in means_rgb.items():
        R, G, B = rgb
        Y = 0.299 * R + 0.587 * G + 0.114 * B
        U = -0.169 * R - 0.331 * G + 0.5 * B
        V = 0.5 * R - 0.419 * G - 0.081 * B
        means_yuv[region] = [Y, U, V]

    return means_yuv

def apply_color_space_conversion(frame):
    if frame is None or frame.size == 0:
        print("Ошибка: Пустой или некорректный кадр")
        return None
    
    # получаем координаты лицевых зон
    regions = extract_face_regions(frame)
    if regions is None or not regions:
        print("Ошибка: Не удалось извлечь регионы лица")
        return None
    
    # получаем средние для каждого региона
    facial_means_rgb = get_facial_regions_means(frame, regions)
    if not facial_means_rgb:
        print("Ошибка: Не удалось вычислить средние значения RGB")
        return None
    
    # переводим в yuv
    facial_means_yuv = rgb_to_yuv(facial_means_rgb)
    
    return facial_means_yuv

def apply_time_domain_normalization(spatial_temporal_map):
    regions = spatial_temporal_map[0].keys()
    # у нас массив 300 кадров, каждый кадр массив 4 зоны, каждая зона массив из 3 чисел - значений y u v; c = 3 (y, u, v), r = 4 (зоны), t = 300 (кадров, потому что 10 секунд хотим)
    C, R, T = 3, len(regions), len(spatial_temporal_map) 
    normalized = np.zeros((C, R, T))
    
    for r, region in enumerate(regions):
        values = np.array([frame[region] for frame in spatial_temporal_map])  # [T, C]
        mean = values.mean(axis=0)
        std = values.std(axis=0) + 1e-8
        normalized[:, r, :] = ((values - mean) / std).T  # [C, T]
    
    return normalized

def add_white_noise(normalized_spatial_temporal_map, noise_std=0.05):
    noise = np.random.normal(0, noise_std, normalized_spatial_temporal_map.shape)
    noisy_map = normalized_spatial_temporal_map + noise
    return noisy_map
