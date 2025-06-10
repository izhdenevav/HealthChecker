import numpy as np

from . import facepoints

# выделяем ключевые точки 4 лицевых зон на карте (щеки, межбровка, переносица), возвращаем словарем
def extract_face_regions(frame):
    regions = {}

    landmarks = facepoints.extract_all_landmarks(frame)

    regions['between_eyebrows_landmarks'] = np.array(facepoints.get_eyebrows_landmarks(landmarks), dtype=np.int32)
    regions['nose_landmarks'] = np.array(facepoints.get_nose_landmarks(landmarks), dtype=np.int32)
    regions['left_cheek_landmarks'] = np.array(facepoints.get_cheeckL_landmarks(landmarks), dtype=np.int32)
    regions['right_cheek_landmarks'] = np.array(facepoints.get_cheeckR_landmarks(landmarks), dtype=np.int32)

    return regions

# считаем средние значения пикселей каждой зоны лица
def get_facial_regions_means(frame, regions):
    if not regions:
        return None

    means = {}

    for region, coords in regions.items():
        min_x = min(coords, key=lambda p: p[0])[0]
        max_x = max(coords, key=lambda p: p[0])[0]
        min_y = min(coords, key=lambda p: p[1])[1]
        max_y = max(coords, key=lambda p: p[1])[1]

        cropped = frame[min_y:max_y, min_x:max_x]

        mean = np.mean(cropped, axis=(0, 1))

        means[region] = mean

    return means

# переводим средние значения пикселей из RGB в YUV по схеме, предложенной в статье
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
    # получаем координаты лицевых зон
    regions = extract_face_regions(frame)
    # получаем средние для каждого региона
    facial_means_rgb = get_facial_regions_means(frame, regions)
    # переводим в yuv
    facial_means_yuv = rgb_to_yuv(facial_means_rgb)

    return facial_means_yuv

# нормализуем полученную пространственно-временную (зоны/кадры) карту по кадрам
def apply_time_domain_normalization(spatial_temporal_map):
    normalized_map = {}

    regions = spatial_temporal_map[0].keys()

    for region in regions:
        values = np.array([frame[region] for frame in spatial_temporal_map])
        mean = values.mean(axis=0)
        std = values.std(axis=0) + 1e-8
        normalized = (values - mean) / std
        normalized_map[region] = normalized

    return normalized_map

# добавляем белый шум к карте
def add_white_noise(normalized_spatial_temporal_map, noise_std=0.1):
    noisy_series = {}
    for key, data in normalized_spatial_temporal_map.items():
        noise = np.random.normal(0, noise_std, data.shape)
        noisy_series[key] = data + noise
    return noisy_series