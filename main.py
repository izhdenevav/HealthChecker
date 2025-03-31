import cv2
import dlib
import numpy as np
import tensorflow as tf
from scipy.fftpack import dct, idct
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ================================
# 1. Предобработка видео: загрузка и построение пространственно-временной карты
# ================================
# Инициализация детектора лиц и предсказателя landmarks
detector = dlib.get_frontal_face_detector()
# Укажите путь к файлу модели предсказания лицевых landmarks
landmark_model_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(landmark_model_path)

def extract_face_regions(frame, face_rect):
    """
    Разбивает область лица на 4 региона на основе landmarks.
    Если landmarks недоступны, можно разбить прямоугольник на 4 равные части.
    """
    x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
    
    # Попытка получить landmarks
    shape = predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), face_rect)
    landmarks = np.array([(pt.x, pt.y) for pt in shape.parts()])
    
    # Для упрощения, определим регионы по среднему положению landmarks:
    # Например: лоб, левый щек, правый щек, подбородок
    # Можно взять средние координаты для групп точек:
    forehead = landmarks[17:27]      # точки лба (ориентировочно)
    left_cheek = landmarks[0:9]        # левая часть лица
    right_cheek = landmarks[9:17]      # правая часть лица
    chin = landmarks[6:11]           # центральная нижняя часть лица (подбородок)
    
    # Найдем ограничивающие прямоугольники для каждого региона:
    def region_from_landmarks(pts):
        x1, y1 = np.min(pts, axis=0)
        x2, y2 = np.max(pts, axis=0)
        return int(x1), int(y1), int(x2 - x1), int(y2 - y1)
    
    regions = []
    for pts in [forehead, left_cheek, right_cheek, chin]:
        rx, ry, rw, rh = region_from_landmarks(pts)
        # Извлечем ROI и усредним по пространству (возьмем зеленый канал)
        roi = frame[ry:ry+rh, rx:rx+rw]
        if roi.size == 0:
            regions.append(0)
        else:
            avg_val = np.mean(roi[:, :, 1])  # зеленый канал
            regions.append(avg_val)
    return regions

def build_spatiotemporal_map(video_path):
    """
    Загружает видео, детектирует лицо в каждом кадре, делит его на 4 региона
    и формирует пространственно-временную карту сигнала (T, 4)
    """
    print("Start upload the video")
    cap = cv2.VideoCapture(video_path)
    spatiotemporal_map = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Детектируем лицо
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) == 0:
            continue  # если лицо не найдено, пропускаем кадр
        
        # Берем первое обнаруженное лицо
        face_rect = faces[0]
        # Извлекаем 4 региона по landmarks
        regions = extract_face_regions(frame, face_rect)
        spatiotemporal_map.append(regions)
    
    cap.release()
    print("Finish upload the video")
    # Преобразуем список в массив numpy
    spatiotemporal_map = np.array(spatiotemporal_map, dtype=np.float32)
    return spatiotemporal_map

# ================================
# 2. Модуль разложения сигнала (Multi-frequency decompose) на основе DCT
# ================================
class MultiFrequencyDecompose(tf.keras.layers.Layer):
    def __init__(self, K=3, **kwargs):
        """
        K: число частотных диапазонов для разделения.
        """
        super(MultiFrequencyDecompose, self).__init__(**kwargs)
        self.K = K

    def call(self, x):
        # x имеет форму (batch, T, R)
        T = tf.shape(x)[1]
        # Применяем DCT по оси времени
        x_dct = tf.signal.dct(x, type=2, norm='ortho', axis=-1)  # (batch, T, R)
        
        # Разделяем коэффициенты на K диапазонов по оси времени
        band_size = T // self.K
        bands = []
        for k in range(self.K):
            start = k * band_size
            # Для последнего диапазона берем остаток
            end = tf.cond(tf.equal(k, self.K - 1),
                          lambda: T,
                          lambda: (k + 1) * band_size)
            # Создаем маску для коэффициентов
            mask = tf.concat([
                tf.zeros((start,)),
                tf.ones((end - start,)),
                tf.zeros((T - end,))
            ], axis=0)
            mask = tf.reshape(mask, (1, T, 1))
            masked = x_dct * mask
            # Обратное DCT
            filtered = tf.signal.idct(masked, type=2, norm='ortho', axis=-1)
            bands.append(filtered)
        # Конкатенация по последней оси: (batch, T, R*K)
        return tf.concat(bands, axis=-1)
    
    def get_config(self):
        config = super(MultiFrequencyDecompose, self).get_config()
        config.update({"K": self.K})
        return config

# ================================
# 3. Модули для модели FastBVP-Net
# ================================
class TemporalMultiScaleConv(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(TemporalMultiScaleConv, self).__init__(**kwargs)
        self.conv3 = tf.keras.layers.Conv1D(filters, kernel_size=3, padding='same', activation='relu')
        self.conv5 = tf.keras.layers.Conv1D(filters, kernel_size=5, padding='same', activation='relu')
        self.conv7 = tf.keras.layers.Conv1D(filters, kernel_size=7, padding='same', activation='relu')
    
    def call(self, x):
        out3 = self.conv3(x)
        out5 = self.conv5(x)
        out7 = self.conv7(x)
        return tf.concat([out3, out5, out7], axis=-1)
    
    def get_config(self):
        config = super(TemporalMultiScaleConv, self).get_config()
        config.update({"filters": self.conv3.filters})
        return config

class SpectrumSelfAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads=4, key_dim=16, **kwargs):
        super(SpectrumSelfAttention, self).__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
    
    def call(self, x):
        attn_output = self.mha(x, x)
        return attn_output

    def get_config(self):
        config = super(SpectrumSelfAttention, self).get_config()
        config.update({"num_heads": self.mha.num_heads, "key_dim": self.mha.key_dim})
        return config

def build_fastbvp_net(input_shape, K=3):
    """
    input_shape: (T, R) – T: число временных отсчетов, R: число регионов (например, 4).
    K: число частотных диапазонов.
    """
    inputs = tf.keras.Input(shape=input_shape)  # например, (T, 4)
    
    # Модуль разложения сигнала с использованием DCT
    x = MultiFrequencyDecompose(K=K)(inputs)  # (T, 4*K)
    
    # Signal Refinement Sub-network
    x = TemporalMultiScaleConv(filters=32)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    # Используем padding='same', чтобы сохранить размерность при пулинге
    x = tf.keras.layers.MaxPooling1D(pool_size=2, padding='same')(x)  # Если T=1999, то output T=ceil(1999/2)=1000
    
    x = TemporalMultiScaleConv(filters=64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Spectrum Self-Attention модуль
    attn = SpectrumSelfAttention(num_heads=4, key_dim=16)(x)
    x = tf.keras.layers.Add()([x, attn])
    
    # Signal Reconstruction Sub-network
    x = tf.keras.layers.UpSampling1D(size=2)(x)  # увеличиваем T: 1000*2=2000
    # Так как целевая размерность T=1999, добавим слой обрезки
    x = tf.keras.layers.Cropping1D(cropping=(0, 1))(x)  # теперь T=2000-1=1999
    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(x)
    outputs = tf.keras.layers.Conv1D(1, kernel_size=1, padding='same', activation='linear')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ================================
# 4. Функции обучения и инференса
# ================================
def train_fastbvp_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    print(f"Размерности данных перед обучением: y_train={y_train.shape}, y_pred={model.output_shape}")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop]
    )
    return history

def predict_bvp(model, X_test):
    return model.predict(X_test)

# ================================
# 5. Пример использования: загрузка видео, предобработка и обучение модели
# ================================
if __name__ == '__main__':
    # Путь к видео с лицом
    video_path = 'before copy.mp4'
    
    # Извлекаем пространственно-временную карту (форма: (T, 4))
    spatiotemporal_map = build_spatiotemporal_map(video_path)
    print("Извлеченная пространственно-временная карта:", spatiotemporal_map.shape)

    # Если обнаружен только один кадр, преобразуем форму из (4,) в (1, 4)
    if spatiotemporal_map.ndim == 1:
        spatiotemporal_map = spatiotemporal_map[np.newaxis, :]

    # Для обучения модели требуется пара (X, y)
    # Здесь y – истинный сигнал BVP (например, измеренный пульс)
    # Создаем синтетический y, равный среднему по регионам с добавлением шума
    y_signal = spatiotemporal_map.mean(axis=1, keepdims=True)
    y_signal_noisy = y_signal + 0.05 * np.random.randn(*y_signal.shape)
    y_data = y_signal_noisy.astype(np.float32)

    # Добавляем размерность батча для X и y:
    X_data = spatiotemporal_map[np.newaxis, ...]  # форма (1, T, 4)
    y_data = y_data[np.newaxis, ...]              # форма (1, T, 1)
    
    # Разбиваем на обучающую и валидационную выборки (здесь демонстрация с одним примером)
    X_train, X_val = X_data, X_data
    y_train, y_val = y_data, y_data
    
    # Параметры входа: T и число регионов (4)
    T = spatiotemporal_map.shape[0]
    R = spatiotemporal_map.shape[1]
    
    model = build_fastbvp_net(input_shape=(T, R), K=3)
    model.summary()
    
    # Обучаем модель
    history = train_fastbvp_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=1)
    
    # Применяем модель для инференса
    reconstructed_bvp = predict_bvp(model, X_data)
    
    # Сохраняем модель
    model.save('fastbvp_net.h5')
    
    print("Обучение завершено. Пример восстановленного сигнала BVP:")
    print(reconstructed_bvp[0])
