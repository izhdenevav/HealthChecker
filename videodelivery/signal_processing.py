import numpy as np
import torch
import cv2
from . import breathing_rate_calculator

class SignalProcessor:
    """
    класс SignalProcessor отвечает за обработку видеосигналов и расчет частоты дыхания
    """
    def __init__(self, fps=30, resp_band=(0.13, 0.3), max_frames=30000000):
        self.fps = fps
        self.resp_min, self.resp_max = resp_band
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Инициализация массивов для хранения необработанных и отфильтрованных сигналов
        self.cg_raw = np.zeros(max_frames, dtype=np.float32)
        self.cg_filtered = np.zeros(max_frames, dtype=np.float32)
        self.data_index = 0
        self.br_calculator = breathing_rate_calculator.BreathingRateCalculator(fps=fps)
    
    # Преобразование RGB-изображения в YCgCo цветовое пространство и расчет среднего значения Cg-компоненты
    def _rgb_to_ycgco(self, frame, roi):
        try:
            # Извлечение координат области интереса (ROI): x, y, ширина, высота
            x, y, w, h = roi
            # Выделение области интереса из кадра
            roi_patch = frame[y:y+h, x:x+w]
            # Проверка, что область интереса не пуста
            if roi_patch.size == 0:
                return None
                
            if torch.cuda.is_available():
                roi_patch_gpu = torch.from_numpy(roi_patch).to(self.device).float() / 255.0
                b, g, r = roi_patch_gpu[:, :, 0], roi_patch_gpu[:, :, 1], roi_patch_gpu[:, :, 2]
                # Вычисление Cg-компоненты по формуле: 128 + (-0.25 * R + 0.5 * G - 0.25 * B) * 255
                cg = 128 + (-0.25 * r + 0.5 * g - 0.25 * b) * 255
                return cg.mean().item()
            else:
                b, g, r = cv2.split(roi_patch.astype(np.float32) / 255.0)
                # Вычисление Cg-компоненты по формуле: 128 + (-0.25 * R + 0.5 * G - 0.25 * B) * 255
                cg = 128 + (-0.25 * r + 0.5 * g - 0.25 * b) * 255
                return np.mean(cg)
        except:
            return None

    # Применение фильтра на основе БПФ (быстрого преобразования Фурье) для выделения дыхательного сигнала
    def _apply_fft_filter(self, signal):
        n = len(signal)
        # Если сигнал слишком короткий, возвращаем нулевой массив
        if n < 10:
            return np.zeros(n, dtype=np.float32)
        signal_tensor = torch.tensor(signal, dtype=torch.float32).to(self.device)
        # Вычисление БПФ для сигнала
        fft_signal = torch.fft.fft(signal_tensor)
        # Вычисление частот, соответствующих БПФ
        freqs = torch.fft.fftfreq(n, 1 / self.fps).to(self.device)
        # Создание маски для фильтрации частот в заданном диапазоне (resp_min, resp_max)
        mask = (torch.abs(freqs) >= self.resp_min) & (torch.abs(freqs) <= self.resp_max)
        # Применение маски к спектру
        fft_signal_filtered = fft_signal * mask
        # Обратное БПФ для получения отфильтрованного сигнала
        filtered_signal = torch.fft.ifft(fft_signal_filtered).real
        return filtered_signal.cpu().numpy()

    # Обработка кадра и расчет частоты дыхания
    def process(self, frame, rois):     
        # Вычисление Cg-компоненты для всех областей интереса (ROIs)
        valid_values = np.array([self._rgb_to_ycgco(frame, roi) for roi in rois if self._rgb_to_ycgco(frame, roi) is not None], dtype=np.float32)
        if valid_values.size == 0:
            return None, None
        
        # Вычисление среднего значения Cg для всех областей интереса
        mean_cg = np.mean(valid_values)
        if self.data_index < len(self.cg_raw):
            self.cg_raw[self.data_index] = mean_cg
            self.data_index += 1
        
        # Определение окна данных
        start_idx = max(0, self.data_index - self.fps * 15)
        signal_to_filter = self.cg_raw[start_idx:self.data_index]
        # Применение фильтра БПФ к сигналу
        filtered_signal = self._apply_fft_filter(signal_to_filter)
        self.cg_filtered[start_idx:self.data_index] = filtered_signal
        
        if self.data_index > 0:
            filtered_value = self.cg_filtered[self.data_index - 1]
            self.br_calculator.add_signal(filtered_value)
            # Расчет частоты дыхания
            br = self.br_calculator.calculate_br()
            return filtered_value, br
        return None, None
    
    # Сброс состояния процессора сигналов
    def reset(self):
        self.cg_raw = np.zeros_like(self.cg_raw)
        self.cg_filtered = np.zeros_like(self.cg_filtered)
        self.data_index = 0
        self.br_calculator.reset()
    
    def get_all_data(self):
        return self.cg_raw[:self.data_index], self.cg_filtered[:self.data_index]