import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from collections import deque
import cv2

class SignalProcessor:
    def __init__(self, fps=30, resp_band=(0.13, 0.33), warmup_frames=15):
        """
        :param fps: Частота кадров
        :param resp_band: Диапазон частот дыхания (0.13-0.33 Гц)
        :param warmup_frames: Кадры для стабилизации перед анализом
        """
        self.fps = fps
        self.resp_min, self.resp_max = resp_band
        self.warmup_frames = warmup_frames
        self.frame_counter = 0
        self.is_ready = False
        self.cg_raw = deque(maxlen=fps*15)
        self.cg_filtered = []

        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.ax1.set_title("Raw Cg Signal")
        self.ax2.set_title("Filtered Respiratory Signal")

    def _rgb_to_ycgco(self, frame, roi):
        try:
            x, y, w, h = roi
            roi_patch = frame[y:y+h, x:x+w]
            if roi_patch.size == 0:
                return None
                
            b, g, r = cv2.split(roi_patch.astype(np.float32)/255.0)
            cg = 128 + (-0.25*r + 0.5*g - 0.25*b)*255
            return np.mean(cg)
        except:
            return None

    def _apply_fft_filter(self, signal):
        n = len(signal)
        if n < 10: return np.zeros_like(signal)
        fft_signal = fft(signal)
        freqs = np.fft.fftfreq(n, 1/self.fps)
        mask = (np.abs(freqs) >= self.resp_min) & (np.abs(freqs) <= self.resp_max)
        return np.real(ifft(fft_signal * mask))

    def process(self, frame, rois):
        self.frame_counter += 1
        # У меня mediapipe положение нужных точек не сразу находит, поэтому даём ему время
        if not self.is_ready:
            if self.frame_counter > self.warmup_frames:
                self.is_ready = True
                print("Система готова к анализу!")
            return None
        
        valid_values = []
        for roi in rois:
            cg = self._rgb_to_ycgco(frame, roi)
            if cg is not None:
                valid_values.append(cg)
        
        if not valid_values:
            return None
            
        self.cg_raw.append(np.mean(valid_values))
        # Фильтруем
        if len(self.cg_raw) > self.fps*5:
            self.cg_filtered = self._apply_fft_filter(list(self.cg_raw))
            self._update_plots()
            
        return self.cg_filtered

    def _update_plots(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.plot(self.cg_raw, color='blue', label='Raw')
        self.ax2.plot(self.cg_filtered, color='red', label='Filtered')
        for ax in [self.ax1, self.ax2]:
            ax.legend()
            ax.grid(True)
            
        plt.pause(0.01)

    def reset(self):
        # сброс состояния
        self.cg_raw.clear()
        self.cg_filtered = []
        self.frame_counter = 0
        self.is_ready = False