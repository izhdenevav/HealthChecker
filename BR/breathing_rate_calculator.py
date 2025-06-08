import numpy as np
from scipy.signal import find_peaks
import time

class BreathingRateCalculator:
    def __init__(self, fps=30, analysis_interval=5):
        self.fps = fps
        self.analysis_interval = analysis_interval  # в секундах
        self.buffer_size = int(fps * 15)  # буфер на 60 секунд
        self.signal_buffer = []
        self.last_analysis_time = time.time()
        self.start_time = time.time()
        self.peaks = []
        self.num_peaks = 0

    def add_signal(self, signal_value):
        self.signal_buffer.append(signal_value)
        if len(self.signal_buffer) > self.buffer_size:
            self.signal_buffer.pop(0)

    def calculate_br(self):
        current_time = time.time()
        if (current_time - self.last_analysis_time >= self.analysis_interval) or ((current_time - self.last_analysis_time >= 5) and (current_time - self.start_time >= self.analysis_interval)):
            if len(self.signal_buffer) < (self.fps * self.analysis_interval):
                return None  

            signal = np.array(self.signal_buffer)
            duration_seconds = len(signal) / self.fps
            duration_minutes = duration_seconds / 60.0
            #print(signal)
            if np.max(signal) > np.min(signal):
                signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
            else:
                signal = np.zeros_like(signal) 
            #print(signal)
            self.peaks, _ = find_peaks(signal, distance=int(self.fps), prominence=0.1)
            self.num_peaks = len(self.peaks)

            br = (self.num_peaks / duration_minutes) if duration_minutes > 0 else 0
            self.last_analysis_time = current_time
            #print(current_time - self.start_time)
            return br
        return None

    def reset(self):
        self.signal_buffer = []
        self.last_analysis_time = time.time()