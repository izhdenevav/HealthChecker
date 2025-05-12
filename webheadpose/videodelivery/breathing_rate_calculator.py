import numpy as np
from scipy.signal import find_peaks
import time

class BreathingRateCalculator:
    def __init__(self, fps=30, analysis_interval=10):
        self.fps = fps
        self.analysis_interval = analysis_interval 
        self.buffer_size = int(fps * analysis_interval)
        self.signal_buffer = []
        self.last_analysis_time = time.time()

    def add_signal(self, signal_value):
        self.signal_buffer.append(signal_value)
        if len(self.signal_buffer) > self.buffer_size:
            self.signal_buffer.pop(0)

    def calculate_br(self):
        current_time = time.time()
        if current_time - self.last_analysis_time >= self.analysis_interval:
            if len(self.signal_buffer) < self.buffer_size:
                return None  

            signal = np.array(self.signal_buffer)
            signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
            peaks, properties = find_peaks(signal, distance=int(self.fps * 0.5), prominence=0.1)
            num_peaks = len(peaks)
            #print(peaks, num_peaks)
            duration_minutes = self.analysis_interval / 60.0
            br = num_peaks / duration_minutes
            self.last_analysis_time = current_time
            return br
        return None

    def reset(self):
        self.signal_buffer = []
        self.last_analysis_time = time.time()