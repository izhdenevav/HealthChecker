
# Breathing Rate Calculator

> "A living person has a very interesting characteristic...
> He breathes"
**Semenov Arseniy ©, 2025 B.C.**

The principle of the program is based on changing the shades of certain parts of the face. For proper operation of the program it is necessary to:

- Make sure that no cosmetics are applied to the area of your face under the eyes, as they may interfere with the correct operation of the program and distort the result
- Throughout the program, look clearly into the camera, do not make sudden movements of the head    
- Make sure that the camera is connected correctly and shoots at least 30 frames per second (yes, in 99% of cases the second condition is met for modern cameras, but if you doubt - it is better to check)

### Launch of the program
To run the program, run the main.py file:
```bash
python main.py
```
Profit!


## How it works

### General Overview

The program operates through the following steps:

1. Captures video frames and checks head position.
2. Identifies regions of interest (ROIs) on the face using facial landmark processing.
3. Extracts a breathing-related signal from the ROIs and processes it.
4. Calculates the breathing rate based on the processed signal.
5. Displays results and plots graphs in real-time.

### Detailed Description

#### 1. Video Capture and Head Position Check

- In `main.py`, video frames are captured using OpenCV (`cv2.VideoCapture`).
- The `check_position` function (presumably from `angle_fun.py`) verifies if the head is correctly positioned (based on yaw, pitch, and roll angles).
- If the head position is incorrect, the frame is skipped to avoid inaccurate measurements.

#### 2. Face Processing

- The `FaceProcessor` class in `face_processor.py` uses MediaPipe Face Mesh to detect facial landmarks.
- Regions of interest (ROIs) are defined on the left and right cheeks using predefined landmark indices (101 for the left cheek, 330 for the right cheek).
- The size of ROIs is calculated proportionally to the inter-eye distance to adapt to face size.

#### 3. Signal Processing

- The `SignalProcessor` class in `signal_processing.py` handles the extraction and processing of the breathing signal:
  - **Signal Extraction**: ROIs are converted from RGB to the YCgCo color space, and the Cg (Chrominance Green) component is extracted, which is sensitive to skin color changes due to blood flow.
  - **Cg Formula**: `Cg = 128 + (-0.25 * R + 0.5 * G - 0.25 * B) * 255`.
  - **Raw Signal**: The mean Cg value for all ROIs is inverted and stored in the `cg_raw` array.
  - **Filtering**: A Fast Fourier Transform (FFT) is applied to filter the signal within the breathing frequency range (0.05–0.43 Hz, corresponding to 3–26 breaths per minute). The filtered signal is stored in `cg_filtered`.

**Convertation from RGB to YCgCo:**
```python
def _rgb_to_ycgco(self, frame, roi):
    x, y, w, h = roi
    roi_patch = frame[y:y+h, x:x+w]
    b, g, r = cv2.split(roi_patch.astype(np.float32) / 255.0)
    cg = 128 + (-0.25 * r + 0.5 * g - 0.25 * b) * 255
    return np.mean(cg)
```

**Signal Filtering:**
```python
def _apply_fft_filter(self, signal):
    signal_tensor = torch.tensor(signal, dtype=torch.float32).to(self.device)
    fft_signal = torch.fft.fft(signal_tensor)
    freqs = torch.fft.fftfreq(n, 1 / self.fps).to(self.device)
    mask = (torch.abs(freqs) >= self.resp_min) & (torch.abs(freqs) <= self.resp_max)
    fft_signal_filtered = fft_signal * mask
    filtered_signal = torch.fft.ifft(fft_signal_filtered).real
    return filtered_signal.cpu().numpy()
```

#### 4. Breathing Rate Calculation

- The `BreathingRateCalculator` class in `breathing_rate_calculator.py` accumulates filtered signal values.
- Every 10 seconds, the signal is analyzed to detect peaks (using `scipy.signal.find_peaks`), which correspond to breaths.
- The breathing rate is calculated as the number of peaks per minute.

**Calculating of br:**
```python
def calculate_br(self):
    signal = np.array(self.signal_buffer)
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
    peaks, properties = find_peaks(signal, distance=int(self.fps * 0.5), prominence=0.1)
    num_peaks = len(peaks)
    duration_minutes = self.analysis_interval / 60.0
    br = num_peaks / duration_minutes
    return br
```

#### 5. Display and Graph Plotting

- Frames display ROIs, analysis status, and, if calculated, the breathing rate.
- Graphs of the raw (`cg_raw`) and filtered (`cg_filtered`) signals are plotted in real-time using Matplotlib and displayed via OpenCV.

**Graph Plotting Mechanism in `main.py`:**
- After the first 300 frames and every 30 frames, data from `cg_raw` and `cg_filtered` is retrieved using the `get_all_data` method.
- These arrays update the graph lines (`line_raw` and `line_filtered`), which are then rendered and displayed.

### Signal Processing and Graph Plotting Details

- **Data Source for Graphs**:
  - **`cg_raw`**: An array containing inverted mean Cg values for each frame, calculated from ROIs.
  - **`cg_filtered`**: An array with the filtered signal, obtained after applying FFT to `cg_raw`.

- **Data Calculation**:
  - The raw signal (`cg_raw`) is formed by extracting Cg from ROIs and averaging the values for each frame.
  - The filtered signal (`cg_filtered`) is obtained by applying an FFT filter to the last segment of `cg_raw` (over the past 60 seconds), retaining only breathing frequencies.

- **Graph Plotting**:
  - Graphs are updated every 30 frames after the initial 300 frames.
  - Matplotlib renders the data, and the image is converted to an OpenCV format for display.


