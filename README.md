
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


## How it works (далее на русском)

### Общий обзор

Программа работает следующим образом:

1. Захватывает видеокадры и проверяет положение головы.
2. Определяет регионы интереса (ROIs) на лице с помощью обработки лицевых ориентиров.
3. Извлекает сигнал, связанный с дыханием, из ROIs и обрабатывает его.
4. Рассчитывает частоту дыхания на основе обработанного сигнала.
5. Отображает результаты и строит графики в реальном времени.

### Подробное описание

#### 1. Захват видео и проверка положения головы

- В файле `main.py` видео захватывается покадрово с помощью OpenCV (`cv2.VideoCapture`).
- Функция `check_position` (предположительно из `angle_fun.py`) проверяет, правильно ли расположена голова (по углам поворота: yaw, pitch, roll).
- Если положение головы неверное, кадр пропускается, чтобы избежать некорректных измерений.

#### 2. Обработка лица

- Класс `FaceProcessor` в файле `face_processor.py` использует MediaPipe Face Mesh для обнаружения лицевых ориентиров.
- Регионы интереса (ROIs) определяются на левой и правой щеках с использованием заранее заданных индексов ориентиров (101 для левой щеки, 330 для правой).
- Размер ROIs рассчитывается пропорционально расстоянию между глазами, чтобы адаптироваться к размеру лица.

#### 3. Обработка сигнала

- Класс `SignalProcessor` в файле `signal_processing.py` отвечает за извлечение и обработку сигнала дыхания:
  - **Извлечение сигнала**: ROIs преобразуются из RGB в цветовое пространство YCgCo, и извлекается компонент Cg (Chrominance Green), который чувствителен к изменениям цвета кожи из-за кровотока.
  - **Формула Cg**: `Cg = 128 + (-0.25 * R + 0.5 * G - 0.25 * B) * 255`.
  - **Сырой сигнал**: Среднее значение Cg для всех ROIs инвертируется и сохраняется в массиве `cg_raw`.
  - **Фильтрация**: Применяется быстрое преобразование Фурье (FFT) для фильтрации сигнала в диапазоне частот дыхания (0.05–0.43 Гц, что соответствует 3–26 вдохам в минуту). Отфильтрованный сигнал сохраняется в `cg_filtered`.

**Перевод в YCgCo:**
```python
def _rgb_to_ycgco(self, frame, roi):
    x, y, w, h = roi
    roi_patch = frame[y:y+h, x:x+w]
    b, g, r = cv2.split(roi_patch.astype(np.float32) / 255.0)
    cg = 128 + (-0.25 * r + 0.5 * g - 0.25 * b) * 255
    return np.mean(cg)
```

**Фильтрация сигнала:**
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

#### 4. Расчет частоты дыхания

- Класс `BreathingRateCalculator` в файле `breathing_rate_calculator.py` накапливает значения отфильтрованного сигнала.
- Каждые 10 секунд сигнал анализируется для поиска пиков (с помощью `scipy.signal.find_peaks`), которые соответствуют вдохам.
- Частота дыхания рассчитывается как количество пиков в минуту.

**Рассчет частоты дыхания:**
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

#### 5. Отображение и построение графиков

- На кадрах отображаются ROIs, статус анализа и, если рассчитана, частота дыхания.
- Графики сырого (`cg_raw`) и отфильтрованного (`cg_filtered`) сигналов строятся в реальном времени с помощью Matplotlib и отображаются через OpenCV.

**Механизм построения графиков в `main.py`:**
- После первых 300 кадров и каждые 30 кадров данные из `cg_raw` и `cg_filtered` извлекаются с помощью метода `get_all_data`.
- Эти массивы обновляют линии графиков (`line_raw` и `line_filtered`), которые затем отрисовываются и отображаются.

### Подробности обработки сигналов и построения графиков

- **Источник данных для графиков**:
  - **`cg_raw`**: Массив, содержащий средние значения Cg для каждого кадра, рассчитанные из ROIs.
  - **`cg_filtered`**: Массив с отфильтрованным сигналом, полученным после применения FFT к `cg_raw`.

- **Расчет данных**:
  - Сырой сигнал (`cg_raw`) формируется путем извлечения Cg из ROIs и усреднения значений для каждого кадра.
  - Отфильтрованный сигнал (`cg_filtered`) получается путем применения FFT-фильтра к последнему сегменту `cg_raw` (за последние 60 секунд), сохраняя только частоты дыхания.

- **Построение графиков**:
  - Графики обновляются каждые 30 кадров после начальных 300 кадров.
  - Matplotlib отрисовывает данные, а затем изображение преобразуется в формат OpenCV для отображения.



