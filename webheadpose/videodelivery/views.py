from django.http import StreamingHttpResponse
from django.shortcuts import render
from collections import deque

import cv2
import numpy as np
import torch
from scipy.signal import find_peaks

from . import facepoints
from . import data_preprocessing
from . import decompose_module
from . import srrn
from . import face_processor
from . import signal_processing

# Класс обработки кадров(код арины)
class FrameProcessor:
    def __init__(self):
        self.N = 10
        self.br_value = 0
        self.landmark_buffers = [deque(maxlen=self.N) for _ in range(6)]
        self.yaw_buffer = deque(maxlen=self.N)
        self.pitch_buffer = deque(maxlen=self.N)
        self.roll_buffer = deque(maxlen=self.N)
        self.frames_per_calculation = 300
        self.frames_cnt = 0
        self.spatial_temporal_map = []
        self.model = srrn.SRRN(in_channels=3, R=4, T=300)
        self.bpm = 0.0
        self.face_processor = face_processor.FaceProcessor()
        self.signal_processor = signal_processing.SignalProcessor(max_frames=10000)

        # === Параметры камеры ===
        self.model_points = np.array([
            (0.0, 0.0, 0.0),  # 1. Кончик носа
            (0.0, -330.0, -65.0),  # 2. Подбородок
            (-225.0, 170.0, -135.0),  # 3. Левый глаз (внешний угол)
            (225.0, 170.0, -135.0),  # 4. Правый глаз (внешний угол)
            (-150.0, -150.0, -125.0),  # 5. Левый угол рта
            (150.0, -150.0, -125.0)  # 6. Правый угол рта
        ], dtype=np.float64)

        focal_length = 1.10851252e+03
        center = (640, 360)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

    def process(self, frame):
        raw_points = facepoints.extract_all_landmarks(frame)
        my_points = []
        if raw_points:
            my_points = facepoints.get_face_tracking_landmarks(raw_points)

        if len(my_points) != 6:
            return frame  # Нет всех точек — просто вернём кадр как есть

        for j, point in enumerate(my_points):
            self.landmark_buffers[j].append(point)

        if not all(len(buf) > 0 for buf in self.landmark_buffers):
            return frame  # Буферы ещё не накопились

        image_points = np.array([
            np.mean(self.landmark_buffers[j], axis=0) for j in range(6)
        ], dtype=np.float64)

        # Решаем PnP
        _, rotation_vector, _ = cv2.solvePnP(
            self.model_points, image_points, self.camera_matrix, None, flags=cv2.SOLVEPNP_SQPNP)

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
        yaw, pitch, roll = angles[1], angles[0], angles[2]

        if pitch > 90:
            pitch = 180 - pitch
        elif pitch < -90:
            pitch = -(180 + pitch)

        self.yaw_buffer.append(yaw)
        self.pitch_buffer.append(pitch)
        self.roll_buffer.append(roll)

        smooth_yaw = np.median(self.yaw_buffer)
        smooth_pitch = np.median(self.pitch_buffer)
        smooth_roll = np.median(self.roll_buffer)

        # Вывод текста на кадр
        cv2.putText(frame, f"Yaw: {smooth_yaw:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Pitch: {smooth_pitch:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Roll: {smooth_roll:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"HR: {self.bpm:.1f}", (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame,
                            f"BR: {self.br_value:.2f} breaths/min",
                            (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if (abs(smooth_yaw) <= 10 and smooth_yaw <= 3 and abs(smooth_pitch) <= 7 and abs(smooth_roll) <= 10):
            cv2.putText(frame, "Correct!", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            processed_frame, rois = self.face_processor.process_frame(frame)
            if rois:
                filtered_value, br = self.signal_processor.process(frame, rois)
                if br is not None:
                    self.br_value = br
            
            if self.frames_cnt <= self.frames_per_calculation:
                print(self.frames_cnt)
                csc = data_preprocessing.apply_color_space_conversion(frame)
                self.spatial_temporal_map.append(csc)

                self.frames_cnt += 1
            else:
                tdn = data_preprocessing.apply_time_domain_normalization(self.spatial_temporal_map)
                noised = data_preprocessing.add_white_noise(tdn)

                # [С, R, T]
                processed_data = decompose_module.DCTiDCT(noised, number_of_stripes=4, fps=30)
                # [1, C, R, T]
                data = np.expand_dims(processed_data, axis=0)

                data_tensor = torch.tensor(data, dtype=torch.float32)

                with torch.no_grad():
                    bvp_signal = self.model(data_tensor)

                sig = bvp_signal[0]

                peaks, _ = find_peaks(sig, distance=0.5*30, height=np.percentile(sig, 75))

                num_beats = len(peaks) - 1
                duration_s = len(sig) / 30
                bpm = num_beats / duration_s * 60
                self.bpm = bpm

                self.frames_cnt = 0
                self.spatial_temporal_map = []
        else:
            cv2.putText(frame, "Change position!", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame


# Генератор кадров
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Не удалось открыть видеопоток')

    processor = FrameProcessor()

    while cap.isOpened():
        success, frame = cap.read()

        if not success or frame is None:
            continue

        frame = processor.process(frame)

        if frame is None:
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


# Страница index.html
def index(request):
    return render(request, 'index.html')

# Поток видео
def video_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
