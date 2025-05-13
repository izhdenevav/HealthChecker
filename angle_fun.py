import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

import facepoints
import os


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class FrameHeadProcessor:
    def __init__(self):
        self.N = 5
        self.landmark_buffers = [deque(maxlen=self.N) for _ in range(6)]
        self.yaw_buffer = deque(maxlen=self.N)
        self.pitch_buffer = deque(maxlen=self.N)
        self.roll_buffer = deque(maxlen=self.N)
        self.flag = False

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
            return frame

        for j, point in enumerate(my_points):
            self.landmark_buffers[j].append(point)

        if not all(len(buf) > 0 for buf in self.landmark_buffers):
            return frame

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
        # cv2.putText(frame, f"Yaw: {smooth_yaw:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # cv2.putText(frame, f"Pitch: {smooth_pitch:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # cv2.putText(frame, f"Roll: {smooth_roll:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if (abs(smooth_yaw) <= 10 and smooth_yaw <= 3 and abs(smooth_pitch) <= 7 and abs(smooth_roll) <= 10):
            cv2.putText(frame, "Correct!", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            self.flag = True
        elif(self.flag==False or abs(smooth_yaw) >= 14 or smooth_yaw >= 5 or abs(smooth_pitch) >= 10 or abs(smooth_roll) >= 14):
            cv2.putText(frame, "Change position!", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            self.flag = False

        return frame

# Это можно удалить!!!
if __name__ == "__main__":
    # x_size, y_size = 200, 250
    cap = cv2.VideoCapture(0)
    # overlay = cv2.imread("D:\\Downloads\\foni-papik-pro-gljk-p-kartinki-idealnoe-litso-na-prozrachnom-fon-9.png", cv2.IMREAD_UNCHANGED)
    # overlay = cv2.resize(overlay, (x_size, y_size))
    head_pose = FrameHeadProcessor()

    # === Цикл обработки видео ===
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            continue

        frame = cv2.flip(frame, 1)
        # height, width, _ = frame.shape

        frame = head_pose.process(frame)

        cv2.imshow("Head Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()