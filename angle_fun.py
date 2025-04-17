import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp

import facepoints
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

N = 5
# Буферы для сглаживания ключевых точек
landmark_buffers = [deque(maxlen=N) for _ in range(6)]
yaw_buffer = deque(maxlen=N)
pitch_buffer = deque(maxlen=N)
roll_buffer = deque(maxlen=N)
FLAG = False

def check_position(frame, landmark_buffers, yaw_buffer, pitch_buffer, roll_buffer):
    # === 3D модель ключевых точек головы ===
    model_points = np.array([
        (0.0, 0.0, 0.0),         # 1. Кончик носа
        (0.0, -330.0, -65.0),    # 2. Подбородок
        (-225.0, 170.0, -135.0), # 3. Левый глаз (внешний угол)
        (225.0, 170.0, -135.0),  # 4. Правый глаз (внешний угол)
        (-150.0, -150.0, -125.0),# 5. Левый угол рта
        (150.0, -150.0, -125.0)  # 6. Правый угол рта
    ], dtype=np.float64)

    # === Параметры камеры ===
    focal_length = 640
    center = (640, 360)  # Предполагаем разрешение 640x480
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    raw_points = facepoints.extract_all_landmarks(frame)
    my_points = []
    if raw_points:
        my_points = facepoints.get_face_tracking_landmarks(raw_points)
    
    if len(my_points) != 6:
        # print("Not all points")
        return False

    for j, point in enumerate(my_points):
        landmark_buffers[j].append(point)

    if not all(len(buf) > 0 for buf in landmark_buffers):
        return False
    
    image_points = np.array([
        np.mean(landmark_buffers[j], axis=0) for j in range(6)
    ], dtype=np.float64)
    # print(image_points)
    # for point in image_points:
    #     cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

    # === Решаем PnP ===
    _, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_SQPNP)
    
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    yaw, pitch, roll = angles[1], angles[0], angles[2]

    if(pitch > 90):
        pitch = 180 - pitch
    elif(pitch < -90):
        pitch = -(180 + pitch)

    yaw_buffer.append(yaw)
    pitch_buffer.append(pitch)
    roll_buffer.append(roll)

    smooth_yaw = np.median(yaw_buffer)
    smooth_pitch = np.median(pitch_buffer)
    smooth_roll = np.median(roll_buffer)

    if (abs(yaw-smooth_yaw) >= 10) or (abs(roll-smooth_roll) >= 10) or (abs(pitch-smooth_pitch) >= 10):
        FLAG = False
    # cv2.putText(frame, f"Yaw: {smooth_yaw:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # cv2.putText(frame, f"Pitch: {smooth_pitch:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # cv2.putText(frame, f"Roll: {smooth_roll:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if(abs(smooth_yaw) <= 10 and smooth_yaw <= 3 and abs(smooth_pitch) <= 7 and abs(smooth_roll) <= 10):
        FLAG = True
        return True
    FLAG = False
    return False

# Это можно удалить!!!
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    # === Цикл обработки видео ===
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            continue

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        # Пример запуска функции
        if check_position(frame, landmark_buffers, yaw_buffer, pitch_buffer, roll_buffer):
            cv2.putText(frame, "Correct!", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Change possition!", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Head Pose Estimation", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()