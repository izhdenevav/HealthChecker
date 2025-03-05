import cv2
import numpy as np
import mediapipe as mp

# Инициализация MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Захват видео с веб-камеры
cap = cv2.VideoCapture(0)

# 3D модель ключевых точек головы
model_points = np.array([
    (0.0, 0.0, 0.0),         # 1. Кончик носа
    (0.0, -330.0, -65.0),    # 2. Подбородок
    (-225.0, 170.0, -135.0), # 3. Левый глаз (внешний угол)
    (225.0, 170.0, -135.0),  # 4. Правый глаз (внешний угол)
    (-150.0, -150.0, -125.0),# 5. Левый угол рта
    (150.0, -150.0, -125.0)  # 6. Правый угол рта
], dtype=np.float64)

# Фокусное расстояние (приблизительное, можно подстроить)
focal_length = 800
center = (320, 240)  # Центр изображения (предполагается разрешение 640x480)
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float64)

# Цикл обработки видео
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Отзеркаливание и преобразование в RGB (MediaPipe требует RGB)
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Обнаружение лица
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Извлекаем координаты 6 ключевых точек (в пикселях)
            image_points = np.array([
                [face_landmarks.landmark[1].x * frame.shape[1], face_landmarks.landmark[1].y * frame.shape[0]],  # Кончик носа
                [face_landmarks.landmark[199].x * frame.shape[1], face_landmarks.landmark[199].y * frame.shape[0]],  # Подбородок
                [face_landmarks.landmark[33].x * frame.shape[1], face_landmarks.landmark[33].y * frame.shape[0]],  # Левый глаз
                [face_landmarks.landmark[263].x * frame.shape[1], face_landmarks.landmark[263].y * frame.shape[0]],  # Правый глаз
                [face_landmarks.landmark[61].x * frame.shape[1], face_landmarks.landmark[61].y * frame.shape[0]],  # Левый угол рта
                [face_landmarks.landmark[291].x * frame.shape[1], face_landmarks.landmark[291].y * frame.shape[0]]   # Правый угол рта
            ], dtype=np.float64)

            # Решаем PnP для определения поворота головы
            _, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE
            )

            # Конвертируем в углы поворота
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

            yaw, pitch, roll = angles[1], angles[0], angles[2]

            # Отображение углов на экране
            cv2.putText(frame, f"Yaw: {yaw:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Pitch: {pitch:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Roll: {roll:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Отрисовка ключевых точек на лице
            for point in image_points:
                cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

    # Отображение изображения
    cv2.imshow("Head Pose Estimation", frame)

    # Выход по клавише ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
