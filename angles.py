import time

import cv2
import numpy as np
import mediapipe as mp

# === Инициализация YOLO ===
yolo_net = cv2.dnn.readNet("yolov4-tiny-3l_best.weights", "yolov4-tiny-3l.cfg")  # Файлы YOLO
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]
conf_threshold = 0.5  # Минимальная уверенность
nms_threshold = 0.4   # Порог подавления слабых боксов

# === Инициализация MediaPipe Face Mesh ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# === Захват видео с веб-камеры ===
cap = cv2.VideoCapture(0)

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
focal_length = 800
center = (320, 240)  # Предполагаем разрешение 640x480
camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float64)

counter = 0

# === Цикл обработки видео ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Отзеркалим изображение для удобства
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # === Детекция лица с помощью YOLO ===
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    yolo_outs = yolo_net.forward(output_layers)

    faces = []
    for out in yolo_outs:
        for detection in out:
            scores = detection[5:]
            confidence = max(scores)
            if confidence > conf_threshold:
                # Координаты ограничивающей рамки
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                faces.append((x, y, w, h, confidence))

    # Подавление незначимых рамок (NMS)
    indices = cv2.dnn.NMSBoxes(
        [f[:4] for f in faces], [f[4] for f in faces], conf_threshold, nms_threshold
    )
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h, _ = faces[i]

            # Ограничим область лица
            face_roi = frame[y:y+h, x:x+w]

            # Преобразуем в RGB для MediaPipe
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)

            # === Обнаружение ключевых точек лица ===
            results = face_mesh.process(rgb_face)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Извлекаем координаты 6 ключевых точек (переводим в глобальные координаты кадра)
                    image_points = np.array([
                        [face_landmarks.landmark[1].x * w + x, face_landmarks.landmark[1].y * h + y],  # Кончик носа
                        [face_landmarks.landmark[199].x * w + x, face_landmarks.landmark[199].y * h + y],  # Подбородок
                        [face_landmarks.landmark[33].x * w + x, face_landmarks.landmark[33].y * h + y],  # Левый глаз
                        [face_landmarks.landmark[263].x * w + x, face_landmarks.landmark[263].y * h + y],  # Правый глаз
                        [face_landmarks.landmark[61].x * w + x, face_landmarks.landmark[61].y * h + y],  # Левый угол рта
                        [face_landmarks.landmark[291].x * w + x, face_landmarks.landmark[291].y * h + y]   # Правый угол рта
                    ], dtype=np.float64)

                    # === Решаем PnP для определения поворота головы ===
                    _, rotation_vector, translation_vector = cv2.solvePnP(
                        model_points, image_points, camera_matrix, None, flags=cv2.SOLVEPNP_ITERATIVE
                    )

                    # Конвертация в углы поворота
                    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
                    print(counter)
                    if(counter%40 == 0):
                        counter = 0
                        yaw, pitch, roll = angles[1], angles[0], angles[2]

                    if(pitch > 90):
                        pitch = 180 - pitch
                    elif(pitch < -90):
                        pitch = -(180 + pitch)
                    # === Отображение информации ===
                    cv2.putText(frame, f"Yaw: {yaw:.2f}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Pitch: {pitch:.2f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Roll: {roll:.2f}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Рисуем ключевые точки
                    for point in image_points:
                        cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

            # Отрисовка рамки лица
            if(abs(yaw) > 10 or abs(roll) > 10 or abs(pitch) > 10):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)   
            counter += 1
    # === Отображение результата ===
    cv2.imshow("Head Pose Estimation with YOLO", frame)

    # Выход по ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# === Освобождение ресурсов ===
cap.release()
cv2.destroyAllWindows()
