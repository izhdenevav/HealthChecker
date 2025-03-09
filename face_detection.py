import cv2
import numpy as np

# === 1. Загрузка YOLO модели ===
yolo_net = cv2.dnn.readNet("yolov4-tiny-3l_best.weights", "yolov4-tiny-3l.cfg")  # Файлы модели
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]  # Выходные слои

# === 2. Захват видео с веб-камеры ===
cap = cv2.VideoCapture(0)

# === 3. Основной цикл обработки ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape  # Размер кадра

    # === 4. Подготовка изображения для YOLO ===
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    yolo_net.setInput(blob)
    yolo_outs = yolo_net.forward(output_layers)  # Прогон через сеть

    faces = []  # Список обнаруженных лиц
    conf_threshold = 0.5  # Минимальный порог уверенности
    nms_threshold = 0.4  # Порог подавления слабых боксов

    # === 5. Обработка результатов YOLO ===
    for out in yolo_outs:
        for detection in out:
            scores = detection[5:]  # Вероятности
            confidence = max(scores)  # Уверенность
            if confidence > conf_threshold:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                faces.append((x, y, w, h, confidence))

    # === 6. Фильтрация (Non-Maximum Suppression) ===
    indices = cv2.dnn.NMSBoxes([f[:4] for f in faces], [f[4] for f in faces], conf_threshold, nms_threshold)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h, conf = faces[i]

            # === 7. Отрисовка рамки вокруг лица ===
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # === 8. Отображение видео ===
    cv2.imshow("YOLO Face Detection", frame)

    # Выход по ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# === 9. Освобождение ресурсов ===
cap.release()
cv2.destroyAllWindows()
