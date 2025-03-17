import cv2
import numpy as np


# Загрузка предобученного классификатора для лица
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
facemark = cv2.face.createFacemarkLBF()
facemark.loadModel("lbfmodel.yaml")  # Нужно скачать модель LBF отдельно

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Загрузка изображения
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:  # Убедимся, что обнаружены лица
       faces_list = [np.array([x, y, w, h], dtype=np.int32) for (x, y, w, h) in faces]
       success, landmarks = facemark.fit(gray, faces)


    # Отображение лиц и ключевых точек
    if success:
        for landmark in landmarks:
            chin_x, chin_y = int(landmark[0][30][0]), int(landmark[0][30][1])
            cv2.circle(frame, (chin_x, chin_y), 4, (0, 0, 255), -1) 
            # Подбородок (центральная нижняя точка)
            chin_x, chin_y = int(landmark[0][8][0]), int(landmark[0][8][1])
            cv2.circle(frame, (chin_x, chin_y), 4, (0, 0, 255), -1)

            # Левый глаз (центр)
            left_eye_x, left_eye_y = int(landmark[0][42][0]), int(landmark[0][42][1])
            cv2.circle(frame, (left_eye_x, left_eye_y), 4, (0, 0, 255), -1)

            # Правый глаз (центр)
            right_eye_x, right_eye_y = int(landmark[0][39][0]), int(landmark[0][39][1])
            cv2.circle(frame, (right_eye_x, right_eye_y), 4, (0, 0, 255), -1)

            # Левый угол рта
            left_mouth_x, left_mouth_y = int(landmark[0][48][0]), int(landmark[0][48][1])
            cv2.circle(frame, (left_mouth_x, left_mouth_y), 4, (0, 0, 255), -1)

            # Правый угол рта
            right_mouth_x, right_mouth_y = int(landmark[0][54][0]), int(landmark[0][54][1])
            cv2.circle(frame, (right_mouth_x, right_mouth_y), 4, (0, 0, 255), -1)

            
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Faces", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
