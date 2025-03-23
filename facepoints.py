import cv2
import mediapipe as mp
import numpy as np


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
BREATHING_LANDMARKS = [
    1,   # Кончик носа
    175, # Подбородок
    8,   # Середина между глазами
    48, 278,
    115, 220, 45,
    344, 440, 275,
    236, 456, # Переносица
    209, 429,
]
CHEEK_RIGHT = [34, 58, 64, 47]
CHEEK_LEFT = [264, 288, 294, 277]
BETWEEN_EYEBROWS = [107, 55, 285, 336]
NOSE_BRIDGE = [114, 115, 278, 277]

def extract_landmarks(frame):
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                h, w, _ = frame.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                landmarks.append((x, y))
    return landmarks

def filter_landmarks(landmarks, *selected_indices_arrays):
    showing_landmarks = []
    for idx, landmark in enumerate(landmarks):
        for selected_indices in selected_indices_arrays:
            if (idx in selected_indices) and (landmark not in showing_landmarks):
                showing_landmarks.append(landmark)
    return showing_landmarks

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка при открытии видео.")
        return
    ret, prev_frame = cap.read()
    if not ret:
        print("Ошибка при чтении видео.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    all_landmarks = extract_landmarks(prev_frame)
    breathing_landmarks = filter_landmarks(all_landmarks, CHEEK_RIGHT, CHEEK_LEFT, BETWEEN_EYEBROWS, NOSE_BRIDGE)
    prev_points = np.array(breathing_landmarks, dtype=np.float32)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None)
        for (x, y) in next_points:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #Научился нормально проводить контуры по точкам, но по-хорошему вынести бы это в отдельную функцию
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                right_cheek_points = np.array([[int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)] for idx in CHEEK_RIGHT], np.int32)
                cv2.polylines(frame, [right_cheek_points], isClosed=True, color=(0, 255, 0), thickness=1)
                left_cheek_points = np.array([[int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)] for idx in CHEEK_LEFT], np.int32)
                cv2.polylines(frame, [left_cheek_points], isClosed=True, color=(0, 255, 0), thickness=1)
                between_eyebrows_points = np.array([[int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)] for idx in BETWEEN_EYEBROWS], np.int32)
                cv2.polylines(frame, [between_eyebrows_points], isClosed=True, color=(255, 0, 0), thickness=1)
                nose_bridge_points = np.array([[int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)] for idx in NOSE_BRIDGE], np.int32)
                cv2.polylines(frame, [nose_bridge_points], isClosed=True, color=(0, 0, 255), thickness=1)
        cv2.imshow('Breathing Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        prev_gray = gray.copy()
        prev_points = next_points
    cap.release()
    cv2.destroyAllWindows()
video_path = 'vidMe.mp4'
process_video(video_path)