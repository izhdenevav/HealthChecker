import cv2
# import mediapipe as mp
import numpy as np

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
FACE_TRACKING = [1, 199, 33, 263, 61, 291]

def extract_all_landmarks(frame):
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        landmarks = []
        h, w, _ = frame.shape
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x, y = landmark.x * w, landmark.y * h
                    landmarks.append((x, y))
        return landmarks

def get_breathing_landmarks(all_landmarks):
    showing_landmarks = []
    for idx, landmark in enumerate(all_landmarks):
        if (idx in BREATHING_LANDMARKS) and (landmark not in showing_landmarks):
            showing_landmarks.append(landmark)
    return showing_landmarks

def get_nose_landmarks(all_landmarks):
    showing_landmarks = []
    for idx, landmark in enumerate(all_landmarks):
        if (idx in NOSE_BRIDGE) and (landmark not in showing_landmarks):
            showing_landmarks.append(landmark)
    return showing_landmarks

def get_eyebrows_landmarks(all_landmarks):
    showing_landmarks = []
    for idx, landmark in enumerate(all_landmarks):
        if (idx in BETWEEN_EYEBROWS) and (landmark not in showing_landmarks):
            showing_landmarks.append(landmark)
    return showing_landmarks

def get_cheeckL_landmarks(all_landmarks):
    showing_landmarks = []
    for idx, landmark in enumerate(all_landmarks):
        if (idx in CHEEK_LEFT) and (landmark not in showing_landmarks):
            showing_landmarks.append(landmark)
    return showing_landmarks

def get_cheeckR_landmarks(all_landmarks):
    showing_landmarks = []
    for idx, landmark in enumerate(all_landmarks):
        if (idx in CHEEK_RIGHT) and (landmark not in showing_landmarks):
            showing_landmarks.append(landmark)
    return showing_landmarks

def get_face_tracking_landmarks(all_landmarks):
    showing_landmarks = []
    for idx in FACE_TRACKING:
        showing_landmarks.append(all_landmarks[idx])
    return showing_landmarks

#Если надо достать точки на той части лица, которых нет в списке поддерживаемых
def get_custom_landmarks(landmarks, *selected_indices_arrays):
    showing_landmarks = []
    for idx, landmark in enumerate(landmarks):
        for selected_indices in selected_indices_arrays:
            if (idx in selected_indices) and (landmark not in showing_landmarks):
                showing_landmarks.append(landmark)
    return showing_landmarks
