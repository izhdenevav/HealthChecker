import cv2
import numpy as np
import mediapipe as mp

class FaceProcessor:
    """
    Класс отвечает за процесс обработки лица и нахождения областей интереса накаждом кадре видеофрагмента
    """
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.landmark_indices = {
            'left_cheek': 101,  # Левая щека
            'right_cheek': 330  # Правая щека
        }
        
    def _get_face_size(self, landmarks, frame_width):
        # Получение размера лица на основе расстояния между глазами
        left_eye = landmarks[33] 
        right_eye = landmarks[263]
        # Вычисление евклидова расстояния между глазами
        eye_distance = np.sqrt(
            (left_eye.x - right_eye.x)**2 + 
            (left_eye.y - right_eye.y)**2
        ) * frame_width # Масштабирование расстояния до ширины кадра
        # Возвращаем удвоенное расстояние как оценку ширины лица
        return eye_distance * 2

    def _calculate_roi(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        face_width = self._get_face_size(landmarks, w)
        roi_size = int(face_width * 0.1) #размер областей интереса составляет 0.1 от площади лица
        left_cheek = landmarks[self.landmark_indices['left_cheek']] # Получение координат ключевой точки левой щеки
        lx, ly = left_cheek.x * w, left_cheek.y * h # Перевод в пиксели
        
        right_cheek = landmarks[self.landmark_indices['right_cheek']] # Так же для правой щеки
        rx, ry = right_cheek.x * w, right_cheek.y * h
        lx += int(roi_size * 0.2)
        rx -= int(roi_size * 0.2)
        # Формирование областей интереса
        roi_left = (int(lx - roi_size//2), int(ly - roi_size//2), roi_size, roi_size)
        roi_right = (int(rx - roi_size//2), int(ry - roi_size//2), roi_size, roi_size)
        
        return roi_left, roi_right

    def process_frame(self, frame):
        # Получение ключевых точек лица
        landmarks = self._get_landmarks(frame)
        rois = [] # Список для хранения областей интереса
        
        if landmarks:
            # Вычисление областей интереса для левой и правой щеки
            roi_left, roi_right = self._calculate_roi(landmarks, frame.shape)
            rois = [roi_left, roi_right] # Добавление областей в список
            
            # Отрисовка прямоугольников и текста для каждой области интереса
            for (x, y, w, h) in rois:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.putText(frame, f"{w}x{h}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return frame, rois

    def _get_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        return results.multi_face_landmarks[0].landmark if results.multi_face_landmarks else None