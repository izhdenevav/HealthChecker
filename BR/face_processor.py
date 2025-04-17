import cv2
import numpy as np
import mediapipe as mp

class KalmanFilter:
    def __init__(self, process_noise=1e-6, measurement_noise=1e-3):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise

    def update(self, x, y):
        measurement = np.array([[x], [y]], np.float32)
        self.kf.correct(measurement)
        predicted = self.kf.predict()
        return predicted[0, 0], predicted[1, 0]

class FaceProcessor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.landmark_indices = {
            'left_cheek': 101,  # Левая щека
            'right_cheek': 330  # Правая щека
        }
        
        self.kf_left = KalmanFilter()
        self.kf_right = KalmanFilter()

    def _get_face_size(self, landmarks, frame_width):
        # Тут считаю ширину лица
        left_eye = landmarks[33] 
        right_eye = landmarks[263]
        eye_distance = np.sqrt(
            (left_eye.x - right_eye.x)**2 + 
            (left_eye.y - right_eye.y)**2
        ) * frame_width
        return eye_distance * 2

    def _calculate_roi(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        face_width = self._get_face_size(landmarks, w)
        roi_size = int(face_width * 0.1)
        left_cheek = landmarks[self.landmark_indices['left_cheek']]
        lx, ly = self.kf_left.update(left_cheek.x * w, left_cheek.y * h)
        
        right_cheek = landmarks[self.landmark_indices['right_cheek']]
        rx, ry = self.kf_right.update(right_cheek.x * w, right_cheek.y * h)
        lx += int(roi_size * 0.2)
        rx -= int(roi_size * 0.2)
        roi_left = (int(lx - roi_size//2), int(ly - roi_size//2), roi_size, roi_size)
        roi_right = (int(rx - roi_size//2), int(ry - roi_size//2), roi_size, roi_size)
        
        return roi_left, roi_right

    def process_frame(self, frame):
        landmarks = self._get_landmarks(frame)
        rois = []
        
        if landmarks:
            roi_left, roi_right = self._calculate_roi(landmarks, frame.shape)
            rois = [roi_left, roi_right]
            
            # Визуализация
            for (x, y, w, h) in rois:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.putText(frame, f"{w}x{h}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return frame, rois

    def _get_landmarks(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        return results.multi_face_landmarks[0].landmark if results.multi_face_landmarks else None

"""if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    processor = FaceProcessor()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame, rois = processor.process_frame(frame)
        cv2.imshow("Face ROI (Final)", processed_frame)
        key = cv2.waitKey(1)   
        if key == ord('q') or key == 27:  # 27 — код ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()"""