import cv2
import mediapipe as mp
import numpy as np

print(mp.__file__)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture('./subject33/vid.avi')  
# cap = cv2.VideoCapture(0)  

CHEEK_RIGHT = [34, 58, 64, 47]
CHEEK_LEFT = [264, 288, 294, 277]
BETWEEN_EYEBROWS = [107, 55, 285, 336]
NOSE_BRIDGE = [114, 115, 278, 277]

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks_pixel = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

                right_cheek_points = np.array([[int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)] for idx in CHEEK_RIGHT], np.int32)
                left_cheek_points = np.array([[int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)] for idx in CHEEK_LEFT], np.int32)
                between_eyebrows_points = np.array([[int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)] for idx in BETWEEN_EYEBROWS], np.int32)
                nose_bridge_points = np.array([[int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)] for idx in NOSE_BRIDGE], np.int32)

                cv2.polylines(frame, [right_cheek_points], isClosed=True, color=(0, 255, 0), thickness=1)
                cv2.polylines(frame, [left_cheek_points], isClosed=True, color=(0, 255, 0), thickness=1)
                cv2.polylines(frame, [between_eyebrows_points], isClosed=True, color=(255, 0, 0), thickness=1)
                cv2.polylines(frame, [nose_bridge_points], isClosed=True, color=(0, 0, 255), thickness=1)

                # min_x = min(nose_bridge_points, key=lambda p: p[0])[0]
                # max_x = max(nose_bridge_points, key=lambda p: p[0])[0]
                # min_y = min(nose_bridge_points, key=lambda p: p[1])[1]
                # max_y = max(nose_bridge_points, key=lambda p: p[1])[1]

                # Вырезаем фрагмент изображения
                # cropped_nose_bridge = frame[min_y:max_y, min_x:max_x]

        # frame_yuv = cv2.cvtColor(cropped_nose_bridge, cv2.COLOR_BGR2YUV)

        # yuv_to_gray = cv2.cvtColor(frame_yuv, cv2.COLOR_BGR2GRAY)
        # mean_value = np.mean(yuv_to_gray)
        # normalized_frame = yuv_to_gray - mean_value

        # mean = 0
        # stddev = 25
        # noise = np.random.normal(mean, stddev, cropped_nose_bridge.shape).astype(np.float32)
        # noisy_frame = np.clip(cropped_nose_bridge + noise, 0, 255).astype(np.float32)
        # noisy_frame_gray = cv2.cvtColor(noisy_frame.astype(np.uint8), cv2.COLOR_BGR2GRAY)

        # dct_frame = cv2.dct(np.float32(noisy_frame_gray))

        # rows, cols = cropped_nose_bridge.shape

        # dct_low = np.zeros_like(dct_frame)
        
        # dct_low[:10, :10] = dct_frame[:10, :10]

        # frame_filtered = cv2.idct(dct_low)

        cv2.imshow("Face Landmarks", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
