import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt


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

def extract_landmarks(frame):
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                if idx in BREATHING_LANDMARKS:
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append((x, y))
    return landmarks

def bandpass_filter(signal, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    landmarks_history = []
    timestamps = []
    ret, prev_frame = cap.read() 
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_points = np.array(extract_landmarks(prev_frame), dtype=np.float32)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, None) # Вычисление оптического потока
        if len(next_points) > 0:
            y_displacement = np.mean(next_points[:, 1] - prev_points[:, 1])
            landmarks_history.append(y_displacement)
            timestamps.append(frame_count / fps) 
        for (x, y) in next_points:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        cv2.imshow('Breathing Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        prev_gray = gray.copy()
        prev_points = next_points
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()
    filtered_signal = bandpass_filter(landmarks_history, 0.1, 0.5, fps) # Фильтрация сигнала
    peaks, _ = find_peaks(filtered_signal, height=0) 
    breathing_rate = len(peaks) / (len(filtered_signal) / fps) * 60
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, filtered_signal, label='Filtered Signal')
    plt.plot(np.array(timestamps)[peaks], np.array(filtered_signal)[peaks], "x", label='Peaks')
    plt.title(f'Breathing Rate: {breathing_rate:.2f} bpm')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Displacement')
    plt.legend()
    plt.show()

video_path = 'any_vid.mp4'
process_video(video_path)