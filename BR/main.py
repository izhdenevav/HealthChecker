import cv2
from collections import deque
from angle_fun import check_position
from face_processor import FaceProcessor
from signal_processing import SignalProcessor
import matplotlib.pyplot as plt
import numpy as np

N = 10
landmark_buffers = [deque(maxlen=N) for _ in range(6)]
yaw_buffer = deque(maxlen=N)
pitch_buffer = deque(maxlen=N)
roll_buffer = deque(maxlen=N)

def main():
    face_processor = FaceProcessor()
    signal_processor = SignalProcessor(max_frames=10000)
    cap = cv2.VideoCapture('./dataset/vidMe1.mp4')
    cv2.namedWindow("Face ROI (Final)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face ROI (Final)", 600, 800)
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if check_position(frame, landmark_buffers, yaw_buffer, pitch_buffer, roll_buffer):
                cv2.putText(frame, "Correct!", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Change position!", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue
            
            processed_frame, rois = face_processor.process_frame(frame)
            if rois:
                signal_processor.process(frame, rois)
            
            status_color = (0, 255, 0) 
            cv2.putText(processed_frame, 
                        f"Status: {'ANALYZING'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.imshow("Face ROI (Final)", processed_frame)
            
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        """cg_raw, cg_filtered = signal_processor.get_all_data()
        if cg_raw.size > 0:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 1, 1)
            plt.plot(cg_raw, color='blue', label='Raw')
            plt.title("Raw Cg Signal")
            plt.legend()
            plt.grid(True)
            
            if cg_filtered.size > 0:
                plt.subplot(2, 1, 2)
                plt.plot(cg_filtered, color='red', label='Filtered')
                plt.title("Filtered Respiratory Signal")
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            plt.show()"""

if __name__ == "__main__":
    main()