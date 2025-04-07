import cv2
import threading
import matplotlib as plt
from face_processor import FaceProcessor
from signal_processing import SignalProcessor

def main():
    face_processor = FaceProcessor()
    signal_processor = SignalProcessor(warmup_frames=45)
    cap = cv2.VideoCapture('./dataset/vidMe1.mp4')

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            processed_frame, rois = face_processor.process_frame(frame)
            if rois:
                filtered_signal = signal_processor.process(frame, rois)
            status_color = (0, 255, 0) if signal_processor.is_ready else (0, 0, 255)
            cv2.imshow("Face ROI (Final)", processed_frame)
            cv2.putText(processed_frame, 
                       f"Status: {'ANALYZING' if signal_processor.is_ready else 'WARMUP'}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            key = cv2.waitKey(1)   
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()