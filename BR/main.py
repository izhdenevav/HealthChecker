import cv2
from collections import deque
from angle_fun import check_position
from face_processor import FaceProcessor
from signal_processing import SignalProcessor
import numpy as np
import matplotlib.pyplot as plt

N = 10
landmark_buffers = [deque(maxlen=N) for _ in range(6)]
yaw_buffer = deque(maxlen=N)
pitch_buffer = deque(maxlen=N)
roll_buffer = deque(maxlen=N)

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
ax1.set_title("Raw Cg Signal")
ax2.set_title("Filtered Respiratory Signal")

line_raw, = ax1.plot([], [], color='blue', label='Raw')
ax1.legend()
ax1.grid(True)
line_filtered, = ax2.plot([], [], color='red', label='Filtered')
ax2.legend()
ax2.grid(True)

def main():
    face_processor = FaceProcessor()
    signal_processor = SignalProcessor(max_frames=10000)
    cap = cv2.VideoCapture('./dataset/nobreathe.mp4')
    cv2.namedWindow("Face ROI (Final)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face ROI (Final)", 1000, 1300)
    
    frame_counter = 0
    br_value = None
    
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
                filtered_value, br = signal_processor.process(frame, rois)
                if br is not None:
                    br_value = br
            
            status_color = (0, 255, 0) 
            cv2.putText(processed_frame, 
                        f"Status: {'ANALYZING'}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            if br_value is not None:
                cv2.putText(processed_frame,
                            f"BR: {br_value:.2f} breaths/min",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            cv2.imshow("Face ROI (Final)", processed_frame)
            
            frame_counter += 1
            print(frame_counter)

            if (frame_counter > 300) and (frame_counter % 30 == 0):
                cg_raw, cg_filtered = signal_processor.get_all_data()
                if cg_raw.size > 0:
                    line_raw.set_data(np.arange(len(cg_raw)), cg_raw)
                    ax1.relim()
                    ax1.autoscale_view()
                    if cg_filtered.size > 0:
                        line_filtered.set_data(np.arange(len(cg_filtered)), cg_filtered)
                        ax2.relim()
                        ax2.autoscale_view()

                    fig.canvas.draw()
                    width, height = fig.get_size_inches() * fig.dpi
                    width, height = int(width), int(height)
                    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)

                    cv2.imshow("Plots", image)

            key = cv2.waitKey(1)
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        

if __name__ == "__main__":
    main()