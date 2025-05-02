from collections import deque

import numpy as np
import cv2
import torch
from scipy.signal import find_peaks

from data_preprocessing import apply_color_space_conversion, apply_time_domain_normalization, add_white_noise
from decompose_module import DCTiDCT
from srrn import SRRN
# from angle_fun import check_position

# N = 5
# # Буферы для сглаживания ключевых точек
# landmark_buffers = [deque(maxlen=N) for _ in range(6)]
# yaw_buffer = deque(maxlen=N)
# pitch_buffer = deque(maxlen=N)
# roll_buffer = deque(maxlen=N)

def main():
    # face_processor = FaceProcessor()
    # signal_processor = SignalProcessor(warmup_frames=45)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera")
        exit()

    spatial_temporal_map = []
    frame_count = 0
    # 10 секунд видео
    number_of_frames_per_calculation = 300

    model = SRRN(in_channels=3, R=4, T=300)
    checkpoint = torch.load("srrn_best.pth", map_location="cpu")
    model.load_state_dict(checkpoint)

    model.eval()

    try:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret: 
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # if check_position(frame, landmark_buffers, yaw_buffer, pitch_buffer, roll_buffer):
            #     cv2.putText(frame, "Correct!", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            # else:
            #     cv2.putText(frame, "Change possition!", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            #     continue
            
            # processed_frame, rois = face_processor.process_frame(frame)
            # if rois:
                # filtered_signal = signal_processor.process(frame, rois)
            # status_color = (0, 255, 0) if signal_processor.is_ready else (0, 0, 255)
            # cv2.imshow("Face ROI (Final)", processed_frame)
            # cv2.putText(processed_frame, 
                       # f"Status: {'ANALYZING' if signal_processor.is_ready else 'WARMUP'}",
                       # (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            if frame_count <= number_of_frames_per_calculation:
                csc = apply_color_space_conversion(frame)
                spatial_temporal_map.append(csc)

                frame_count += 1
            elif frame_count > number_of_frames_per_calculation:
                tdn = apply_time_domain_normalization(spatial_temporal_map)
                noised = add_white_noise(tdn)

                # [С, R, T]
                processed_data = DCTiDCT(noised, number_of_stripes=4, fps=30)
                # [1, C, R, T]
                data = np.expand_dims(processed_data, axis=0)

                data_tensor = torch.tensor(data, dtype=torch.float32)

                with torch.no_grad():
                    bvp_signal = model(data_tensor)

                sig = bvp_signal[0]  

                peaks, _ = find_peaks(sig, distance=0.5*30, height=np.percentile(sig, 75))

                num_beats = len(peaks) - 1
                duration_s = len(sig) / 30
                bpm = num_beats / duration_s * 60
                print(f"HR ≈ {bpm:.1f} уд/мин")

                frame_count = 0
                spatial_temporal_map = []
            
            
            key = cv2.waitKey(1)   
            if key == ord('q') or key == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




