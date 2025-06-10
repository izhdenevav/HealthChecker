from collections import deque

import numpy as np
import cv2
import torch
from scipy.signal import find_peaks

from data_preprocessing import apply_color_space_conversion, apply_time_domain_normalization, add_white_noise
from decompose_module import DCTiDCT
from srrn import SRRN

def main():
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




