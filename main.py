import cv2
import numpy as np
import torch

from the_old_shit.data_preprocessing import apply_color_space_conversion
from the_old_shit.data_preprocessing import apply_time_domain_normalization
from the_old_shit.data_preprocessing import add_white_noise
from the_old_shit.decompose_module import DCTiDCT
from the_old_shit.srrn import SRRN

cap = cv2.VideoCapture('vid.avi')

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

spatial_temporal_map = []
frame_count = 0
number_of_frames_per_calculation = 300

srrn = SRRN()

try:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret: 
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if frame_count < number_of_frames_per_calculation:
            csc = apply_color_space_conversion(frame)
            spatial_temporal_map.append(csc)

            frame_count += 1
        else:
            tdn = apply_time_domain_normalization(spatial_temporal_map)
            noised = add_white_noise(tdn)
            processed_data = DCTiDCT(noised, number_of_stripes=4, fps=30)
            
            stm_data = np.expand_dims(noised, axis=0)
            multi_band_data = np.expand_dims(processed_data, axis=0)
            stm_tensor = torch.tensor(stm_data, dtype=torch.float32)
            multi_band_tensor = torch.tensor(multi_band_data, dtype=torch.float32)
            result = srrn(multi_band_tensor, stm_tensor)

            spatial_temporal_map = []
            frame_count = 0
        
        key = cv2.waitKey(1)   
        if key == ord('q') or key == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()