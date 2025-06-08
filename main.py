import cv2

from the_old_shit.data_preprocessing import apply_color_space_conversion
from the_old_shit.data_preprocessing import apply_time_domain_normalization
from the_old_shit.data_preprocessing import add_white_noise
from the_old_shit.decompose_module import DCTiDCT

cap = cv2.VideoCapture('vid.avi')

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

spatial_temporal_map = []
frame_count = 0
number_of_frames_per_calculation = 300

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
        elif frame_count >= number_of_frames_per_calculation - 1:
            tdn = apply_time_domain_normalization(spatial_temporal_map)
            noised = add_white_noise(tdn)
            processed_data = DCTiDCT(noised, number_of_stripes=4, fps=30)
            

            frame_count = 0
        
        key = cv2.waitKey(1)   
        if key == ord('q') or key == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()