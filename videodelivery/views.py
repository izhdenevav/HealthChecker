import base64
import json
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from collections import deque
import numpy as np
import torch
import cv2
from scipy.signal import find_peaks

from . import facepoints
from . import data_preprocessing
from . import decompose_module
from . import srrn
from . import face_processor
from . import signal_processing

global_processor = None

class FrameProcessor:
    def __init__(self):
        self.N = 15
        self.landmark_buffers = [deque(maxlen=self.N) for _ in range(6)]
        self.yaw_buffer = deque(maxlen=self.N)
        self.pitch_buffer = deque(maxlen=self.N)
        self.roll_buffer = deque(maxlen=self.N)
        self.frames_per_calculation = 300
        self.frames_cnt = 0
        self.spatial_temporal_map = []  

        self.model = srrn.SRRN(in_channels=3, R=4, T=self.frames_per_calculation)
        self.model.load_state_dict(torch.load('./srrn_best.pth', map_location=torch.device('cpu')))
        # self.model.eval()

        self.bpm = 0.0
        self.face_processor = face_processor.FaceProcessor()
        self.signal_processor = signal_processing.SignalProcessor(max_frames=10000)
        self.br = None
        self.br_value = 0.0
        self.cg_filtered = np.zeros(30000000, dtype=np.float32)

        self.model_points = np.array([
            (0.0, 0.0, 0.0),        # Кончик носа
            (0.0, -330.0, -65.0),   # Подбородок
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)

        focal_length = 1108.5
        center = (640, 360)
        self.camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

    def process(self, frame):
        raw_points = facepoints.extract_all_landmarks(frame)
        my_points = facepoints.get_face_tracking_landmarks(raw_points) if raw_points else []

        if len(my_points) != 6:
            return "Недостаточно точек (меньше 6)", self.bpm, self.br_value

        for j, pt in enumerate(my_points):
            self.landmark_buffers[j].append(pt)

        if not all(len(buf) > 0 for buf in self.landmark_buffers):
            return "Ожидание данных", self.bpm, self.br_value

        image_points = np.array([
            np.mean(self.landmark_buffers[j], axis=0) for j in range(6)
        ], dtype=np.float64)

        try:
            success, rvec, _ = cv2.solvePnP(self.model_points, image_points, self.camera_matrix, None, flags=cv2.SOLVEPNP_SQPNP)
            if not success:
                return "Ошибка PnP", self.bpm, self.br_value
        except Exception:
            return "Ошибка solvePnP", self.bpm, self.br_value

        rmat, _ = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        yaw, pitch, roll = angles[1], angles[0], angles[2]

        if pitch > 90: pitch = 180 - pitch
        elif pitch < -90: pitch = -(180 + pitch)

        self.yaw_buffer.append(yaw)
        self.pitch_buffer.append(pitch)
        self.roll_buffer.append(roll)

        sy, sp, sr = np.mean(self.yaw_buffer) + 10, np.mean(self.pitch_buffer), np.mean(self.roll_buffer)

        is_correct = abs(sy) <= 30 and abs(sp) <= 26 and abs(sr) <= 30
        position = "Правильно!" if is_correct else "Измените положение!"

        if is_correct:
            if self.frames_cnt < self.frames_per_calculation:
                print(self.frames_cnt)
                csc = data_preprocessing.apply_color_space_conversion(frame)
                self.spatial_temporal_map.append(csc)
                self.frames_cnt += 1
            else:
                tdn = data_preprocessing.apply_time_domain_normalization(self.spatial_temporal_map)
                noised = data_preprocessing.add_white_noise(tdn)
                processed = decompose_module.DCTiDCT(noised, number_of_stripes=4, fps=30)
                data = np.expand_dims(processed, axis=0)
                data_tensor = torch.tensor(data, dtype=torch.float32)

                with torch.no_grad():
                    bvp = self.model(data_tensor)[0]
                    peaks, _ = find_peaks(bvp, distance=0.5 * 30, height=np.percentile(bvp, 75))
                    if len(peaks) > 1:
                        bpm = (len(peaks) - 1) / (len(bvp) / 30) * 60
                        self.bpm = bpm

                self.frames_cnt = 0
                self.spatial_temporal_map.clear()

            processed_frame, rois = self.face_processor.process_frame(frame)
            if rois:
                filtered_value, self.br = self.signal_processor.process(frame, rois)
                if self.br is not None:
                    print("changed")
                    self.br_value = self.br
                    _, self.cg_filtered = self.signal_processor.get_all_data()

        return position, self.bpm, self.br_value, self.cg_filtered

# === VIEWS ===

def index(request):
    return render(request, 'index.html')

@csrf_exempt
def process_frame(request):
    global global_processor
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid method'}, status=405)
    
    try:
        if global_processor is None:
            global_processor = FrameProcessor()

        data = json.loads(request.body)

        if 'image' not in data:
            return JsonResponse({'error': 'No image field in request'}, status=400)

        try:
            img_data = base64.b64decode(data['image'])
        except Exception as e:
            return JsonResponse({'error': f'Base64 decode failed: {str(e)}'}, status=400)

        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return JsonResponse({'error': 'Failed to decode image'}, status=400)

        position, hr, breathing, cg_filtered = global_processor.process(frame)

        return JsonResponse({
            'position': position,
            'hr': hr,
            'breathing': breathing,
            'cg_filtered': cg_filtered[-300:].tolist()
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)

@login_required
def measurements(request):
    return render(request, 'videodelivery/measurements.html')
