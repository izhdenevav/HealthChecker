import os
import json

import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from data_preprocessing import apply_color_space_conversion, apply_time_domain_normalization, add_white_noise
from decompose_module import DCTiDCT

class PulseSegmentDataset(Dataset):
    def __init__(self, segments_dir, ann_path, fps=30, stripe_count=4):
        
        self.segments = sorted(os.listdir(segments_dir))
        with open(ann_path, 'r') as f:
            self.annotations = json.load(f)
        self.dir = segments_dir
        self.fps = fps
        self.stripe_count = stripe_count

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        fname = self.segments[idx]
        hr = float(self.annotations[fname])

        cap = cv2.VideoCapture(os.path.join(self.dir, fname))
        frames = []

        while True:
            ret, frame = cap.read()

            if not ret: 
                break

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()

        st_map = [ apply_color_space_conversion(f) for f in frames ]
        tdn = apply_time_domain_normalization(st_map)
        noised = add_white_noise(tdn)

        X_np = DCTiDCT(noised, number_of_stripes=self.stripe_count, fps=self.fps)
        # [C, R, T]
        X = torch.tensor(X_np, dtype=torch.float32)

        y = torch.tensor(hr, dtype=torch.float32)
        return X, y

dataset = PulseSegmentDataset(
    segments_dir='../dataset/segments',
    ann_path='../dataset/results.json'
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)