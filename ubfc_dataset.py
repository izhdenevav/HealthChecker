import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from data_preprocessing import apply_color_space_conversion, apply_time_domain_normalization, add_white_noise

class UBFCrPPGDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, fps=30, max_frames=1387):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.fps = fps
        self.max_frames = max_frames
        self.subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('subject') and os.path.isdir(os.path.join(data_dir, d))])
        self.num_subjects = len(self.subjects)
        
        if split == 'train':
            self.subjects = self.subjects[:12]
        elif split == 'test':
            self.subjects = self.subjects[12:15]
        else:
            raise ValueError("split должен быть 'train' или 'test'")
        
        self.video_paths = []
        self.gt_paths = []
        
        for subject in self.subjects:
            outer_dir = os.path.join(data_dir, subject)
            inner_dir = os.path.join(outer_dir, subject.split('-')[0])
            if os.path.isdir(inner_dir):
                video_path = os.path.join(inner_dir, 'vid.avi')
                gt_path = os.path.join(inner_dir, 'ground_truth.txt')
                if os.path.exists(video_path) and os.path.exists(gt_path):
                    self.video_paths.append(video_path)
                    self.gt_paths.append(gt_path)
        
        self.num_samples = len(self.video_paths)
        if self.num_samples == 0:
            raise ValueError("Не найдено видео или ground truth файлов в указанной директории")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):

        video_path = self.video_paths[idx]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        # print(f"Видео {video_path}: {total_frames} кадров, FPS: {fps}, Длительность: {total_frames/fps:.3f} сек")
        
        frames = []
        frame_count = 0
        while frame_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            frame_count += 1
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"Видео пустое: {video_path}")
        if len(frames) < self.max_frames:
            frames.extend([frames[-1]] * (self.max_frames - len(frames)))
        elif len(frames) > self.max_frames:
            frames = frames[:self.max_frames]
        
        spatial_temporal_map = []

        for frame in frames:
            facial_means_yuv = apply_color_space_conversion(frame)
            if facial_means_yuv is None:
                facial_means_yuv = {
                    'between_eyebrows_landmarks': [0, 0, 0],
                    'nose_landmarks': [0, 0, 0],
                    'left_cheek_landmarks': [0, 0, 0],
                    'right_cheek_landmarks': [0, 0, 0]
                }
            spatial_temporal_map.append(facial_means_yuv)
        
        normalized_map = apply_time_domain_normalization(spatial_temporal_map)
        noisy_map = add_white_noise(normalized_map, noise_std=0.05)
        spatial_temporal_map_tensor = torch.from_numpy(noisy_map).float()
        
        gt_path = self.gt_paths[idx]
        with open(gt_path, 'r') as f:
            data = [float(x) for x in f.read().split()]
            # print(f"Длина ground truth: {len(data)}")

        before_T = len(data) // 3
        ppg_signal = data[:before_T]
        hr_values = data[before_T:2*before_T]
        timesteps = data[2*before_T:3*before_T]

        T = self.max_frames
        
        if len(ppg_signal) < T:
            ppg_signal.extend([ppg_signal[-1]] * (T - len(ppg_signal)))
        elif len(ppg_signal) > T:
            ppg_signal = ppg_signal[:T]
        
        if len(hr_values) < T:
            hr_values.extend([hr_values[-1]] * (T - len(hr_values)))
        elif len(hr_values) > T:
            hr_values = hr_values[:T]
        
        if len(timesteps) < T:
            timesteps.extend([timesteps[-1]] * (T - len(timesteps)))
        elif len(timesteps) > T:
            timesteps = timesteps[:T]
        
        ppg_tensor = torch.tensor(ppg_signal, dtype=torch.float32)
        hr_tensor = torch.tensor(hr_values, dtype=torch.float32)
        timesteps_tensor = torch.tensor(timesteps, dtype=torch.float32)
        
        if self.transform is not None:
            sample = {
                'spatial_temporal_map': spatial_temporal_map_tensor,
                'ppg_signal': ppg_tensor,
                'hr_values': hr_tensor,
                'timesteps': timesteps_tensor
            }
            sample = self.transform(sample)
            return sample
        
        return {
            'spatial_temporal_map': spatial_temporal_map_tensor,
            'ppg_signal': ppg_tensor,
            'hr_values': hr_tensor,
            'timesteps': timesteps_tensor
        }
