import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from srrn import SRRN
from ubfc_dataset import UBFCrPPGDataset
from decompose_module import DCTiDCT

def compute_metrics(pred, target):
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    mae = mean_absolute_error(target_np, pred_np)
    rmse = np.sqrt(mean_squared_error(target_np, pred_np))
    pearson_r = np.corrcoef(pred_np.flatten(), target_np.flatten())[0, 1]
    return mae, rmse, pearson_r

# def compute_metrics(pred, target):
#     pred_np = pred.cpu().numpy()
#     target_np = target.cpu().numpy()
#     mae = mean_absolute_error(target_np, pred_np)
#     rmse = np.sqrt(mean_squared_error(target_np, pred_np))
#     pearson_r = np.corrcoef(pred_np.flatten(), target_np.flatten())[0, 1]

#     pearson_r_inv = np.corrcoef((-pred_np).flatten(), target_np.flatten())[0, 1]

#     if abs(pearson_r_inv) > abs(pearson_r):
#         pearson_r = pearson_r_inv
#         pred = -pred

#     return mae, rmse, pearson_r

def pearson_loss(pred, target):
    pred_mean = pred.mean(dim=-1, keepdim=True)
    target_mean = target.mean(dim=-1, keepdim=True)
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    corr = (pred_centered * target_centered).sum(dim=-1) / (
        torch.sqrt((pred_centered ** 2).sum(dim=-1)) * torch.sqrt((target_centered ** 2).sum(dim=-1)) + 1e-8
    )
    return -corr.mean()

def combined_loss(pred, target):
    l1_loss = nn.L1Loss()(pred, target)
    pearson = pearson_loss(pred, target)
    return l1_loss + 0.2 * pearson

# class OversamplingSampler:
#     def __init__(self, dataset, hr_ranges):
#         self.dataset = dataset
#         self.hr_ranges = hr_ranges
#         self.indices = []
#         for hr_min, hr_max in hr_ranges:
#             indices = []
#             for i in range(len(dataset)):
#                 hr_values = dataset[i]['hr_values'].numpy()  # Среднее HR для видео
#                 mean_hr = np.mean(hr_values)
#                 if hr_min <= mean_hr < hr_max:
#                     indices.append(i)
#             self.indices.append(indices)

#     def __iter__(self):
#         sampled_indices = []
#         max_len = max(len(indices) for indices in self.indices)
#         for indices in self.indices:
#             sampled = np.random.choice(indices, size=max_len, replace=True)
#             sampled_indices.extend(sampled)
#         np.random.shuffle(sampled_indices)
#         return iter(sampled_indices)

#     def __len__(self):
#         return sum(max(len(indices) for indices in self.indices) for _ in self.indices)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(),            
        logging.FileHandler("training.log")  
    ]
)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    data_dir = 'D:/Studying/Fourth/UBFC-rPPG' 
    T = 1387 
    batch_size = 4
    num_epochs = 50 
    number_of_stripes = 4

    print(f"Start to upload dataset from {data_dir}\n")
    train_dataset = UBFCrPPGDataset(data_dir, split='train', fps=30, max_frames=T)
    test_dataset = UBFCrPPGDataset(data_dir, split='test', fps=30, max_frames=T)
    print("Datasets ready!")

    # # Oversampling для балансировки по HR
    # hr_ranges = [(40, 60), (60, 80), (80, 100), (100, 120)]  # Пример диапазонов HR
    # print("Start to make a sampler...")
    # train_sampler = OversamplingSampler(train_dataset, hr_ranges)
    # print("Sampler ready!")

    print("Start to make a dataloaders...")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        # sampler=train_sampler,
        num_workers=4,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    print("Dataloaders ready!")

    model = SRRN(in_channels=3, R=4, T=T).to(device)
    optimizer = Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    # criterion = combined_loss
    criterion = nn.L1Loss()

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            spatial_temporal_map = batch['spatial_temporal_map'].to(device)  # [B, C=3, R=4, T=1387]
            ppg_signal = batch['ppg_signal'].to(device)  # [B, T=1387]

            batch_size = spatial_temporal_map.size(0)
            multi_band_signals = []
            for b in range(batch_size):
                st_map_np = spatial_temporal_map[b].permute(1, 2, 0).cpu().numpy()  # [R=4, T=1387, C=3]
                st_map_dict = {
                    'between_eyebrows_landmarks': st_map_np[0],
                    'nose_landmarks': st_map_np[1],
                    'left_cheek_landmarks': st_map_np[2],
                    'right_cheek_landmarks': st_map_np[3]
                }

                multi_band = DCTiDCT(st_map_dict, number_of_stripes=number_of_stripes, fps=30)
                multi_band_signals.append(multi_band)
            
            multi_band_signals = torch.from_numpy(np.stack(multi_band_signals)).float().to(device)  # [B, C=3, K=4, T=1387]
            # print(multi_band_signals.shape)

            optimizer.zero_grad()
            pred = model(multi_band_signals)  # [B, T=1387]
            print(f"before loss, pred: {pred.shape} ppg: {ppg_signal.shape}")
            min_len = min(pred.shape[1], ppg_signal.shape[1])
            pred = pred[:, :min_len]
            ppg_signal = ppg_signal[:, :min_len]
            loss = criterion(pred, ppg_signal)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item() * batch_size

        avg_train_loss = total_train_loss / len(train_dataset)

        all_preds = []
        all_targets = []

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                spatial_temporal_map = batch['spatial_temporal_map'].to(device)
                ppg_signal = batch['ppg_signal'].to(device)

                batch_size = spatial_temporal_map.size(0)
                multi_band_signals = []
                for b in range(batch_size):
                    st_map_np = spatial_temporal_map[b].permute(1, 2, 0).cpu().numpy()
                    st_map_dict = {
                        'between_eyebrows_landmarks': st_map_np[0],
                        'nose_landmarks': st_map_np[1],
                        'left_cheek_landmarks': st_map_np[2],
                        'right_cheek_landmarks': st_map_np[3]
                    }
                    multi_band = DCTiDCT(st_map_dict, number_of_stripes=number_of_stripes, fps=30)
                    multi_band_signals.append(multi_band)
                
                multi_band_signals = torch.from_numpy(np.stack(multi_band_signals)).float().to(device)
                
                pred = model(multi_band_signals)
                min_len = min(pred.shape[1], ppg_signal.shape[1])
                pred = pred[:, :min_len]
                ppg_signal = ppg_signal[:, :min_len]

                all_preds.append(pred.cpu())
                all_targets.append(ppg_signal.cpu())

                loss = criterion(pred, ppg_signal)
                total_val_loss += loss.item() * batch_size

        avg_val_loss = total_val_loss / len(test_dataset)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        mae, rmse, pearson_r = compute_metrics(all_preds, all_targets)

        logging.info(f"Epoch {epoch+1}/{num_epochs} Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f} Val MAE: {mae:.4f}, RMSE: {rmse:.4f}, Pearson r: {pearson_r:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "srrn_best.pth")
            print("Model saved to srrn_best.pth")

if __name__ == "__main__":
    main()