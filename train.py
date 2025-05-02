import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from srrn import SRRN
from dataset import loader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SRRN(in_channels=3, R=4, T=300).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.L1Loss()

    train_loader = loader

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            # [B, 3, 4, 300]
            X_batch = X_batch.to(device)
            # [B, ]
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            # [B, ]
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}  Train Loss: {avg_loss:.3f}")

        scheduler.step(avg_loss)

    torch.save(model.state_dict(), "srrn_best.pth")
    print("Model saved to srrn_best.pth")

if __name__ == "__main__":
    main()
