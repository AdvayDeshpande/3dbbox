import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import os
from one import train_dataloader, val_dataloader, test_dataloader
from model_new import *

# Define the train function outside the main block
def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, checkpoint_dir='checkpoints', model_name='model.pth'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.train()

    for epoch in range(num_epochs):
        total_train_loss = 0.0
        for batch in train_loader:
            image = batch["image"].to(device)  # (batch, 3, 224, 224)
            point_cloud = batch["point_cloud"].to(device)  # (batch, 3, 224, 224)
            mask = batch["mask"].unsqueeze(1).to(device)  # (batch, 1, 224, 224)
            bbox = batch["bbox"].to(device)  # (batch, 8, 3)

            optimizer.zero_grad()
            predicted_bbox = model(image, point_cloud, mask)
            loss = criterion(predicted_bbox, bbox)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation Step
        avg_val_loss = validate(model, val_loader, criterion)

        # Logging
        logging.info("", extra={"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save checkpoint and weights after each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        save_checkpoint(model, optimizer, epoch, checkpoint_path)

        # Save model weights
        if (epoch + 1) % 10 == 0:  # Save weights every 10 epochs
            model_weights_path = os.path.join(checkpoint_dir, f'{model_name}_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), model_weights_path)
            print(f"Model weights saved to {model_weights_path}")


# Validation Loop
def validate(model, val_loader, criterion):
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"].to(device)
            point_cloud = batch["point_cloud"].to(device)
            mask = batch["mask"].unsqueeze(1).to(device)
            bbox = batch["bbox"].to(device)

            predicted_bbox = model(image, point_cloud, mask)
            loss = criterion(predicted_bbox, bbox)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    return avg_val_loss

# Save checkpoint function
def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(filename="training_log.txt", level=logging.INFO,
                        format="%(asctime)s - Epoch %(epoch)d - Train Loss: %(train_loss).4f - Val Loss: %(val_loss).4f")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Model
    model = MultiModal3DBoundingBoxModel().to(device)

    # Loss function and optimizer
    criterion = nn.SmoothL1Loss(beta=1.0)  # Huber Loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train Model
    train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=10)
