import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

from config import LR, EPOCHS, BATCH_SIZE, TRAIN_CSV, VAL_CSV, DATA_DIR
from dataset import CustomImageDataset, get_transforms, load_data
from model import BeanLesionClassifier
from utils import plot_training_history

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Training loop"""
    model.train()
    total_loss_train_plot = []
    total_acc_train_plot = []

    for epoch in range(num_epochs):
        total_acc_train = 0
        total_loss_train = 0

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch_idx, (inputs, labels) in enumerate(train_bar, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            train_loss = criterion(outputs, labels)
            total_loss_train += train_loss.item()
            train_loss.backward()

            train_acc = (torch.argmax(outputs, axis=1) == labels).sum().item()
            total_acc_train += train_acc
            optimizer.step()

            # Show running averages
            avg_loss = total_loss_train / batch_idx
            avg_acc = total_acc_train / (batch_idx * inputs.size(0)) * 100
            train_bar.set_postfix({'avg_loss': f'{avg_loss:.4f}', 'avg_acc': f'{avg_acc:.2f}%'})

        total_loss_train_plot.append(round(total_loss_train/1000, 4))
        total_acc_train_plot.append(round(total_acc_train/len(train_loader.dataset)*100, 4))

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {total_loss_train/len(train_loader):.4f}, Training Accuracy: {total_acc_train/len(train_loader.dataset)*100:.2f}%")

    return total_loss_train_plot, total_acc_train_plot

def main():
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_df, val_df = load_data(TRAIN_CSV, VAL_CSV, DATA_DIR)
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # Create datasets
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)

    train_dataset = CustomImageDataset(train_df, transform=train_transform)
    val_dataset = CustomImageDataset(val_df, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Build model
    classifier = BeanLesionClassifier()
    model = classifier.build_model(freeze_backbone=False)  # Full training

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)

    # Train model
    loss_history, acc_history = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, device)

    # Plot training history
    plot_training_history(loss_history, acc_history, "Full Training")

    # Save model
    classifier.save_model('bean_lesion_full_training.pth')
    print("Model saved as 'bean_lesion_full_training.pth'")

if __name__ == "__main__":
    main()
