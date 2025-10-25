import matplotlib.pyplot as plt
import numpy as np
import torch
from config import CLASS_NAMES

def plot_class_distribution(dataframe, title="Class Distribution"):
    """Plot class distribution bar chart"""
    dataframe['category'].value_counts().plot.bar()
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.show()

def plot_training_history(loss_history, acc_history, title="Training History"):
    """Plot training loss and accuracy"""
    epochs = range(1, len(loss_history) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_history, label='Training Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_history, label='Training Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()

def show_sample_images(dataset, n_rows=3, n_cols=3):
    """Display sample images from dataset"""
    f, axarr = plt.subplots(n_rows, n_cols, figsize=(10, 10))
    for i in range(n_rows):
        for j in range(n_cols):
            idx = np.random.randint(0, len(dataset))
            image, label = dataset[idx]

            # Convert tensor to numpy and permute dimensions for plotting
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy().transpose(1, 2, 0)

            # Denormalize if needed (assuming ImageNet normalization)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)

            axarr[i, j].imshow(image)
            axarr[i, j].set_title(f'Class: {CLASS_NAMES[label]}')
            axarr[i, j].axis('off')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    """Plot confusion matrix using seaborn"""
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
