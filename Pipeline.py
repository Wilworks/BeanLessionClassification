#!/usr/bin/env python
# coding: utf-8

# Dependencies

# In[66]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import List
from torchvision.transforms import transforms 
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torchvision import models
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
import random
import time
import copy


# In[67]:


#Lets use mps if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")


# In[68]:


train_df = pd.read_csv('archive/train.csv')
val_df = pd.read_csv('archive/val.csv')

train_df['image:FILE'] = 'archive/' + train_df['image:FILE']
val_df['image:FILE'] = 'archive/' + val_df['image:FILE']
print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")


# In[69]:


print(train_df.head())
print(val_df.head())


# In[70]:


train_df['category'].value_counts().plot.bar()
plt.title('Training Set Class Distribution')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.show()


# In[71]:


#Shapes
print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {val_df.shape}")


# In[72]:


#class names
class_names = train_df['category'].unique().tolist()
print(f"Class names: {class_names}")


# In[73]:


#Map class names to indices with actual class names
class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}  
print(f"Class to index mapping: {class_to_idx}")    


# In[93]:


train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# train_dataset = CustomImageDataset(train_df, transform=train_transform)
# val_dataset = CustomImageDataset(val_df, transform=val_transform)
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Dataset Customization

# In[94]:


class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = [class_to_idx[cat] for cat in self.dataframe['category'].tolist()]

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.labels[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image).to(device)  # Remove the /255.0 !
        return image, label
train_dataset = CustomImageDataset(train_df, transform=train_transform)
val_dataset = CustomImageDataset(val_df, transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")   



# View Some images

# In[95]:


n_rows = 3
n_cols = 3
f, axarr = plt.subplots(n_rows, n_cols, figsize=(10,10))
for i in range(n_rows):
    for j in range(n_cols):
        image = train_dataset[np.random.randint(0, train_dataset.__len__())][0].cpu()

        # Debug: check the actual values
        print(f"Image shape: {image.shape}, Min: {image.min()}, Max: {image.max()}")

        # Handle both grayscale and RGB
        if image.shape[0] == 1:  # Grayscale
            axarr[i,j].imshow(image.squeeze(), cmap='gray')
        else:  # RGB
            axarr[i,j].imshow(image.permute(1,2,0))

        axarr[i,j].axis('off')
plt.tight_layout()
plt.show()


# In[96]:


LR = 1e-3
EPOCHS = 15
BATCH_SIZE = 4


# In[97]:


train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")   


# In[98]:


google_net = models.googlenet(weights='DEFAULT')


# Training entire Weights without new layers

# In[99]:


for param in google_net.parameters():
    param.requires_grad = True


# In[100]:


google_net.fc


# In[101]:


num_classes = len(train_df['category'].unique())
google_net.fc = nn.Linear(in_features=1024, out_features=num_classes)
google_net = google_net.to(device)


# In[102]:


google_net.fc


# In[103]:


google_net.to(device)


# In[105]:


from tqdm import tqdm

loss_fun = nn.CrossEntropyLoss()
optimizer = Adam(google_net.parameters(), lr=LR)
total_loss_train_plot = []
total_acc_train_plot = []

for epoch in range(EPOCHS):
    total_acc_train = 0
    total_loss_train = 0

    train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')

    for batch_idx, (inputs, labels) in enumerate(train_bar, 1):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = google_net(inputs)
        train_loss = loss_fun(outputs, labels)
        total_loss_train += train_loss.item()
        train_loss.backward()

        train_acc = (torch.argmax(outputs, axis=1) == labels).sum().item()
        total_acc_train += train_acc
        optimizer.step()

        # Show running averages (more informative)
        avg_loss = total_loss_train / batch_idx
        avg_acc = total_acc_train / (batch_idx * inputs.size(0)) * 100
        train_bar.set_postfix({'avg_loss': f'{avg_loss:.4f}', 'avg_acc': f'{avg_acc:.2f}%'})

    total_loss_train_plot.append(round(total_loss_train/1000, 4))
    total_acc_train_plot.append(round(total_acc_train/train_dataset.__len__()*100, 4))

    print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {total_loss_train/len(train_loader):.4f}, Training Accuracy: {total_acc_train/train_dataset.__len__()*100:.2f}%")


# In[108]:


with torch.no_grad():
    total_acc_val = 0
    total_loss_val = 0
    all_preds = []
    all_labels = []
    for inputs, labels in val_loader:
        predictions = google_net(inputs)

        acc = (torch.argmax(predictions, axis=1) == labels).sum().item()
        total_acc_val += acc

        val_loss = loss_fun(predictions, labels)
        total_loss_val += val_loss.item()

        all_preds.extend(torch.argmax(predictions, axis=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    print(f"Validation Loss: {total_loss_val/len(val_loader):.4f}, Validation Accuracy: {total_acc_val/val_dataset.__len__()*100:.2f}%")

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Full Training')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

# Save the trained model
torch.save(google_net.state_dict(), 'bean_lesion_full_training.pth')
print("Full training model saved as 'bean_lesion_full_training.pth'")


# Transfer Learning

# In[106]:


Google_Transfer = models.googlenet(weights='DEFAULT')

for param in Google_Transfer.parameters():
    param.requires_grad = False
Google_Transfer.fc = nn.Linear(in_features=1024, out_features=num_classes)
Google_Transfer = Google_Transfer.to(device)
Google_Transfer.fc.to(device)


# In[107]:


from tqdm import tqdm

loss_fun = nn.CrossEntropyLoss()
optimizer = Adam(Google_Transfer.parameters(), lr=LR)
total_loss_train_plot = []
total_acc_train_plot = []

for epoch in range(EPOCHS):
    total_acc_train = 0
    total_loss_train = 0

    train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')

    for batch_idx, (inputs, labels) in enumerate(train_bar, 1):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = Google_Transfer(inputs)
        train_loss = loss_fun(outputs, labels)
        total_loss_train += train_loss.item()
        train_loss.backward()

        train_acc = (torch.argmax(outputs, axis=1) == labels).sum().item()
        total_acc_train += train_acc
        optimizer.step()

        # Show running averages (more informative)
        avg_loss = total_loss_train / batch_idx
        avg_acc = total_acc_train / (batch_idx * inputs.size(0)) * 100
        train_bar.set_postfix({'avg_loss': f'{avg_loss:.4f}', 'avg_acc': f'{avg_acc:.2f}%'})

    total_loss_train_plot.append(round(total_loss_train/1000, 4))
    total_acc_train_plot.append(round(total_acc_train/train_dataset.__len__()*100, 4))

    print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {total_loss_train/len(train_loader):.4f}, Training Accuracy: {total_acc_train/train_dataset.__len__()*100:.2f}%")


# In[109]:


with torch.no_grad():
    total_acc_val = 0
    total_loss_val = 0
    all_preds = []
    all_labels = []
    for inputs, labels in val_loader:
        predictions = Google_Transfer(inputs)

        acc = (torch.argmax(predictions, axis=1) == labels).sum().item()
        total_acc_val += acc

        val_loss = loss_fun(predictions, labels)
        total_loss_val += val_loss.item()

        all_preds.extend(torch.argmax(predictions, axis=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    print(f"Validation Loss: {total_loss_val/len(val_loader):.4f}, Validation Accuracy: {total_acc_val/val_dataset.__len__()*100:.2f}%")

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Transfer Learning')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

# Save the trained model
torch.save(Google_Transfer.state_dict(), 'bean_lesion_transfer_learning.pth')
print("Transfer learning model saved as 'bean_lesion_transfer_learning.pth'")


# In[ ]:




