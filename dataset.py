import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from config import CLASS_NAMES, MEAN, STD, IMAGE_SIZE

# Map class names to indices
class_to_idx = {class_name: i for i, class_name in enumerate(CLASS_NAMES)}

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
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
            image = self.transform(image)
        return image, label

def get_transforms(augment=False):
    if augment:
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
    return transform

def load_data(train_csv, val_csv, data_dir):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    train_df['image:FILE'] = data_dir + train_df['image:FILE']
    val_df['image:FILE'] = data_dir + val_df['image:FILE']

    return train_df, val_df
