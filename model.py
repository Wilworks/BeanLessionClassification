import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, DEVICE

class BeanLesionClassifier:
    def __init__(self, model_name='googlenet', pretrained=True):
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = None
        self.device = torch.device(DEVICE)

    def build_model(self, freeze_backbone=False):
        """Build and return the model"""
        if self.model_name == 'googlenet':
            self.model = models.googlenet(weights='DEFAULT' if self.pretrained else None)

            if freeze_backbone:
                # Freeze all layers except the final fully connected layer
                for param in self.model.parameters():
                    param.requires_grad = False

            # Replace the final fully connected layer
            self.model.fc = nn.Linear(in_features=1024, out_features=NUM_CLASSES)

        self.model = self.model.to(self.device)
        return self.model

    def save_model(self, filepath):
        """Save model state dict"""
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        """Load model state dict"""
        if self.model is None:
            self.build_model()
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()
        return self.model
