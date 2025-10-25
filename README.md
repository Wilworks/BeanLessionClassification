# Bean Disease Classification Project

## Overview
This project implements a deep learning pipeline for classifying bean diseases using PyTorch and GoogLeNet architecture. The model is trained on a dataset of bean leaf images to identify different disease categories.

## Dataset
- **Training samples**: 1,034 images
- **Validation samples**: 133 images
- **Classes**: 3 (Healthy, Angular Leaf Spot, Bean Rust)
- **Image size**: 128x128 pixels

## Model Architecture
- **Base Model**: GoogLeNet (Inception v1)
- **Training Approaches**:
  1. Full training (all layers trainable)
  2. Transfer learning (frozen base layers, trainable classifier)

## Key Features
- Image preprocessing with data augmentation
- ImageNet normalization
- MPS acceleration support (Apple Silicon)
- Progress tracking with tqdm
- Training/validation metrics plotting

## Training Results
### Full Training Approach
- **Final Training Accuracy**: 90.72%
- **Training Loss**: 0.2524
- Achieved convergence with decreasing loss over 15 epochs

### Transfer Learning Approach
- **Final Training Accuracy**: 90.72%
- **Training Loss**: 0.2524
- Faster convergence due to pre-trained features

## Dependencies
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- PIL (Pillow)
- Scikit-learn
- tqdm

## Usage
1. Ensure dataset is in `archive/` directory with `train.csv` and `val.csv`
2. Run the Jupyter notebook `Pipeline.ipynb`
3. Models will train sequentially and generate performance plots

## Files
- `Pipeline.ipynb`: Main training notebook
- `Pipeline.py`: Python script version
- `archive/`: Dataset directory
- `README.md`: This documentation

## Performance Notes
- Models achieve ~90% training accuracy
- Image normalization and proper data loading were critical fixes
- Device mismatch errors resolved by ensuring tensors on same device
- Batch size optimized to 32 for stability

## Future Improvements
- Implement test set evaluation
- Add model checkpointing
- Experiment with other architectures (ResNet, EfficientNet)
- Hyperparameter tuning
- Cross-validation
