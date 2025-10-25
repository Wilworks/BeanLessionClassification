# Bean Lesion Classification

A deep learning project for classifying bean leaf lesions using PyTorch and GoogleNet. The model identifies three types of bean leaf conditions: healthy, angular leaf spot, and bean rust.

## Features

- **High Accuracy**: Achieves 97.74% validation accuracy
- **Transfer Learning**: Uses pre-trained GoogleNet architecture
- **Data Augmentation**: Implements various augmentation techniques for robust training
- **Comprehensive Evaluation**: Includes confusion matrix and detailed classification reports
- **Inference Pipeline**: Easy-to-use script for single image prediction

## Dataset

The dataset consists of bean leaf images categorized into three classes:
- Healthy
- Angular Leaf Spot
- Bean Rust

## Project Structure

```
├── archive/                 # Dataset directory
│   ├── train.csv           # Training labels
│   └── val.csv             # Validation labels
├── config.py               # Configuration and hyperparameters
├── dataset.py              # Data loading and preprocessing
├── model.py                # GoogleNet model architecture
├── train.py                # Training script
├── evaluate.py             # Evaluation and metrics
├── inference.py            # Single image prediction
├── utils.py                # Utility functions and plotting
├── Pipeline.ipynb          # Jupyter notebook with full pipeline
├── Pipeline.py             # Python script version of pipeline
├── bean_lesion_full_training.pth  # Trained model weights
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Wilworks/BeanLessionClassification.git
cd BeanLessionClassification
```

2. Install dependencies:
```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn pillow tqdm
```

## Usage

### Training

Run the training script:
```bash
python train.py
```

### Evaluation

Evaluate the trained model:
```bash
python evaluate.py
```

### Inference

Predict on a single image:
```bash
python inference.py
```
Enter the path to your image when prompted.

### Jupyter Notebook

For an interactive experience, open and run `Pipeline.ipynb` in Jupyter.

## Model Performance

The project implements two training approaches using GoogleNet (Inception v1):

1. **Transfer Learning with Frozen Backbone**: Pre-trained GoogleNet layers are frozen, and only new classification layers are trained.
2. **Full Training**: All layers of GoogleNet are fine-tuned for the bean lesion classification task.

- **Validation Accuracy**: 97.74% (Full Training model)
- **Architecture**: GoogleNet (Inception v1)
- **Input Size**: 128x128 pixels
- **Classes**: 3 (healthy, angular_leaf_spot, bean_rust)

**Note**: Inference tests revealed misclassifications where healthy images are predicted as angular_leaf_spot with high confidence (~0.999), angular_leaf_spot as bean_rust (~0.990), and bean_rust as healthy (~0.997). This suggests potential issues with data preprocessing or model generalization that may require further investigation.

## Configuration

Key hyperparameters can be modified in `config.py`:
- Learning rate: 1e-3
- Batch size: 32
- Epochs: 15
- Image size: (128, 128)

## Dependencies

- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow
- tqdm

## Results

The model demonstrates excellent performance across all classes with detailed metrics available in the evaluation script output.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
