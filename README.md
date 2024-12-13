# MNIST Digit Recognition with CNN

![Build Status](https://github.com/aniruddhbalram97/practice-first-ai-model-with-actions/actions/workflows/ml-pipeline.yml/badge.svg)

## Description

This project implements a Convolutional Neural Network (CNN) for recognizing handwritten digits from the MNIST dataset. The model is designed to achieve high accuracy while maintaining a low number of parameters.

### Project Structure
├── .github
│ └── workflows
│ └── ml-pipeline.yml # GitHub Actions CI/CD pipeline
├── augmented_samples # Directory for saving augmented images
├── data # Directory for storing the MNIST dataset
├── model.py # Contains the CNN model definition and training logic
├── requirements.txt # Python package dependencies
├── test_model.py # Unit tests for the model
├── visualize_augmentations.py # Script to visualize augmented images
└── README.md # Project documentation


### CNN Model Structure

The CNN model consists of the following layers:
- **Convolutional Layers**: Three convolutional layers with ReLU activations and batch normalization.
- **Global Average Pooling**: Reduces the spatial dimensions while retaining important features.
- **Fully Connected Layer**: Outputs the final predictions for the 10 digit classes (0-9).

## Build and Run Instructions

### Individual Model Training

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model**:
   ```bash
   python model.py
   ```

### Running Tests

To run the tests, execute the following command:
```bash
pytest test_model.py
```

This will run all the unit tests defined in `test_model.py`, including checks for model parameters, input/output shapes, accuracy, and the presence of ReLU activations.

### Visualizing Augmented Images

To visualize the augmented images, run:
```bash
python visualize_augmentations.py
```

This script will display the augmented images in a window.
