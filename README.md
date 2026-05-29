# Hand Sign Language Recognition

An AI-powered project that trains a convolutional neural network to recognize hand signs and convert them into the corresponding alphabet letter.

## Overview

This project uses a hand sign image dataset to build a classifier for alphabet gestures. The training flow includes:

- downloading the dataset
- preprocessing hand sign images
- applying data augmentation
- training a CNN model with TensorFlow/Keras
- evaluating performance on test data
- saving the trained model
- running a Streamlit demo for live image prediction

## Dataset

The notebook accesses the Kaggle dataset `ash2703/handsignimages` with separate `Train` and `Test` folders. The model is trained on 24 hand-sign classes representing letters such as A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, and Y.

## Model

The model is a sequential CNN with:

- convolutional layers and max pooling
- flattening followed by dense layers
- L2 regularization and dropout
- Adam optimizer
- categorical crossentropy loss

The input images are resized to `28×28` pixels and normalized to the `[0, 1]` range.

## Requirements

Install the required packages before running the notebook or demo:

```bash
pip install opendatasets kagglehub pandas numpy matplotlib scikit-learn tensorflow pillow opencv-python streamlit
```

## Usage

### 1. Run the notebook

Open `Hand_Sign_Language.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.

If using Colab, the notebook includes a badge at the top for quick access.

### 2. Download the dataset

The notebook downloads the dataset automatically from Kaggle using `kagglehub.dataset_download('ash2703/handsignimages')`.

If you run locally outside Kaggle, make sure the dataset is available in a folder containing the expected `Train` and `Test` directories.

### 3. Train the model

The notebook trains the CNN using data generators with augmentation for the training set and rescaling for the validation set.

### 4. Evaluate the model

After training, it evaluates accuracy on the test set and prints the results.

### 5. Save the model

The trained model is saved to `model.h5`.

### 6. Run the Streamlit demo

The notebook includes a Streamlit app that:

- loads `model.h5`
- allows uploading a hand sign image
- preprocesses the image
- predicts the alphabet letter
- displays confidence score

To run the demo from a Python script or terminal:

```bash
streamlit run app.py
```

> Note: The notebook contains the Streamlit app code; you may copy it into a separate `app.py` file to launch it directly.

## Notes

- The model supports 24 classes and intentionally omits letters with similar or unsupported hand shapes (such as J and Z).
- Use clear, evenly lit hand sign images for best prediction accuracy.
- Adjust training parameters such as epochs, batch size, and image augmentation to improve performance.

## Contact

If you want to extend this project, consider adding:

- support for more letters and numbers
- live camera input for real-time recognition
- a full web interface for text conversion
- model export to TensorFlow Lite or ONNX

