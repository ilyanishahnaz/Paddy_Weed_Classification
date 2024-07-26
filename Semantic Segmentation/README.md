# Weed & Paddy Classification Multiclass Semantic Segmentation Model

## Overview
This repository contains the code and resources for a semantic segmentation model focused on multiclass classification of weeds and paddy. The main objective is to classify and segment different classes within RGB images, specifically distinguishing between three different types of weed (broadleaved, grass and sedges) and paddy. The repository includes preprocessing scripts, data augmentation techniques, and model evaluation methods. Three models are compared, UNet + Resnet50 encoder, PSPNet + Resnet50 encoder & SegNet

## Directory Contents
1. **FYP_Preprocess_Eval.ipynb**: Jupyter notebook demonstrating various preprocessing methods, data augmentation, training, and evaluation.
   - **Background Removal**: Comparing HSV color space and YCbCr color space methods.
   - **Demo Pixel Mapping**: Changing the ground truth to pixel labels 0, 1, 2, 3.
   - **Data Augmentation**: Techniques applied to enhance the dataset.
   - **Single Class Images Training**: Training on images containing a single class.
   - **Multiclass Images Training**: Training on images containing multiple classes.
   - **Model Evaluation**: Assessing the performance of the model.

2. **Dataset & Weights.txt**: Contains the Google Drive directory link to access the dataset and the pre-trained weights used in the model.

3. **Pixelmask.py**: Script to iterate over a folder and apply pixel mapping. Run this script once to generate pixel-mapped images.

4. **remove_bg2.py**: Script to iterate over a folder and perform background removal. Run this script once to preprocess images by removing the background.

## Usage Instructions

### 1. Preprocessing and Evaluation
Open and run the `FYP_Preprocess_Eval.ipynb` notebook to:
- Compare background removal techniques using HSV and YCbCr color spaces.
- Apply pixel mapping to change ground truth to pixel labels (0, 1, 2, 3).
- Perform data augmentation on the dataset.
- Train the model on single class and multiclass images.
- Evaluate the model's performance.

### 2. Accessing Dataset and Weights
Refer to the `Dataset & Weights.txt` file for the Google Drive link to download the dataset and pre-trained weights.

### 3. Pixel Mapping
Run the `Pixelmask.py` script to iterate over a folder and apply pixel mapping to the images. This script converts the ground truth annotations to pixel labels.

```bash
python Pixelmask.py
```

### 4. Background Removal
Run the `remove_bg2.py` script to iterate over a folder and remove the background from the images using the chosen background removal technique.

```bash
python remove_bg2.py
```

## Requirements
- Python 3.10
- Jupyter Notebook
- OpenCV
- NumPy
- TensorFlow / Keras 2.10
- segmentation_models library

You can install the required libraries using pip:

```bash
pip install opencv-python numpy tensorflow==2.10 keras==2.10 segmentation-models
```

## Acknowledgements
This project is part of a Final Year Project (FYP) focused on developing a robust semantic segmentation model for weed and paddy classification. 
