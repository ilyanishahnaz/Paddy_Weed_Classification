# Weed & Paddy Classification Project

## Overview
Weeds pose significant threats to the cultivation and development of paddy fields. Traditional weed suppression often relies on manual labor and intuition, resulting in time-consuming and less efficient processes. This project aims to develop robust classification models for paddy and weeds to provide farmers with actionable insights for efficient crop management, particularly in weed mapping. This method helps farmers accurately identify weed types and implement targeted herbicide spraying strategies.

## Project Approaches
This project implements and analyzes two main approaches for weed and paddy classification:

1. **Instance Segmentation Using YOLOv8-Seg**
2. **Semantic Segmentation Models**

### Instance Segmentation Using YOLOv8-Seg
Initial investigations include using YOLOv8-Seg to gauge the feasibility of the model on the selected dataset. YOLOv8-Seg is utilized to perform instance segmentation, identifying individual instances of weeds and paddy within the images.

#### Directory Contents
1. **mask_to_binary.py**: Script to convert ground truth color annotations to binary masks for instance segmentation.
2. **binary_to_coco.py**: Script to convert binary masks to COCO format annotations.
3. **coco_to_yolo.py**: Script to convert COCO format annotations to YOLO format for training with YOLOv8-Seg.
4. **yolov8-seg.ipynb**: Jupyter notebook for training and evaluating the YOLOv8-Seg model.

#### Usage Instructions
1. **Preprocessing**: Convert the ground truth color annotations to binary masks, then to COCO format, and finally to YOLO format using the provided scripts.
2. **Training**: Use the `yolov8-seg.ipynb` notebook to train and evaluate the YOLOv8-Seg model.

### Semantic Segmentation Models
Three semantic segmentation models are implemented and analyzed:
1. **PSPNet + ResNet50**
2. **UNet + ResNet50**
3. **SegNet**

The UNet + ResNet50 model shows the best performance in the overall metrics, achieving an accuracy of 96%.

#### Preprocessing Methods
- **Background Removal**: Comparing HSV color space and YCbCr color space methods.
- **Pixel Mapping**: Changing the ground truth to pixel labels (0, 1, 2, 3).
- **Data Augmentation**: Techniques applied to enhance the dataset.

#### Directory Contents
1. **FYP_Preprocess_Eval.ipynb**: Jupyter notebook demonstrating preprocessing methods, data augmentation, training, and evaluation as well as simple herbicide recommender based on the selected model.
2. **Dataset & Weights.txt**: Contains the Google Drive directory link to access the dataset and the pre-trained weights used in the model.
3. **Pixelmask.py**: Script to iterate over a folder and apply pixel mapping.
4. **remove_bg2.py**: Script to iterate over a folder and perform background removal.

## Requirements
- Python 3.x
- Jupyter Notebook
- OpenCV
- NumPy
- TensorFlow / Keras 2.10
- segmentation_models library
- ultralytics

Install the required libraries using pip:

```bash
pip install opencv-python numpy tensorflow==2.10 keras==2.10 segmentation-models ultralytics
```

## Acknowledgements
This project focuses on developing robust models for weed and paddy classification. 

