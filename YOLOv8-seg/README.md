# Paddy and Weed Classification Using YOLOv8-Seg for Instance Segmentation

## Overview
This repository contains the code and resources for an instance segmentation model using YOLOv8-Seg for the classification and segmentation of paddy and weeds. The main goal is to accurately identify and segment different instances of paddy and weeds in agricultural images.

## Directory Contents

1. **mask_to_binary.py**: Script to convert ground truth color annotations to binary masks for instance segmentation.

2. **binary_to_coco.py**: Script to convert binary masks to COCO format annotations.

3. **coco_to_yolo.py**: Script to convert COCO format annotations to YOLO format for training with YOLOv8-Seg.

4. **yolov8-seg.ipynb**: Jupyter notebook for training and evaluating the YOLOv8-Seg model.

## Usage Instructions

### Preprocessing
1. **Convert Ground Truth to Binary Masks**
   Run the `mask_to_binary.py` script to convert the ground truth color annotations to binary masks.

   ```bash
   python mask_to_binary.py
   ```

2. **Convert Binary Masks to COCO Format**
   Run the `binary_to_coco.py` script to convert the binary masks to COCO format annotations.

   ```bash
   python binary_to_coco.py
   ```

3. **Convert COCO Format to YOLO Format**
   Run the `coco_to_yolo.py` script to convert the COCO format annotations to YOLO format.

   ```bash
   python coco_to_yolo.py
   ```

### Training with YOLOv8-Seg
Once the preprocessing is complete and the data is in YOLO format, you can train the YOLOv8-Seg model using the provided Jupyter notebook.

1. **Install the required library**:

   ```bash
   pip install ultralytics
   ```

2. **Open and run the Jupyter notebook**:
   Use the `yolov8-seg.ipynb` notebook for training and evaluating the YOLOv8-Seg model. The notebook contains code and instructions for setting up the dataset, training the model, and evaluating its performance.

   ```bash
   jupyter notebook yolov8-seg.ipynb
   ```

## Requirements
- Python 3.x
- ultralytics

Install the required library using pip:

```bash
pip install ultralytics
```

## Acknowledgements
This project focuses on developing an instance segmentation model for paddy and weed classification using YOLOv8-Seg. 
