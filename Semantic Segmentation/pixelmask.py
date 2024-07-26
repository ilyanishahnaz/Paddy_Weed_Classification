import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# Define color ranges and corresponding values
color_ranges = {
    'black': ([0, 0, 0], [0, 0, 0]),
    'green': ([0, 120, 0], [50, 255, 50]),
    'blue': ([128, 0, 0], [255, 30, 30]),
    'red': ([0, 0, 128], [30, 30, 255]),
    'cyan': ([0, 128, 128], [100, 255, 255])
}

color_to_value = {
    'black': 0,
    'green': 1, #paddy
    'blue': 2, #broad
    'red': 3, #grass 
    'cyan': 4 #sedges
}

def extract_color_mask(image, color_name):
    lower, upper = color_ranges[color_name]
    lower = np.array(lower, dtype=np.uint8)
    upper = np.array(upper, dtype=np.uint8)
    mask = cv2.inRange(image, lower, upper)
    return mask

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.JPG')):
            # Load the mask image
            input_path = os.path.join(input_folder, filename)
            mask_image = cv2.imread(input_path)
            mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)

            # Initialize an empty mask
            labeled_mask = np.zeros(mask_image.shape[:2], dtype=np.uint8)

            # Loop through each color range and apply the corresponding pixel value
            for color_name, value in color_to_value.items():
                mask = extract_color_mask(mask_image, color_name)
                labeled_mask[mask > 0] = value

            # Save the labeled mask
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, labeled_mask)
            print(f"Processed and saved: {output_path}")

# Example usage
input_folder = 'dataset_fyp/annotated'
output_folder = 'dataset_fyp/masks'
process_folder(input_folder, output_folder)
