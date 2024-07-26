import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# Input and output directories
input_folder = 'dataset_fyp/masks_separated/images/paddy'
output_folder = 'dataset_fyp/masks_separated/images/paddynew'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through the specified filenames
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".JPG"):
        # Construct the full input path
        input_path = os.path.join(input_folder, filename)
        
        # Load image
        image = cv2.imread(input_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        lower_green = np.array([30, 40, 40]) 
        upper_green = np.array([80, 255, 255])  

        # Apply the threshold to get a binary mask for green color
        hsv_mask = cv2.inRange(hsv_image, lower_green, upper_green)

        # Contouring
        mask1 = np.zeros_like(hsv_mask, dtype=np.uint8)
        contours, _ = cv2.findContours(hsv_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for i, c in enumerate(contours):
            areaContour = cv2.contourArea(c)
            if areaContour < 1000:
                continue
            cv2.drawContours(mask1, contours, i, (255), thickness=cv2.FILLED)
        outImg_filtered = cv2.bitwise_and(hsv_mask, mask1)

        result = cv2.bitwise_and(image, image, mask=outImg_filtered)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        # Construct the full output path
        output_path = os.path.join(output_folder, filename)

        # Save the masked image
        cv2.imwrite(output_path, result)

print("Done")
