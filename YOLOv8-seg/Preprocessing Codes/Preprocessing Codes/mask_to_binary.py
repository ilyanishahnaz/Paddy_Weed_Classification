import os
import cv2
import numpy as np

def convert_images_to_binary(input_folder, output_folder):
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.JPG', '.png'))]

    for image_file in image_files:
        # Construct the full paths
        input_path = os.path.join(input_folder, image_file)
        output_path = os.path.join(output_folder, image_file)

        # Read the image
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        kernel = np.ones((2, 2), np.uint8)

        # Perform dilate operation
        dilated_image = cv2.dilate(binary_image, kernel, iterations=1)
        mask = np.zeros_like(dilated_image, dtype=np.uint8)
        contours, _ = cv2.findContours(dilated_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for i, c in enumerate(contours):
            areaContour = cv2.contourArea(c)
            if areaContour < 1000 :
                continue
            cv2.drawContours(mask, contours, i, (255), thickness=cv2.FILLED)
        outImg_filtered = cv2.bitwise_and(dilated_image, mask)


        # Save the binary image
        cv2.imwrite(output_path, outImg_filtered)

    print("Binary image saved")

# Specify the input and output folders
# input_folder_paddy = "paddy_weed_segmentation/input/train_premask/paddy"
# output_folder_paddy = "paddy_weed_segmentation/input/train_masks/paddy"

input_folder_sedges = "paddy_weed_segmentation/input/train_premask/sedges"
output_folder_sedges = "paddy_weed_segmentation/input/train_masks/sedges"

input_folder_broad = "paddy_weed_segmentation/input/train_premask/broad"
output_folder_broad = "paddy_weed_segmentation/input/train_masks/broad"

input_folder_grass = "paddy_weed_segmentation/input/train_premask/grass"
output_folder_grass = "paddy_weed_segmentation/input/train_masks/grass"


# input_folder_paddy = "dataset/masks/paddy"
# output_folder_paddy = "dataset/binary/paddy"

# input_folder_sedges = "dataset/masks/sedges"
# output_folder_sedges = "dataset/binary/sedges"

# input_folder_broad = "dataset/masks/broad"
# output_folder_broad = "dataset/binary/broad"

# input_folder_grass = "dataset/masks/grass"
# output_folder_grass = "dataset/binary/grass"

# Convert images to binary and save them
# convert_images_to_binary(input_folder_paddy, output_folder_paddy)
convert_images_to_binary(input_folder_sedges, output_folder_sedges)
convert_images_to_binary(input_folder_broad, output_folder_broad)
convert_images_to_binary(input_folder_grass, output_folder_grass)
