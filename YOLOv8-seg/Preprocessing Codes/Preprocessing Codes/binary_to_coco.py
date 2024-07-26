# import glob
# import json
# import os
# import cv2

# # Label IDs of the dataset representing different categories
# category_ids = {
#     "broad": 1,
#     "grass": 2,
#     "paddy": 3,
#     "sedges": 4,
# }

# MASK_EXT = 'JPG'
# ORIGINAL_EXT = 'JPG'
# image_id = 0
# annotation_id = 0

# def images_annotations_info(maskpath):
#     global image_id, annotation_id
#     annotations = []
#     images = []

#     # Iterate through categories and corresponding masks
#     for category in category_ids.keys():
#         for mask_image in glob.glob(os.path.join(maskpath, category, f'*.{MASK_EXT}')):
#             original_file_name = f'{os.path.basename(mask_image).split(".")[0]}.{ORIGINAL_EXT}'
#             mask_image_open = cv2.imread(mask_image)
            
#             # Get image dimensions
#             height, width, _ = mask_image_open.shape

#             # Create or find existing image annotation
#             if original_file_name not in map(lambda img: img['file_name'], images):
#                 image = {
#                     "id": image_id + 1,
#                     "width": width,
#                     "height": height,
#                     "file_name": original_file_name,
#                 }
#                 images.append(image)
#                 image_id += 1
#             else:
#                 image = [element for element in images if element['file_name'] == original_file_name][0]

#             # Find contours in the mask image
#             gray = cv2.cvtColor(mask_image_open, cv2.COLOR_BGR2GRAY)
#             _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#             contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

#             # Create annotation for each contour
#             for contour in contours:
#                 bbox = cv2.boundingRect(contour)
#                 area = cv2.contourArea(contour)
#                 segmentation = contour.flatten().tolist()

#                 annotation = {
#                     "iscrowd": 0,
#                     "id": annotation_id,
#                     "image_id": image['id'],
#                     "category_id": category_ids[category],
#                     "bbox": bbox,
#                     "area": area,
#                     "segmentation": [segmentation],
#                 }

#                 # Add annotation if area is greater than zero
#                 if area > 0:
#                     annotations.append(annotation)
#                     annotation_id += 1
        

#     return images, annotations, annotation_id


# def process_masks(mask_path, dest_json):
#     global image_id, annotation_id
#     image_id = 0
#     annotation_id = 0

#     # Initialize the COCO JSON format with categories
#     coco_format = {
#         "info": {},
#         "licenses": [],
#         "images": [],
#         "categories": [{"id": value, "name": key, "supercategory": key} for key, value in category_ids.items()],
#         "annotations": [],
#     }

#     # Create images and annotations sections
#     coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)

#     # Save the COCO JSON to a file
#     with open(dest_json, "w") as outfile:
#         json.dump(coco_format, outfile, sort_keys=True, indent=4)

#     print("Created %d annotations for images in folder: %s" % (annotation_cnt, mask_path))

# if __name__ == "__main__":
#     train_mask_path = "paddy_weed_segmentation/input/train_premask/"
#     train_json_path = "paddy_weed_segmentation/input/train_images/train.json"
#     process_masks(train_mask_path, train_json_path)

#     val_mask_path = "paddy_weed_segmentation/input/val_premask/"
#     val_json_path = "paddy_weed_segmentation/input/val_images/val.json"
#     process_masks(val_mask_path, val_json_path)

import glob
import json
import os
import cv2

# Label IDs of the dataset representing different categories
category_ids = {
    "broad": 1,
    "paddy": 2,
    "grass": 3,
    "sedges": 4,
}

MASK_EXT = 'JPG'
ORIGINAL_EXT = 'JPG'
image_id = 0
annotation_id = 0
smallest_area = float('inf')  # Initialize smallest_area to positive infinity

def images_annotations_info(maskpath):
    global image_id, annotation_id, smallest_area
    annotations = []
    images = []

    # Iterate through categories and corresponding masks
    for category in category_ids.keys():
        for mask_image in glob.glob(os.path.join(maskpath, category, f'*.{MASK_EXT}')):
            original_file_name = f'{os.path.basename(mask_image).split(".")[0]}.{ORIGINAL_EXT}'
            mask_image_open = cv2.imread(mask_image)
            
            # Get image dimensions
            height, width, _ = mask_image_open.shape

            # Create or find existing image annotation
            if original_file_name not in map(lambda img: img['file_name'], images):
                image = {
                    "id": image_id + 1,
                    "width": width,
                    "height": height,
                    "file_name": original_file_name,
                }
                images.append(image)
                image_id += 1
            else:
                image = [element for element in images if element['file_name'] == original_file_name][0]

            # Find contours in the mask image
            gray = cv2.cvtColor(mask_image_open, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

            # Create annotation for each contour
            for contour in contours:
                bbox = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                segmentation = contour.flatten().tolist()

                annotation = {
                    "iscrowd": 0,
                    "id": annotation_id,
                    "image_id": image['id'],
                    "category_id": category_ids[category],
                    "bbox": bbox,
                    "area": area,
                    "segmentation": [segmentation],
                }

                # Add annotation if area is greater than zero
                if area > 100:
                    annotations.append(annotation)
                    annotation_id += 1
                    
                    # Update smallest area if current area is smaller
                    smallest_area = min(smallest_area, area)

    return images, annotations, annotation_id


def process_masks(mask_path, dest_json):
    global image_id, annotation_id, smallest_area
    image_id = 0
    annotation_id = 0
    smallest_area = float('inf')  # Reset smallest_area for each dataset

    # Initialize the COCO JSON format with categories
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [],
        "categories": [{"id": value, "name": key, "supercategory": key} for key, value in category_ids.items()],
        "annotations": [],
    }

    # Create images and annotations sections
    coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)

    # Save the COCO JSON to a file
    with open(dest_json, "w") as outfile:
        json.dump(coco_format, outfile, sort_keys=True, indent=4)

    print("Created %d annotations for images in folder: %s" % (annotation_cnt, mask_path))
    print("Smallest annotation area:", smallest_area)

if __name__ == "__main__":

    train_mask_path = "dataset_fyp/train_masks/"
    train_json_path = "dataset_fyp/original_bgremoved/train/train.json"
    process_masks(train_mask_path, train_json_path)

    val_mask_path = "dataset_fyp/test_masks/"
    val_json_path = "dataset_fyp/original_bgremoved/train/test.json"
    process_masks(val_mask_path, val_json_path)
    # train_mask_path = "dataset/binary/"
    # train_json_path = "dataset/bg_images/train.json"
    # process_masks(train_mask_path, train_json_path)

    # val_mask_path = "paddy_weed_segmentation/input/val_masks/"
    # val_json_path = "paddy_weed_segmentation/input/val_images_bg_resized/val.json"
    # process_masks(val_mask_path, val_json_path)
