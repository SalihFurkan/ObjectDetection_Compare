import os
import sys
from tqdm import tqdm
from pathlib import Path

def convert_yolo_to_box(label_path):
    """
    Read YOLO format labels and return list of boxes with class IDs.
    
    Args:
        label_path (str): Path to YOLO format label file
        
    Returns:
        list: List of dictionaries with 'class_id' and 'bbox' [x_center, y_center, width, height]
    """
    boxes = []
    
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    class_id = int(parts[0])
                    bbox = [float(x) for x in parts[1:5]]  # x_center, y_center, width, height
                    
                    boxes.append({
                        'class_id': class_id,
                        'bbox': bbox
                    })
    except FileNotFoundError:
        print(f"Label file not found: {label_path}")
    except Exception as e:
        print(f"Error reading label file: {e}")
    
    return boxes


def prepare_image_label_pairs(IMAGE_FOLDER, LABEL_FOLDER):
    """
    Prepares pairs of image and label file paths from the specified folders.
    Args:
        IMAGE_FOLDER (str): Path to the folder containing image files.
        LABEL_FOLDER (str): Path to the folder containing label files.
    Raises:
        ValueError: If either IMAGE_FOLDER or LABEL_FOLDER does not exist or is not a directory.
    Returns:
        None
    Note:
        This function iterates over the images in IMAGE_FOLDER and constructs corresponding label file paths
        in LABEL_FOLDER by matching the image stem with a '.txt' extension.
    """

    if not os.path.exists(IMAGE_FOLDER) or not os.path.isdir(IMAGE_FOLDER):
        raise ValueError(f"Image folder '{IMAGE_FOLDER}' does not exist or is not a directory.")
    
    if not os.path.exists(LABEL_FOLDER) or not os.path.isdir(LABEL_FOLDER):
        raise ValueError(f"Image folder '{LABEL_FOLDER}' does not exist or is not a directory.")
    
    IMAGE_PATHS = sorted(os.listdir(IMAGE_FOLDER))
    
    dataset = []
    for img_path in tqdm(IMAGE_PATHS, desc="Preparing the Dataset"):
        img_full_path = os.path.join(IMAGE_FOLDER, img_path)
        path = Path(img_path).stem
        label_path = os.path.join(LABEL_FOLDER, path + ".txt")
        labels = convert_yolo_to_box(label_path)
        dataset.append({
            'image_path': img_full_path,
            'labels': labels
        })
    return dataset

