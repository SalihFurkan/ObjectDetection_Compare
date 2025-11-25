import os
import cv2
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

def yolo_inference(model_weights, image_label_pairs, imgsz=640, conf_threshold=0.5, iou_threshold=0.45):
    """
    Performs inference using the YOLO object detection model on a list of image-label pairs.

    Args:
        image_label_pairs (list of dictinory): A list where each element is a dictinory containing an image path and its corresponding labels info.
        conf_threshold (float): Confidence threshold for filtering detected objects.

    Returns:
        list: A list of detection results for each image, where each result contains detected objects above the confidence threshold.
    """

    model = YOLO(model_weights)

    total_time = 0.0
    num_images = 0

    gt_prediction_pairs = []

    for i, img_label_pair in enumerate(tqdm(image_label_pairs, desc="YOLO Inference")):

        img_name = img_label_pair["image_path"]
        labels = img_label_pair["labels"]

        gt_prediction_pair = {}

        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"{img_name} in not an image file")
            continue

        img_path = img_name
        image = Image.open(img_path).convert("RGB")

        # Get original dimensions
        orig_width, orig_height = image.size

        # Step 1: Calculate padding to make it square (letterbox)
        max_dim = max(orig_width, orig_height)
        padded_image = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        
        # Calculate padding offsets
        pad_left = (max_dim - orig_width) // 2
        pad_top = (max_dim - orig_height) // 2
        
        # Paste original image onto black canvas
        padded_image.paste(image, (pad_left, pad_top))

        # Step 2: Resize to 640x640
        image = padded_image.resize((imgsz, imgsz), Image.BILINEAR)

        if image is None:
            print(f"Could not read {img_path}")
            continue

        # Calculate scale factor
        scale = imgsz / max_dim

        start_time = time.time()
        detections = model(image, imgsz=imgsz, iou=iou_threshold, verbose=False)

        detect = []
        for result in detections:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            names = result.names
            for box, conf, class_id in zip(boxes, confs, class_ids):
                label = names[int(class_id)]
                
                # Adjust box coordinates back to original image
                x1, y1, x2, y2 = box
                # Remove resize scaling
                x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                # Remove padding
                x1, y1, x2, y2 = x1-pad_left, y1-pad_top, x2-pad_left, y2-pad_top
                
                adjusted_box = [x1, y1, x2, y2]
                
                if label == "person" and conf > conf_threshold:
                    detect.append({
                        "bbox": adjusted_box,
                        "confidence": conf,
                        "class_id": 0
                    })
                if (label == "truck" or label == "car" or label == "forklift") and conf > conf_threshold:
                    detect.append({
                        "bbox": adjusted_box,
                        "confidence": conf,
                        "class_id": 1
                    })
                if (label == "fire") and conf > conf_threshold:
                    detect.append({
                        "bbox": adjusted_box,
                        "confidence": conf,
                        "class_id": 2
                    })

        gt_labels = []
        for label in labels:
            class_id = label["class_id"]
            x_center, y_center, width, height = label["bbox"]
            # Convert from YOLO to absolute coordinates (original image size)
            x1 = (x_center - width/2) * orig_width
            y1 = (y_center - height/2) * orig_height
            x2 = (x_center + width/2) * orig_width
            y2 = (y_center + height/2) * orig_height

            label["bbox"] = [x1, y1, x2, y2]
            label["class_id"] = class_id

            gt_labels.append(label)

        end_time = time.time()
        total_time += (end_time - start_time)
        num_images += 1

        gt_prediction_pair["image_path"] = img_name
        gt_prediction_pair["predictions"] = detect
        gt_prediction_pair["gt_labels"] = gt_labels

        gt_prediction_pairs.append(gt_prediction_pair)

    mean_time = total_time / num_images if num_images > 0 else 0
    print(f"\nTotal Time: {total_time:.2f} seconds for {num_images} images")
    print(f"Mean Time per Image: {mean_time:.4f} seconds")

    return gt_prediction_pairs, {"mean_time": f"{mean_time:.4f}", "total_time": f"{total_time:.2f}", "num_images": num_images}