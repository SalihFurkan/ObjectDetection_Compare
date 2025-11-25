import os
import cv2
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
from rfdetr.util.coco_classes import COCO_CLASSES

def model_initialization(model_id, model_weigths, imgsz=384, imgsz_s=512):

    if "small" in model_id.lower():
        from rfdetr import RFDETRSmall
        model = RFDETRSmall(pretrain_weights=model_weigths, task='detect', resolution=imgsz_s)
    elif "nano" in model_id.lower():
        from rfdetr import RFDETRNano
        model = RFDETRNano(pretrain_weights=model_weigths, task='detect', resolution=imgsz)
    elif "base" in model_id.lower():
        from rfdetr import RFDETRBase
        model = RFDETRBase(pretrain_weights=model_weigths)
    elif "medium" in model_id.lower():
        from rfdetr import RFDETRMedium
        model = RFDETRMedium(pretrain_weights=model_weigths)
    else:
        from rfdetr import RFDETRSmall
        model = RFDETRSmall(pretrain_weights=model_weigths)

    return model



def rfdetr_inference(model_id, model_weights, image_label_pairs, pretrained=True, imgsz=384, imgsz_s=512, conf_threshold=0.5, iou_threshold=0.45):
    """
    Performs inference using the RFDETR object detection model on a list of image-label pairs.

    Args:
        image_label_pairs (list of dictinory): A list where each element is a dictinory containing an image path and its corresponding labels info.
        conf_threshold (float): Confidence threshold for filtering detected objects.

    Returns:
        list: A list of detection results for each image, where each result contains detected objects above the confidence threshold.
    """

    model = model_initialization(model_id, model_weights, imgsz=imgsz, imgsz_s=imgsz_s)

    PERSON_CATEGORY_ID = [k for k, v in COCO_CLASSES.items() if v == "person"][0]
    FORKLIFT_CATEGORY_ID = [k for k, v in COCO_CLASSES.items() if v == "truck" or v == "car"]

    total_time = 0.0
    num_images = 0

    gt_prediction_pairs = []

    for i, img_label_pair in enumerate(tqdm(image_label_pairs, desc="RFDETR Inference")):

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

        # Determine target size based on model
        target_size = imgsz_s if "small" in model_id.lower() else imgsz

        # Step 1: Calculate padding to make it square (letterbox)
        max_dim = max(orig_width, orig_height)
        padded_image = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        
        # Calculate padding offsets
        pad_left = (max_dim - orig_width) // 2
        pad_top = (max_dim - orig_height) // 2
        
        # Paste original image onto black canvas
        padded_image.paste(image, (pad_left, pad_top))

        # Step 2: Resize to target size
        image = padded_image.resize((target_size, target_size), Image.BILINEAR)

        # Calculate scale factor
        scale = target_size / max_dim

        start_time = time.time()
        detections = model.predict(image, threshold=conf_threshold, iou_threshold=iou_threshold)

        detect = []
        for i, box in enumerate(detections.xyxy):
            class_id = int(detections.class_id[i])
            confidence = detections.confidence[i]

            # Adjust box coordinates back to original image
            x1, y1, x2, y2 = box
            # Remove resize scaling
            x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
            # Remove padding
            x1, y1, x2, y2 = x1-pad_left, y1-pad_top, x2-pad_left, y2-pad_top
            
            adjusted_box = [x1, y1, x2, y2]

            if pretrained:
                if class_id == PERSON_CATEGORY_ID:
                    class_id = 0
                elif class_id in FORKLIFT_CATEGORY_ID:
                    class_id = 1
                else:
                    class_id = class_id
            else:
                class_id = class_id - 1

            detect.append({
                "bbox": adjusted_box,
                "confidence": confidence,
                "class_id": class_id
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