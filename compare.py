import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path

import pandas as pd

from inference_rfdetr import rfdetr_inference
from inference_yolo import yolo_inference
from dataset_read import prepare_image_label_pairs
from metric_calculation import evaluate_detections

import gc
import copy
import torch

def clear_memory():
    gc.collect()  # Python garbage collection
    if hasattr(torch, 'mps') and torch.mps.is_available():
        torch.mps.empty_cache()  # Clear MPS cache on Apple Silicon
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

def rfdetr_imp(model_id, model_weight, dataset, imgsz, imgsz_s, conf_threshold, iou_threshold, OUT_FN_DIR, OUT_FP_DIR):
    print("="*50)
    print("RF-DETR Inference")
    print("="*50)

    print(model_id, model_weight)
    if "checkpoint" in model_weight:
        pretrained = False
    else:
        pretrained = True
    gt_prediction_pairs_rfdetr, time_metrics_rfdetr = rfdetr_inference(model_id, model_weight, dataset, pretrained, imgsz, imgsz_s, conf_threshold, iou_threshold)

    total_metrics_rfdetr, person_metrics_rfdetr, forklift_metrics_rfdetr, fire_metrics_rfdetr = evaluate_detections(gt_prediction_pairs_rfdetr, OUT_FN_DIR, OUT_FP_DIR)

    metrics = {}
    metrics.update(time_metrics_rfdetr)
    metrics.update({key: float(value) if not isinstance(value, int) else value for key, value in total_metrics_rfdetr.items()})
    metrics.update({key + "_person": value for key, value in person_metrics_rfdetr.items()})
    metrics.update({key + "_forklift": value for key, value in forklift_metrics_rfdetr.items()})
    metrics.update({key + "_fire": value for key, value in fire_metrics_rfdetr.items()})

    return metrics

def yolo_imp(model_id, model_weight, dataset, imgsz, conf_threshold, iou_threshold, OUT_FN_DIR, OUT_FP_DIR):
    print("="*50)
    print("YOLO Inference")
    print("="*50)
    print(model_id, model_weight)
    gt_prediction_pairs_yolo, time_metrics_yolo = yolo_inference(model_weight, dataset, imgsz, conf_threshold, iou_threshold)

    total_metrics_yolo, person_metrics_yolo, forklift_metrics_yolo, fire_metrics_yolo = evaluate_detections(gt_prediction_pairs_yolo, OUT_FN_DIR, OUT_FP_DIR)

    metrics = {}
    metrics.update(time_metrics_yolo)
    metrics.update({key: float(value) if not isinstance(value, int) else value for key, value in total_metrics_yolo.items()})
    metrics.update({key + "_person": value for key, value in person_metrics_yolo.items()})
    metrics.update({key + "_forklift": value for key, value in forklift_metrics_yolo.items()})
    metrics.update({key + "_fire": value for key, value in fire_metrics_yolo.items()})

    return metrics


import argparse

parser = argparse.ArgumentParser(description="Compare object detector models on a dataset.")
parser.add_argument('--image_folder', type=str, required=True, help='Path to the folder containing test images')
parser.add_argument('--label_folder', type=str, required=True, help='Path to the folder containing test labels')
parser.add_argument("--models", nargs='+', type=str, help="List of model IDs to run (e.g., rfdetr_small yolov8s_merged)")
parser.add_argument("--model_weights", nargs='+', type=str, help="List of model weights to load (e.g., rfdetr_small yolov8s_merged)")
parser.add_argument("--output_csv", type=str, default="comparison.csv", help="The output csv file to save the comparison")
parser.add_argument("--conf_threshold", type=float, default=0.4, help="Confidence Threshold for detection")
parser.add_argument("--iou_threshold", type=float, default=0.8, help="IoU Threshold for detection")
parser.add_argument("--imgsz", type=int, default=640, help="Image size for YOLO detection")
parser.add_argument("--imgsz_rfnano", type=int, default=384, help="Image size for RFDETR Nano detection")
parser.add_argument("--imgsz_rfsmall", type=int, default=512, help="Image size for RFDETR Small detection")
args = parser.parse_args()

IMAGE_FOLDER    = args.image_folder
LABEL_FOLDER    = args.label_folder
MODELS          = args.models
MODEL_WEIGHTS   = args.model_weights
OUTPUT_CSV      = args.output_csv
CONF_THRESHOLD  = args.conf_threshold
IOU_THRESHOLD   = args.iou_threshold
IMGSZ           = args.imgsz
IMGSZ_RF        = args.imgsz_rfnano
IMGSZ_RFs       = args.imgsz_rfsmall

# Prepare Dataset: returns dataset as list of dict containing "image_path" and "labels"
dataset = prepare_image_label_pairs(IMAGE_FOLDER, LABEL_FOLDER)

columns = ["MODEL_ID", "MODEL_WEIGHTS", "CONF_THRESHOLD", "IOU_THRESHOLD", "IMGSZ"]
# Check if OUTPUT_CSV exists.
if os.path.exists(OUTPUT_CSV):
    comparison_csv = pd.read_csv(OUTPUT_CSV)
else:
    comparison_csv = pd.DataFrame(columns=columns)

for model_id, model_weight in zip(MODELS,MODEL_WEIGHTS):
    clear_memory()  # Clear before each model
    dataset_copy = copy.deepcopy(dataset)

    OUT_FN_DIR = Path(f"visual_results/realtime_test/{model_weight.split("/")[-1].split(".")[0]}_conf_{CONF_THRESHOLD}/fn")
    OUT_FP_DIR = Path(f"visual_results/realtime_test/{model_weight.split("/")[-1].split(".")[0]}_conf_{CONF_THRESHOLD}/fp")
    OUT_FN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FP_DIR.mkdir(parents=True, exist_ok=True)

    if "rfdetr" in model_id.lower():
        metrics = rfdetr_imp(model_id, model_weight, dataset_copy, IMGSZ_RF, IMGSZ_RFs, CONF_THRESHOLD, IOU_THRESHOLD, OUT_FN_DIR, OUT_FP_DIR)
    else:
        metrics = yolo_imp(model_id, model_weight, dataset_copy, IMGSZ, CONF_THRESHOLD, IOU_THRESHOLD, OUT_FN_DIR, OUT_FP_DIR)

    # If Model ID YOLO, Img Size is IMGSZ, if RFDETR, Img Size is IMGSZ_RF and IMGSZ_RFs
    if "rfdetrnano" in model_id.lower():
        image_size = IMGSZ_RF
    elif "rfdetrsmall" in model_id.lower():
        image_size = IMGSZ_RFs
    else:
        image_size = IMGSZ

    row = {
        "MODEL_ID": model_id,
        "MODEL_WEIGHTS": model_weight, 
        "CONF_THRESHOLD": CONF_THRESHOLD,
        "IOU_THRESHOLD" : IOU_THRESHOLD,
        "IMGSZ": image_size,
    }
    row.update(metrics)

    comparison_csv = pd.concat([comparison_csv, pd.DataFrame([row])], ignore_index=True)
    clear_memory()  # Clear before each model

comparison_csv.to_csv(OUTPUT_CSV, index=False)


