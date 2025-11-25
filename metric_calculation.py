import numpy as np
from collections import defaultdict
import cv2
from pathlib import Path


def _vis_save_all(image_path, predictions, gt_labels, out_dir, note):
    import cv2
    from pathlib import Path

    img = cv2.imread(image_path)
    if img is None:
        return
    h, w = img.shape[:2]

    classes = {0: "person", 1: "forklift", 2: "fire"}

    # GT = green
    for gt in gt_labels:
        x1, y1, x2, y2 = gt['bbox']  # convert if yours are normalized
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(img, f"{classes[gt['class_id']]}", (int(x1), max(int(y1)-4, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    # PRED = red
    for pred in predictions:
        x1, y1, x2, y2 = pred['bbox']  # convert if normalized
        conf = pred.get('confidence')
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
        label = f"{classes[pred['class_id']]}" + (f" {conf:.2f}" if conf is not None else "")
        cv2.putText(img, label, (int(x1), min(int(y1)+14, h-1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

    out_dir.mkdir(parents=True, exist_ok=True)
    # out_name = f"{Path(image_path).stem}_{note}.jpg"
    out_name = f"{Path(image_path).stem}.jpg"
    cv2.imwrite(str(out_dir / out_name), img)


def _vis_save(image_path, predictions, gt_labels, class_id, out_dir, note):
    img = cv2.imread(image_path)
    if img is None:
        return
    h, w = img.shape[:2]

    classes = {0: "person", 1: "forklift", 2: "fire"}

    # draw GT (green)
    for gt in gt_labels:
        if gt['class_id'] != class_id:
            continue
        x1, y1, x2, y2 = gt['bbox']
        # if coords are normalized, uncomment the next 2 lines:
        # x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(img, f"{classes[class_id]}", (int(x1), max(int(y1)-4, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

    # draw PRED (red)
    for pred in predictions:
        if pred['class_id'] != class_id:
            continue
        x1, y1, x2, y2 = pred['bbox']
        conf = pred.get('confidence', None)
        # if coords are normalized, uncomment the next 2 lines:
        # x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
        label = f"{classes[class_id]}" + (f" {conf:.2f}" if conf is not None else "")
        cv2.putText(img, label, (int(x1), min(int(y1)+14, h-1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

    out_name = f"{Path(image_path).stem}_{classes[class_id]}_{note}.jpg"
    cv2.imwrite(str(out_dir / out_name), img)


def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes in xyxy format"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def match_predictions_to_gt(predictions, gt_labels, iou_threshold, class_id):
    """Match predictions to ground truth for a specific class in one image"""
    # Filter by class
    class_preds = [p for p in predictions if p['class_id'] == class_id]
    class_gts = [g for g in gt_labels if g['class_id'] == class_id]
    
    if not class_gts:
        return 0, len(class_preds), 0  # All predictions are FP
    
    if not class_preds:
        return 0, 0, len(class_gts)  # All GT are FN
    
    # Sort predictions by confidence (highest first)
    class_preds = sorted(class_preds, key=lambda x: x['confidence'], reverse=True)
    
    # Track which GT boxes have been matched
    gt_matched = [False] * len(class_gts)
    tp = 0
    fp = 0
    
    for pred in class_preds:
        best_iou = 0
        best_gt_idx = -1
        
        # Find best matching GT box
        for gt_idx, gt in enumerate(class_gts):
            if gt_matched[gt_idx]:
                continue
            
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if match is good enough
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1
    
    fn = len(class_gts) - sum(gt_matched)
    return tp, fp, fn

def evaluate_detections(gt_prediction_pairs, OUT_FN_DIR, OUT_FP_DIR):
    """
    Evaluate object detection performance
    
    Args:
        gt_prediction_pairs: List of dictionaries with keys:
            - 'image_path': path to image
            - 'predictions': list of dicts with 'bbox', 'confidence', 'class_id'
            - 'gt_labels': list of dicts with 'bbox', 'class_id'
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Collect all unique classes
    all_classes = set()
    for pair in gt_prediction_pairs:
        for pred in pair['predictions']:
            all_classes.add(pred['class_id'])
        for gt in pair['gt_labels']:
            all_classes.add(gt['class_id'])

    
    # Initialize metrics storage
    class_metrics_50 = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    class_metrics_50_95 = defaultdict(list)
    
    # Process each image separately
    for pair in gt_prediction_pairs:
        image_path = pair['image_path']
        predictions = pair['predictions']
        gt_labels = pair['gt_labels']
        had_fn, had_fp = False, False

        _vis_save_all(image_path, predictions, gt_labels, Path("visual_results/ssh_final/"), "fn")

        # If no predictions and no GT, skip
        if not predictions and not gt_labels:
            continue
        # If only GT, all are FN
        if not predictions and gt_labels:
            for gt in gt_labels:
                class_metrics_50[gt['class_id']]['fn'] += 1
            had_fn = True
            # _vis_save_all(image_path, predictions, gt_labels, OUT_FN_DIR, "fn")
            continue
        # If only predictions, all are FP
        if predictions and not gt_labels:
            for pred in predictions:
                class_metrics_50[pred['class_id']]['fp'] += 1
            had_fp = True
            # _vis_save_all(image_path, predictions, gt_labels, OUT_FP_DIR, "fp")
            continue
        
        # Process each class in this image
        for class_id in all_classes:
            # Calculate metrics at IoU 0.5
            tp_50, fp_50, fn_50 = match_predictions_to_gt(predictions, gt_labels, 0.5, class_id)
            class_metrics_50[class_id]['tp'] += tp_50
            class_metrics_50[class_id]['fp'] += fp_50
            class_metrics_50[class_id]['fn'] += fn_50
            # If there are any errors for this class in this image,
            # accumulate flags (no saving here)
            if fn_50 > 0: had_fn = True
            if fp_50 > 0: had_fp = True
            
            # Calculate metrics for mAP50-95 (IoU 0.5 to 0.95)
            iou_thresholds = np.arange(0.5, 1.0, 0.05)
            aps_for_image_class = []
            
            for iou_thresh in iou_thresholds:
                tp, fp, fn = match_predictions_to_gt(predictions, gt_labels, iou_thresh, class_id)
                if tp + fp > 0:
                    precision = tp / (tp + fp)
                else:
                    precision = 0 if fn > 0 else 1  # No predictions but no GT = perfect
                aps_for_image_class.append(precision)
            
            class_metrics_50_95[class_id].append(np.mean(aps_for_image_class))

        # save ONCE per image if there's any error
        if had_fp:
            note = "fp"         # prefer FP if both happen
            out_dir = OUT_FP_DIR
            # _vis_save_all(image_path, predictions, gt_labels, out_dir, note)
    
    # Calculate final metrics
    total_tp_50 = sum(metrics['tp'] for metrics in class_metrics_50.values())
    total_fp_50 = sum(metrics['fp'] for metrics in class_metrics_50.values())
    total_fn_50 = sum(metrics['fn'] for metrics in class_metrics_50.values())
    
    # Calculate mAP50 (average precision per class at IoU 0.5)
    class_aps_50 = []
    for class_id in all_classes:
        metrics = class_metrics_50[class_id]
        if metrics['tp'] + metrics['fp'] > 0:
            precision = metrics['tp'] / (metrics['tp'] + metrics['fp'])
            class_aps_50.append(precision)
        elif metrics['fn'] == 0:  # No GT for this class
            class_aps_50.append(1.0)  # Perfect if no GT and no predictions
    
    mAP50 = np.mean(class_aps_50) if class_aps_50 else 0
    
    # Calculate mAP50-95 (average over classes and IoU thresholds)
    class_aps_50_95 = []
    for class_id in all_classes:
        if class_metrics_50_95[class_id]:  # Has some data for this class
            class_aps_50_95.append(np.mean(class_metrics_50_95[class_id]))
    
    mAP50_95 = np.mean(class_aps_50_95) if class_aps_50_95 else 0
    
    # Overall precision, recall, F1
    precision = total_tp_50 / (total_tp_50 + total_fp_50) if (total_tp_50 + total_fp_50) > 0 else 0
    recall = total_tp_50 / (total_tp_50 + total_fn_50) if (total_tp_50 + total_fn_50) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    total_metrics = {
        'mAP50': mAP50,
        'mAP50_95': mAP50_95,
        'precision': precision,
        'recall': recall,
        'TP': total_tp_50,
        'FP': total_fp_50,
        'FN': total_fn_50,
        'F1_score': f1_score
    }

    person_metric = class_metrics_50[0]
    precision = person_metric['tp'] / (person_metric['tp'] + person_metric['fp']) if person_metric['tp'] + person_metric['fp'] > 0 else 0.0
    recall = person_metric['tp'] / (person_metric['tp'] + person_metric['fn']) if person_metric['tp'] + person_metric['fn'] > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    person_metrics = {
        'precision' : precision, 
        'recall' : recall,
        'TP' : person_metric['tp'],
        'FP' : person_metric['fp'],
        'FN' : person_metric['fn'],
        'F1_score' : f1_score
    }

    forklift_metric = class_metrics_50[1]
    precision = forklift_metric['tp'] / (forklift_metric['tp'] + forklift_metric['fp']) if forklift_metric['tp'] + forklift_metric['fp'] > 0 else 0.0
    recall = forklift_metric['tp'] / (forklift_metric['tp'] + forklift_metric['fn']) if forklift_metric['tp'] + forklift_metric['fn'] > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    forklift_metrics = {
        'precision' : precision, 
        'recall' : recall,
        'TP' : forklift_metric['tp'],
        'FP' : forklift_metric['fp'],
        'FN' : forklift_metric['fn'],
        'F1_score' : f1_score
    }

    fire_metric = class_metrics_50[2]
    precision = fire_metric['tp'] / (fire_metric['tp'] + fire_metric['fp']) if fire_metric['tp'] + fire_metric['fp'] > 0 else 0.0
    recall = fire_metric['tp'] / (fire_metric['tp'] + fire_metric['fn']) if fire_metric['tp'] + fire_metric['fn'] > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fire_metrics = {
        'precision' : precision, 
        'recall' : recall,
        'TP' : fire_metric['tp'],
        'FP' : fire_metric['fp'],
        'FN' : fire_metric['fn'],
        'F1_score' : f1_score
    }
    
    return total_metrics, person_metrics, forklift_metrics, fire_metrics