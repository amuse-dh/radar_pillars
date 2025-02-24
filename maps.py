import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

def normalize_class_name(class_name):

    if class_name.startswith('inf_'):
        class_name = class_name[4:]
    return class_name

def parse_box_from_line(line):
    """
    GT: class 0 0 0 0 0 0 0 height width length cx cy cz yaw
    Pred: class 0 0 0 0 0 0 0 width height length cx cy cz yaw confidence
    """
    try:
        parts = line.strip().split()
        if len(parts) < 15:  # minimum required fields
            return None
            
        box = {
            'class': normalize_class_name(parts[0]),
            'height': float(parts[8]),
            'width': float(parts[9]),
            'length': float(parts[10]),
            'cx': float(parts[11]),
            'cy': float(parts[12]),
            'cz': float(parts[13]),
            'yaw': float(parts[14])
        }
        
        # confidence is only in prediction
        if len(parts) > 15:
            box['confidence'] = float(parts[15])
            
        return box
            
    except (IndexError, ValueError) as e:
        print(f"Warning: Failed to parse line: {line}")
        return None

def calculate_3d_iou(box1, box2):
    if box1['class'] != box2['class']:
        return 0.0
        
    box1_min = np.array([
        box1['cx'] - box1['length']/2,
        box1['cy'] - box1['width']/2,
        box1['cz'] - box1['height']/2
    ])
    box1_max = np.array([
        box1['cx'] + box1['length']/2,
        box1['cy'] + box1['width']/2,
        box1['cz'] + box1['height']/2
    ])
    
    box2_min = np.array([
        box2['cx'] - box2['length']/2,
        box2['cy'] - box2['width']/2,
        box2['cz'] - box2['height']/2
    ])
    box2_max = np.array([
        box2['cx'] + box2['length']/2,
        box2['cy'] + box2['width']/2,
        box2['cz'] + box2['height']/2
    ])
    
    intersect_min = np.maximum(box1_min, box2_min)
    intersect_max = np.minimum(box1_max, box2_max)
    
    if np.any(intersect_max < intersect_min):
        return 0.0
    
    intersect_volume = np.prod(intersect_max - intersect_min)
    box1_volume = np.prod(box1_max - box1_min)
    box2_volume = np.prod(box2_max - box2_min)
    
    iou = intersect_volume / (box1_volume + box2_volume - intersect_volume)
    
    return iou

def calculate_2d_iou(box1, box2):
    """
    calculate 2D IoU(only x, y plane)
    """
    if box1['class'] != box2['class']:
        return 0.0
        
    box1_min = np.array([
        box1['cx'] - box1['length']/2,
        box1['cy'] - box1['width']/2
    ])
    box1_max = np.array([
        box1['cx'] + box1['length']/2,
        box1['cy'] + box1['width']/2
    ])
    
    box2_min = np.array([
        box2['cx'] - box2['length']/2,
        box2['cy'] - box2['width']/2
    ])
    box2_max = np.array([
        box2['cx'] + box2['length']/2,
        box2['cy'] + box2['width']/2
    ])
    
    intersect_min = np.maximum(box1_min, box2_min)
    intersect_max = np.minimum(box1_max, box2_max)
    
    if np.any(intersect_max < intersect_min):
        return 0.0
    
    intersect_area = np.prod(intersect_max - intersect_min)
    box1_area = np.prod(box1_max - box1_min)
    box2_area = np.prod(box2_max - box2_min)
    
    iou = intersect_area / (box1_area + box2_area - intersect_area)
    
    return iou

def calculate_ap_from_recalls(precisions, recalls):
    """
    COCO style 101-point interpolation AP
    """
    if not precisions or not recalls:
        return 0.0
    
    if max(recalls) == 0:  # if all recalls are 0
        return 0.0
        
    recall_points = np.linspace(0, 1, 101)
    max_precisions = []
    
    for r in recall_points:
        precs = [p for rec, p in zip(recalls, precisions) if rec >= r]
        max_prec = max(precs) if precs else 0
        max_precisions.append(max_prec)
    
    ap = np.mean(max_precisions)
    return ap

def calculate_3d_map_for_folder(gt_folder, pred_folder, iou_threshold=0.1):
    gt_files = sorted(os.listdir(gt_folder))
    pred_files = sorted(os.listdir(pred_folder))
    
    print(f"Ground Truth 파일 수: {len(gt_files)}")
    print(f"Prediction 파일 수: {len(pred_files)}")
    
    class_stats = {}
    
    for gt_file in gt_files:
        if gt_file in pred_files:
            gt_boxes = []
            with open(os.path.join(gt_folder, gt_file), 'r') as f:
                for line in f:
                    box = parse_box_from_line(line)
                    if box:
                        gt_boxes.append(box)
            
            pred_boxes = []
            with open(os.path.join(pred_folder, gt_file), 'r') as f:
                for line in f:
                    box = parse_box_from_line(line)
                    if box:
                        pred_boxes.append(box)
            
            if not gt_boxes or not pred_boxes:
                continue
                
            for box in gt_boxes:
                cls = box['class']
                if cls not in class_stats:
                    class_stats[cls] = {
                        'total_gt': 0,
                        'predictions': []
                    }
                class_stats[cls]['total_gt'] += 1
            
            for cls in set(box['class'] for box in gt_boxes + pred_boxes):
                cls_gt_boxes = [box for box in gt_boxes if box['class'] == cls]
                cls_pred_boxes = [box for box in pred_boxes if box['class'] == cls]
                
                if not cls_gt_boxes:
                    for pred_box in cls_pred_boxes:
                        if cls not in class_stats:
                            class_stats[cls] = {'total_gt': 0, 'predictions': []}
                        class_stats[cls]['predictions'].append(
                            (pred_box['confidence'], False)
                        )
                    continue
                
                if not cls_pred_boxes:
                    continue
                
                iou_matrix = np.zeros((len(cls_gt_boxes), len(cls_pred_boxes)))
                for i, gt in enumerate(cls_gt_boxes):
                    for j, pred in enumerate(cls_pred_boxes):
                        iou = calculate_3d_iou(gt, pred)
                        iou_matrix[i, j] = iou
                
                cost_matrix = 1 - iou_matrix
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
                
                matched_pred_indices = set()
                for gt_idx, pred_idx in zip(row_indices, col_indices):
                    iou = iou_matrix[gt_idx, pred_idx]
                    if iou >= iou_threshold:
                        class_stats[cls]['predictions'].append(
                            (cls_pred_boxes[pred_idx]['confidence'], True)
                        )
                        matched_pred_indices.add(pred_idx)
                
                for pred_idx, pred_box in enumerate(cls_pred_boxes):
                    if pred_idx not in matched_pred_indices:
                        class_stats[cls]['predictions'].append(
                            (pred_box['confidence'], False)
                        )
    
    aps = []
    print(f"\n클래스별 결과 (3D IoU {iou_threshold} 기준):")
    
    total_metrics = {'tp': 0, 'fp': 0, 'fn': 0}
    
    for cls, stats in class_stats.items():
        if stats['total_gt'] == 0 or not stats['predictions']:
            continue
            
        sorted_predictions = sorted(stats['predictions'], 
                                 key=lambda x: x[0], 
                                 reverse=True)
        
        total_gt = stats['total_gt']
        tp = 0
        fp = 0
        precisions = []
        recalls = []
        
        for i, (conf, is_matched) in enumerate(sorted_predictions, 1):
            if is_matched:
                tp += 1
            else:
                fp += 1
            
            precision = tp / i
            recall = tp / total_gt
            
            precisions.append(precision)
            recalls.append(recall)
        
        # FN 계산
        fn = total_gt - tp

        total_metrics['tp'] += tp
        total_metrics['fp'] += fp
        total_metrics['fn'] += fn
        
        final_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        final_recall = tp / total_gt if total_gt > 0 else 0
        
        ap = calculate_ap_from_recalls(precisions, recalls)
        aps.append(ap)
        
    
    if total_metrics['tp'] + total_metrics['fp'] > 0:
        total_precision = total_metrics['tp'] / (total_metrics['tp'] + total_metrics['fp'])
        print(f"Total Precision: {total_precision:.4f}")
    
    if total_metrics['tp'] + total_metrics['fn'] > 0:
        total_recall = total_metrics['tp'] / (total_metrics['tp'] + total_metrics['fn'])
        print(f"Total Recall: {total_recall:.4f}")
    
    if aps:
        mAP = np.mean(aps)
        print(f"\n3D mAP: {mAP:.4f}")
        return mAP
    else:
        print("\n매칭된 클래스가 없습니다.")
        return 0.0


def post_process_predictions(cls_preds, reg_preds, config, confidence_threshold=0.5, top_k=20):
    """
    Args:
        cls_preds: (B, H, W, num_classes)
        reg_preds: (B, H, W, 7) - (cx, cy, cz, l, w, h, yaw)
        confidence_threshold: confidence score threshold
        top_k
    
    Returns:
        predictions: list of lists [class_name, w, h, l, cx, cy, cz, yaw, confidence]
    """
    batch_size = cls_preds.shape[0]
    predictions = []
    
    for b in range(batch_size):
        # Get confidence scores and class indices
        confidence, class_ids = torch.max(cls_preds[b].sigmoid(), dim=-1)
        
        # Get valid predictions (confidence > threshold)
        valid_mask = confidence > confidence_threshold
        
        # Get predictions for valid positions
        valid_confidence = confidence[valid_mask]
        valid_class_ids = class_ids[valid_mask]
        valid_reg = reg_preds[b][valid_mask]
        
        # top-k
        if len(valid_confidence) > 0:
            k = min(top_k, len(valid_confidence))
            top_k_scores, top_k_indices = torch.topk(valid_confidence, k)
            
            valid_confidence = valid_confidence[top_k_indices]
            valid_class_ids = valid_class_ids[top_k_indices]
            valid_reg = valid_reg[top_k_indices]
        
        frame_predictions = []
        for conf, cls_id, reg in zip(valid_confidence, valid_class_ids, valid_reg):
            cx, cy, cz, l, w, h, yaw = reg.tolist()
            class_name = config.class_names[cls_id.item()]
            
            # Format: class_name 0 0 0 0 0 0 0 height width length cx cy cz yaw confidence
            pred = [class_name] + [0]*7 + [h, w, l, cx, cy, cz, yaw, conf.item()]
            frame_predictions.append(pred)
        
        predictions.append(frame_predictions)
    
    return predictions


def save_predictions(predictions, save_dir, file_names):
    os.makedirs(save_dir, exist_ok=True)
    
    for file_name, frame_preds in zip(file_names, predictions):
        save_path = os.path.join(save_dir, f"{file_name}.txt")
        with open(save_path, 'w') as f:
            for pred in frame_preds:
                line = ' '.join(map(str, pred))
                f.write(line + '\n')