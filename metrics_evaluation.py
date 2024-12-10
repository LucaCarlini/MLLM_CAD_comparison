import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact, bootstrap
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar
from tqdm import tqdm
import pandas as pd
import seaborn as sns

from bar_plot import create_grouped_bar_plot_with_min_max_and_p_values


def load_json_files(base_folder_path):
    """
    Load JSON files from specified 'Positive' and 'Negative' subdirectories.

    Args:
        base_folder_path (str): Path to the base folder containing 'Positive' and 'Negative' directories.

    Returns:
        list: A list of JSON objects loaded from files in the specified directories.
    """
    data = []
    for category in ['Positive', 'Negative']:
        folder_path = os.path.join(base_folder_path, category)
        if not os.path.exists(folder_path):
            continue
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.json'):
                with open(os.path.join(folder_path, file_name), 'r') as file:
                    case_data = json.load(file)
                    data.append(case_data)
    return data


def load_histology_data(histology_path):
    """
    Load histology data from a JSON file.

    Args:
        histology_path (str): Path to the histology JSON file.

    Returns:
        dict: Histology data dictionary.
    """
    with open(histology_path, 'r') as file:
        histology_data = json.load(file)
    return histology_data


def compute_iou_matrix(true_bboxes, pred_bboxes):
    """
    Compute Intersection over Union (IoU) matrix between sets of true and predicted bounding boxes.

    Args:
        true_bboxes (list): List of ground truth bounding boxes.
        pred_bboxes (list): List of predicted bounding boxes.

    Returns:
        np.ndarray: IoU matrix of shape (num_true_bboxes, num_pred_bboxes).
    """
    if len(true_bboxes) == 0 or len(pred_bboxes) == 0:
        return np.zeros((len(true_bboxes), len(pred_bboxes)))

    true_bboxes = np.array(true_bboxes)
    pred_bboxes = np.array(pred_bboxes)

    # Expand dimensions for broadcasting
    true_bboxes_exp = true_bboxes[:, np.newaxis, :]
    pred_bboxes_exp = pred_bboxes[np.newaxis, :, :]

    # Intersection coordinates
    xi1 = np.maximum(true_bboxes_exp[:, :, 0], pred_bboxes_exp[:, :, 0])
    yi1 = np.maximum(true_bboxes_exp[:, :, 1], pred_bboxes_exp[:, :, 1])
    xi2 = np.minimum(true_bboxes_exp[:, :, 2], pred_bboxes_exp[:, :, 2])
    yi2 = np.minimum(true_bboxes_exp[:, :, 3], pred_bboxes_exp[:, :, 3])

    # Intersection area
    inter_width = np.maximum(0, xi2 - xi1)
    inter_height = np.maximum(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Areas
    true_areas = (true_bboxes_exp[:, :, 2] - true_bboxes_exp[:, :, 0]) * (true_bboxes_exp[:, :, 3] - true_bboxes_exp[:, :, 1])
    pred_areas = (pred_bboxes_exp[:, :, 2] - pred_bboxes_exp[:, :, 0]) * (pred_bboxes_exp[:, :, 3] - pred_bboxes_exp[:, :, 1])

    # Union area
    union_area = true_areas + pred_areas - inter_area + 1e-6  # Prevent division by zero

    # IoU
    iou_matrix = inter_area / union_area
    return iou_matrix


def evaluate_frames_bbox(video_data, iou_threshold=0.5):
    """
    Evaluate bounding box predictions against annotations at a given IoU threshold.
    Computes per-video metrics and aggregates them for overall metrics and confidence intervals.

    Args:
        video_data (dict): Dictionary containing per-video frame data.
        iou_threshold (float): IoU threshold for a prediction to be considered a True Positive.

    Returns:
        dict: Dictionary containing overall and per-video metrics for bounding box evaluation.
    """
    per_video_metrics = {}
    iou_per_frame = []
    tp_total = fp_total = fn_total = matched_iou_total = 0
    total_ious = []

    # Iterate over each video
    for case_id, case_data in video_data.items():
        video_tp = video_fp = video_fn = video_matched_iou = 0
        category = case_data['category']
        per_frame_iou_list = []
        per_frame_counts = []

        # Process each frame
        for frame_data in case_data['frames']:
            true_bboxes = frame_data['annotations_bbox']
            pred_bboxes = frame_data['pred_bbox']
            frame_tp = frame_fp = frame_fn = 0

            # Handle different cases of presence/absence of bboxes
            if not true_bboxes and not pred_bboxes:
                # No annotations and no detections: treated as TN scenario, but not counted for bbox eval
                per_frame_iou_list.append(0)
                continue
            elif not true_bboxes:
                # No annotations, but predictions present: all are FP
                video_fp += len(pred_bboxes)
                frame_fp += len(pred_bboxes)
                per_frame_iou_list.append(0)
                total_ious.extend([0] * len(pred_bboxes))
                per_frame_counts.append({'tp': 0, 'fp': frame_fp, 'fn': 0, 'tn': 0})
                continue
            elif not pred_bboxes:
                # Annotations present, but no predictions: all are FN
                video_fn += len(true_bboxes)
                frame_fn += len(true_bboxes)
                per_frame_iou_list.append(0)
                total_ious.extend([0] * len(true_bboxes))
                per_frame_counts.append({'tp': 0, 'fp': 0, 'fn': frame_fn, 'tn': 0})
                continue

            iou_matrix = compute_iou_matrix(true_bboxes, pred_bboxes)
            cost_matrix = 1 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            matched_pred, matched_true = set(), set()
            matched_ious = []

            # Match pairs of true and predicted bboxes
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= iou_threshold:
                    # True Positive
                    video_tp += 1
                    frame_tp += 1
                    video_matched_iou += iou_matrix[r, c]
                    matched_pred.add(c)
                    matched_true.add(r)
                    matched_ious.append(iou_matrix[r, c])
                    total_ious.append(iou_matrix[r, c])
                else:
                    # Mismatch: counted as FP and FN
                    video_fp += 1
                    frame_fp += 1
                    video_fn += 1
                    frame_fn += 1
                    matched_ious.append(iou_matrix[r, c])
                    total_ious.append(iou_matrix[r, c])

            # Unmatched predictions are FP
            unmatched_pred = set(range(len(pred_bboxes))) - matched_pred
            video_fp += len(unmatched_pred)
            frame_fp += len(unmatched_pred)

            # Unmatched annotations are FN
            unmatched_true = set(range(len(true_bboxes))) - matched_true
            video_fn += len(unmatched_true)
            frame_fn += len(unmatched_true)

            mean_iou = np.mean(matched_ious) if matched_ious else 0
            per_frame_iou_list.append(mean_iou)
            per_frame_counts.append({'tp': frame_tp, 'fp': frame_fp, 'fn': frame_fn, 'tn': 0})

        # Update global counters
        tp_total += video_tp
        fp_total += video_fp
        fn_total += video_fn
        matched_iou_total += video_matched_iou

        # Compute per-video metrics
        precision = video_tp / (video_tp + video_fp) if (video_tp + video_fp) > 0 else 0.0
        recall = video_tp / (video_tp + video_fn) if (video_tp + video_fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = video_tp / (video_tp + video_fp + video_fn) if (video_tp + video_fp + video_fn) > 0 else 0.0

        per_video_metrics[case_id] = {
            'category': category,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'per_frame_iou': per_frame_iou_list,
            'per_frame_counts': per_frame_counts
        }

    # Compute overall metrics
    overall_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    overall_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    overall_f1_score = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    overall_accuracy = tp_total / (tp_total + fp_total + fn_total) if (tp_total + fp_total + fn_total) > 0 else 0.0

    total_ious = [iou for iou in total_ious if isinstance(iou, (int, float, np.float64))]
    mean_iou = np.mean(total_ious) if total_ious else 0.0

    overall_metrics = {
        'tp_total': tp_total,
        'fp_total': fp_total,
        'fn_total': fn_total,
        'total_cases': tp_total + fp_total + fn_total,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1_score': overall_f1_score,
        'accuracy': overall_accuracy,
        'mean_iou': mean_iou,
    }

    confidence_level = 0.95
    overall_metrics['precision_95%_CI'] = proportion_confint(tp_total, tp_total + fp_total, alpha=1 - confidence_level, method='beta') if (tp_total + fp_total) > 0 else (0.0, 0.0)
    overall_metrics['recall_95%_CI'] = proportion_confint(tp_total, tp_total + fn_total, alpha=1 - confidence_level, method='beta') if (tp_total + fn_total) > 0 else (0.0, 0.0)
    overall_metrics['accuracy_95%_CI'] = proportion_confint(tp_total, tp_total + fp_total + fn_total, alpha=1 - confidence_level, method='beta') if (tp_total + fp_total + fn_total) > 0 else (0.0, 0.0)

    overall_metrics['mean_iou_95%_CI'] = bootstrap_confidence_interval(total_ious, n_bootstraps=1000, alpha=1 - confidence_level)
    overall_metrics['iou_per_frame'] = total_ious

    return {
        'overall_metrics': overall_metrics,
        'per_video_metrics': per_video_metrics
    }


def evaluate_frames_detection(video_data):
    """
    Evaluate frame-level detection metrics (without bounding box IoU checks).
    Computes per-video and overall metrics (precision, recall, F1, accuracy, specificity).

    Args:
        video_data (dict): Dictionary containing per-video frame data.

    Returns:
        dict: Dictionary containing overall and per-video detection metrics.
    """
    per_video_metrics = {}

    # Compute per-video metrics
    for case_id, case_data in video_data.items():
        category = case_data['category']
        video_tp = video_fp = video_fn = video_tn = 0
        per_frame_counts = []

        for frame_data in case_data['frames']:
            has_annotation = len(frame_data['annotations_bbox']) > 0 and any(bbox != [0, 0, 0, 0] for bbox in frame_data['annotations_bbox'])
            has_detection = frame_data['detections']

            frame_counts = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

            # Determine TP, FP, FN, TN
            if has_annotation and has_detection:
                video_tp += 1
                frame_counts['tp'] = 1
            elif has_annotation and not has_detection:
                video_fn += 1
                frame_counts['fn'] = 1
            elif not has_annotation and has_detection:
                video_fp += 1
                frame_counts['fp'] = 1
            else:
                video_tn += 1
                frame_counts['tn'] = 1

            per_frame_counts.append(frame_counts)

        precision = video_tp / (video_tp + video_fp) if (video_tp + video_fp) > 0 else 0.0
        recall = video_tp / (video_tp + video_fn) if (video_tp + video_fn) > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (video_tp + video_tn) / (video_tp + video_fp + video_fn + video_tn) if (video_tp + video_fp + video_fn + video_tn) > 0 else 0.0
        specificity = video_tn / (video_tn + video_fp) if (video_tn + video_fp) > 0 else 0.0

        per_video_metrics[case_id] = {
            'category': category,
            'tp': video_tp,
            'fp': video_fp,
            'fn': video_fn,
            'tn': video_tn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'specificity': specificity,
            'per_frame_counts': per_frame_counts
        }

    # Compute overall metrics
    overall_tp = sum(v['tp'] for v in per_video_metrics.values())
    overall_fp = sum(v['fp'] for v in per_video_metrics.values())
    overall_fn = sum(v['fn'] for v in per_video_metrics.values())
    overall_tn = sum(v['tn'] for v in per_video_metrics.values())
    total_cases = overall_tp + overall_fp + overall_fn + overall_tn

    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    overall_accuracy = (overall_tp + overall_tn) / total_cases if total_cases > 0 else 0.0
    overall_specificity = overall_tn / (overall_tn + overall_fp) if (overall_tn + overall_fp) > 0 else 0.0
    overall_f1_score = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    confidence_level = 0.95
    precision_ci = proportion_confint(overall_tp, overall_tp + overall_fp, alpha=1 - confidence_level, method='beta') if (overall_tp + overall_fp) > 0 else (0.0, 0.0)
    recall_ci = proportion_confint(overall_tp, overall_tp + overall_fn, alpha=1 - confidence_level, method='beta') if (overall_tp + overall_fn) > 0 else (0.0, 0.0)
    accuracy_ci = proportion_confint(overall_tp + overall_tn, total_cases, alpha=1 - confidence_level, method='beta') if total_cases > 0 else (0.0, 0.0)
    specificity_ci = proportion_confint(overall_tn, overall_tn + overall_fp, alpha=1 - confidence_level, method='beta') if (overall_tn + overall_fp) > 0 else (0.0, 0.0)

    overall_metrics = {
        'tp_total': overall_tp,
        'fp_total': overall_fp,
        'fn_total': overall_fn,
        'tn_total': overall_tn,
        'total_cases': total_cases,
        'precision': overall_precision,
        'precision_95%_CI': precision_ci,
        'recall': overall_recall,
        'recall_95%_CI': recall_ci,
        'f1_score': overall_f1_score,
        'f1_score_95%_CI': None,
        'accuracy': overall_accuracy,
        'accuracy_95%_CI': accuracy_ci,
        'specificity': overall_specificity,
        'specificity_95%_CI': specificity_ci
    }

    return {
        'overall_metrics': overall_metrics,
        'per_video_metrics': per_video_metrics
    }


def bootstrap_confidence_interval(data, func=np.mean, n_bootstraps=1000, alpha=0.05):
    """
    Calculate bootstrapped confidence intervals using BCa adjustment.

    Args:
        data (list or array): Data to bootstrap.
        func (callable): Function to apply (e.g., np.mean).
        n_bootstraps (int): Number of bootstrap samples.
        alpha (float): Significance level for confidence interval.

    Returns:
        tuple: (lower_bound, upper_bound) of the confidence interval.
    """
    results = bootstrap((data,), func, confidence_level=1 - alpha, method='BCa', n_resamples=n_bootstraps)
    return results.confidence_interval.low, results.confidence_interval.high


def bootstrap_accuracy(y_true, y_pred, n_bootstraps=1000, confidence_level=0.95):
    """
    Compute bootstrap confidence intervals for accuracy.

    Args:
        y_true (list): Ground truth labels.
        y_pred (list): Predicted labels.
        n_bootstraps (int): Number of bootstrap samples.
        confidence_level (float): Confidence level for intervals.

    Returns:
        tuple: (mean_accuracy, lower_bound, upper_bound, accuracies) where accuracies is the list of all bootstrap accuracies.
    """
    accuracies = []
    n = len(y_true)

    for _ in range(n_bootstraps):
        indices = np.random.choice(range(n), n, replace=True)
        sample_y_true = [y_true[i] for i in indices]
        sample_y_pred = [y_pred[i] for i in indices]

        cm_sample = confusion_matrix(sample_y_true, sample_y_pred)
        accuracy_sample = np.trace(cm_sample) / np.sum(cm_sample)
        accuracies.append(accuracy_sample)

    mean_accuracy = np.mean(accuracies)
    lower_bound = np.percentile(accuracies, (1 - confidence_level) / 2 * 100)
    upper_bound = np.percentile(accuracies, (1 + confidence_level) / 2 * 100)

    return mean_accuracy, lower_bound, upper_bound, accuracies


def evaluate_histology(video_data, model_key, output_folder, n_bootstraps=1000, confidence_level=0.95):
    """
    Evaluate histology predictions with confusion matrix and bootstrap accuracies.

    Args:
        video_data (dict): Per-video frame data including histology info.
        model_key (str): Model identifier ('GPT', 'Gemini', or 'CAD').
        output_folder (str): Directory to save results.
        n_bootstraps (int): Number of bootstrap samples for accuracy CI.
        confidence_level (float): Confidence level for intervals.

    Returns:
        dict: Histology evaluation metrics including confusion matrix and accuracies.
    """
    histologies_gt = []
    histologies_pred = []

    # Extract predictions and ground truths
    for case_id, case_data in video_data.items():
        for frame_data in case_data['frames']:
            histology_pred = frame_data['histology_pred']
            if ' ' in histology_pred:
                histology_pred = histology_pred.split(' ')[0]
            histology_gt = frame_data['histology_simpl_gt']
            histologies_gt.append(histology_gt)
            histologies_pred.append(histology_pred)
            frame_data['correct_histology'] = (histology_pred == histology_gt)

    labels = ["Adenoma", "Hyperplastic", "Serrated", "Invasive", "None"]
    cm = confusion_matrix(histologies_gt, histologies_pred, labels=labels)
    accuracy = np.trace(cm) / np.sum(cm)

    with np.errstate(divide='ignore', invalid='ignore'):
        class_accuracies = np.diag(cm) / np.sum(cm, axis=1)
        class_accuracies = np.nan_to_num(class_accuracies)

    mean_accuracy, lower_bound, upper_bound, bootstrap_accuracies = bootstrap_accuracy(
        histologies_gt, histologies_pred, n_bootstraps, confidence_level
    )

    # Save confusion matrix plot
    cm_output_path = os.path.join(output_folder, 'confusion_matrices', f"{model_key}_histology_confusion_matrix.png")
    os.makedirs(os.path.dirname(cm_output_path), exist_ok=True)
    save_confusion_matrix(cm, labels, f"{model_key} Histology\nConfusion Matrix", cm_output_path)

    # Organize histology results
    histology_results = {
        'confusion_matrix': cm.tolist(),
        'accuracy': accuracy,
        'class_accuracies': class_accuracies.tolist(),
        'mean_accuracy': mean_accuracy,
        'confidence_interval': [lower_bound, upper_bound],
        'labels': labels,
        'bootstrap_accuracies': bootstrap_accuracies
    }

    # Save as JSON
    results_output_path = os.path.join(output_folder, f"{model_key}_histology_results.json")
    with open(results_output_path, 'w') as f:
        json.dump(histology_results, f)

    return histology_results


def save_confusion_matrix(conf_matrix, labels, title, output_path):
    """
    Save the given confusion matrix as an image.

    Args:
        conf_matrix (np.ndarray): Confusion matrix.
        labels (list): Class labels.
        title (str): Title for the plot.
        output_path (str): Path to save the plot.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(conf_matrix, cmap='Blues')
    clb = plt.colorbar()

    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha='right', fontsize=16)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels, fontsize=16)

    thresh = conf_matrix.max() / 2.
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[0])):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    plt.title(title, fontsize=20, fontweight='bold', y=1.01)
    plt.xlabel('Predicted', fontsize=20)
    plt.ylabel('True', fontsize=20)

    clb.ax.tick_params(labelsize=16)
    clb.mappable.set_clim(vmin=0, vmax=900)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def process_json_data(data, histology_data, model_key, output_folder, iou_threshold=0.5):
    """
    Process the JSON data for a given model to compute various metrics.

    Args:
        data (list): List of case data loaded from JSON files.
        histology_data (dict): Histology ground truth data.
        model_key (str): Model identifier.
        output_folder (str): Folder to save output files.
        iou_threshold (float): IoU threshold for bounding box evaluation.

    Returns:
        dict: Dictionary containing metrics without bbox, with bbox, histology metrics, and per lesion metrics.
    """
    model_has_bboxes = model_key in ['Gemini', 'CAD']
    video_data = {}

    # Organize data by video
    for case_data in tqdm(data, desc=f"Processing {model_key} data"):
        case_id = case_data['details']['Case']
        category = case_data['details']['Category']
        case_ref = case_id + '_' + category

        if case_ref not in video_data:
            video_data[case_ref] = {
                'category': category,
                'total_frames': len(case_data['frames']),
                'time': case_data['details'].get(f"{model_key} time per frame (s)", 0) if model_key == 'GPT' else
                         case_data['details'].get("Gemini time per second of video (s)", 0),
                'frames': []
            }

        for frame in case_data['frames']:
            frame_number = frame['Frame']
            if category == 'Positive':
                frame_data = {
                    'frame_number': frame_number,
                    'annotations_bbox': frame['Annotation'].get('Bounding boxes', []),
                    'detections': None,
                    'pred_bbox': None,
                    'histology_pred': None,
                    'histology_simpl_gt': histology_data.get(case_id, {}).get('Simplified diagnosis', 'Unknown'),
                    'histology_real_gt': histology_data.get(case_id, {}).get('Pathological diagnosis', 'Unknown'),
                    'histology_adenoma_gt': None
                }
            else:
                frame_data = {
                    'frame_number': frame_number,
                    'annotations_bbox': [],
                    'detections': None,
                    'pred_bbox': None,
                    'histology_pred': None,
                    'histology_simpl_gt': 'None',
                    'histology_real_gt': 'None',
                    'histology_adenoma_gt': 'None'
                }

            # Extract model-specific data
            if model_key == 'CAD':
                bbox = frame[model_key].get('Bounding boxes', [])
                frame_data['detections'] = bool(bbox and bbox != [[0, 0, 0, 0]])
                frame_data['pred_bbox'] = bbox

            if model_key in ['GPT', 'Gemini']:
                frame_data['detections'] = frame[model_key]['Lesion detected'] == 'True'
                frame_data['histology_pred'] = frame[model_key]['Histology']
                if 'Bounding boxes' in frame[model_key]:
                    frame_data['pred_bbox'] = frame[model_key]['Bounding boxes']

            # Determine adenoma or non-adenoma
            if frame_data['histology_real_gt'] in ['High-grade adenoma', 'Low-grade adenoma']:
                frame_data['histology_adenoma_gt'] = 'adenoma'
            else:
                frame_data['histology_adenoma_gt'] = 'non-adenoma'

            video_data[case_ref]['frames'].append(frame_data)

    # Evaluate frame-level detection (no bbox)
    metrics_no_bbox = evaluate_frames_detection(video_data)

    # Per lesion metrics
    per_lesion_metrics = compute_per_lesion_metrics(video_data)

    # Evaluate bounding boxes if applicable
    metrics_bbox = evaluate_frames_bbox(video_data, iou_threshold=iou_threshold) if model_has_bboxes else 'None'

    # Histology metrics if applicable
    histology_metrics = evaluate_histology(video_data, model_key, output_folder) if model_key in ['GPT', 'Gemini'] else 'None'

    return {
        'metrics_no_bbox': metrics_no_bbox,
        'metrics_bbox': metrics_bbox,
        'histology_metrics': histology_metrics,
        'per_lesion_metrics': per_lesion_metrics,
    }


def compute_proportion_confidence_interval(successes, n, confidence=0.95):
    """
    Compute confidence interval for a proportion using a Beta method.

    Args:
        successes (int): Number of successes.
        n (int): Total number of trials.
        confidence (float): Confidence level.

    Returns:
        tuple: (lower_bound, upper_bound) of the confidence interval.
    """
    if n == 0:
        return (0.0, 0.0)
    return proportion_confint(successes, n, alpha=1 - confidence, method='beta')


def compute_per_lesion_metrics(video_data):
    """
    Compute per lesion metrics, including confidence intervals and histology-based subgroup metrics.

    Args:
        video_data (dict): Per-video frame data.

    Returns:
        dict: Dictionary containing per lesion metrics and subgroup analyses.
    """
    tp = fp = fn = tn = 0
    tp_adenoma = fn_adenoma = 0
    tp_non_adenoma = fn_non_adenoma = 0

    per_video_metrics = []
    per_video_performance = []
    per_video_counts = []

    for case_id, case_data in video_data.items():
        category = case_data['category']
        total_video_frames = case_data['total_frames']
        histology_gt = video_data[case_id]['frames'][0]['histology_adenoma_gt']

        # Determine lesion detection by majority frames
        lesion_detected = sum(frame['detections'] for frame in case_data['frames']) > (total_video_frames / 2)

        video_counts = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}

        if category == 'Positive':
            # Positive video: TP if lesion detected, otherwise FN
            per_video_metrics.append({
                'case_id': case_id,
                'category': category,
                'histology_gt': histology_gt,
                'tp': 1 if lesion_detected else 0,
                'fn': 0 if lesion_detected else 1
            })
            if lesion_detected:
                tp += 1
                video_counts['tp'] = 1
                if histology_gt == 'adenoma':
                    tp_adenoma += 1
                else:
                    tp_non_adenoma += 1
                per_video_performance.append([1, 0])
            else:
                fn += 1
                video_counts['fn'] = 1
                if histology_gt == 'adenoma':
                    fn_adenoma += 1
                else:
                    fn_non_adenoma += 1
                per_video_performance.append([0, 1])

        elif category == 'Negative':
            # Negative video: TN if no lesion detected, otherwise FP
            per_video_metrics.append({
                'case_id': case_id,
                'category': category,
                'histology_gt': histology_gt,
                'tn': 1 if not lesion_detected else 0,
                'fp': 0 if not lesion_detected else 1
            })
            if lesion_detected:
                fp += 1
                video_counts['fp'] = 1
                per_video_performance.append([0, 1])
            else:
                tn += 1
                video_counts['tn'] = 1
                per_video_performance.append([1, 0])

        per_video_counts.append(video_counts)

    # Compute aggregate metrics
    total_cases = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / total_cases if total_cases > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Confidence intervals
    precision_ci = compute_proportion_confidence_interval(tp, tp + fp) if (tp + fp) > 0 else (0.0, 0.0)
    recall_ci = compute_proportion_confidence_interval(tp, tp + fn) if (tp + fn) > 0 else (0.0, 0.0)
    specificity_ci = compute_proportion_confidence_interval(tn, tn + fp) if (tn + fp) > 0 else (0.0, 0.0)
    accuracy_ci = compute_proportion_confidence_interval(tp + tn, total_cases) if total_cases > 0 else (0.0, 0.0)

    # Histology subgroup metrics
    total_adenoma = tp_adenoma + fn_adenoma
    total_non_adenoma = tp_non_adenoma + fn_non_adenoma
    recall_adenoma = tp_adenoma / total_adenoma if total_adenoma > 0 else 0
    recall_non_adenoma = tp_non_adenoma / total_non_adenoma if total_non_adenoma > 0 else 0
    recall_adenoma_ci = compute_proportion_confidence_interval(tp_adenoma, total_adenoma)
    recall_non_adenoma_ci = compute_proportion_confidence_interval(tp_non_adenoma, total_non_adenoma)

    per_histology_metrics = {
        'adenoma': {
            'total_cases': total_adenoma,
            'detected_cases': tp_adenoma,
            'recall': recall_adenoma,
            'recall_95%_CI': recall_adenoma_ci
        },
        'non_adenoma': {
            'total_cases': total_non_adenoma,
            'detected_cases': tp_non_adenoma,
            'recall': recall_non_adenoma,
            'recall_95%_CI': recall_non_adenoma_ci
        }
    }

    # McNemar test for histology
    a = tp_adenoma
    b = fn_adenoma
    c = tp_non_adenoma
    d = fn_non_adenoma
    table = [[a, b], [c, d]]
    mcnemar_res = mcnemar(table, exact=False)
    mcnemar_res = {
        'statistic': mcnemar_res.statistic,
        'pvalue': mcnemar_res.pvalue
    }

    per_lesion_metrics = {
        'total_tp': tp,
        'total_fp': fp,
        'total_fn': fn,
        'total_tn': tn,
        'total_cases': total_cases,
        'precision': precision,
        'precision_95%_CI': precision_ci,
        'recall': recall,
        'recall_95%_CI': recall_ci,
        'specificity': specificity,
        'specificity_95%_CI': specificity_ci,
        'accuracy': accuracy,
        'accuracy_95%_CI': accuracy_ci,
        'f1_score': f1_score,
        'f1_score_95%_CI': None,
        'per_histology_metrics': per_histology_metrics,
        'per_histology_mcNemar_test': mcnemar_res,
        'per_video_metrics': per_video_metrics,
        'per_video_performance': per_video_performance,
        'per_video_counts': per_video_counts
    }

    return per_lesion_metrics


def save_metrics(metrics, output_folder, filename):
    """
    Save computed metrics as a JSON file.

    Args:
        metrics (dict): Metrics dictionary to save.
        output_folder (str): Directory where the file will be saved.
        filename (str): Name of the output JSON file (without extension).
    """
    output_path = os.path.join(output_folder, f"{filename}.json")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {output_path}")


def compute_overall_metrics(counts_list):
    """
    Compute overall metrics (precision, recall, specificity, accuracy) from a list of counts dictionaries.

    Args:
        counts_list (list): List of dicts with keys 'tp', 'fp', 'fn', 'tn'.

    Returns:
        dict: Dictionary with computed overall metrics.
    """
    tp = sum(count['tp'] for count in counts_list)
    fp = sum(count['fp'] for count in counts_list)
    fn = sum(count['fn'] for count in counts_list)
    tn = sum(count['tn'] for count in counts_list)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'accuracy': accuracy,
    }


def compare_per_frame_metrics(metrics_model1, metrics_model2):
    """
    Compare per-frame metrics between two models using contingency tests.

    Args:
        metrics_model1 (dict): Metrics dictionary for model 1.
        metrics_model2 (dict): Metrics dictionary for model 2.

    Returns:
        dict: p-values for comparisons of precision, recall, specificity, accuracy.
    """
    overall1 = metrics_model1['overall_metrics']
    overall2 = metrics_model2['overall_metrics']

    tp1, fp1, fn1, tn1 = overall1.get('tp_total', 0), overall1.get('fp_total', 0), overall1.get('fn_total', 0), overall1.get('tn_total', 0)
    tp2, fp2, fn2, tn2 = overall2.get('tp_total', 0), overall2.get('fp_total', 0), overall2.get('fn_total', 0), overall2.get('tn_total', 0)

    # Contingency tables for each metric
    table_precision = np.array([[tp1, fp1], [tp2, fp2]])
    table_recall = np.array([[tp1, fn1], [tp2, fn2]])
    table_specificity = np.array([[tn1, fp1], [tn2, fp2]])
    acc1, err1 = tp1 + tn1, fp1 + fn1
    acc2, err2 = tp2 + tn2, fp2 + fn2
    table_accuracy = np.array([[acc1, err1], [acc2, err2]])

    def perform_test(table):
        if np.any(table < 5):
            # Fisher exact test for small counts
            _, p_value = fisher_exact(table)
        else:
            chi2, p_value, dof, expected = chi2_contingency(table)
        return p_value

    p_values = {
        'precision': perform_test(table_precision),
        'recall': perform_test(table_recall),
        'specificity': perform_test(table_specificity),
        'accuracy': perform_test(table_accuracy)
    }

    return p_values


def compare_per_lesion_metrics(metrics_model1, metrics_model2):
    """
    Compare per-lesion metrics between two models using contingency tests.

    Args:
        metrics_model1 (dict): Per lesion metrics for model 1.
        metrics_model2 (dict): Per lesion metrics for model 2.

    Returns:
        dict: p-values for comparisons of precision, recall, specificity, accuracy.
    """
    counts_model1 = metrics_model1['per_video_counts']
    counts_model2 = metrics_model2['per_video_counts']

    tp1 = sum(count['tp'] for count in counts_model1)
    fp1 = sum(count['fp'] for count in counts_model1)
    fn1 = sum(count['fn'] for count in counts_model1)
    tn1 = sum(count['tn'] for count in counts_model1)

    tp2 = sum(count['tp'] for count in counts_model2)
    fp2 = sum(count['fp'] for count in counts_model2)
    fn2 = sum(count['fn'] for count in counts_model2)
    tn2 = sum(count['tn'] for count in counts_model2)

    # Contingency tables
    table_precision = np.array([[tp1, fp1], [tp2, fp2]])
    table_recall = np.array([[tp1, fn1], [tp2, fn2]])
    table_specificity = np.array([[tn1, fp1], [tn2, fp2]])
    acc1, err1 = tp1 + tn1, fp1 + fn1
    acc2, err2 = tp2 + tn2, fp2 + fn2
    table_accuracy = np.array([[acc1, err1], [acc2, err2]])

    def perform_test(table):
        if np.any(table < 5):
            _, p_value = fisher_exact(table)
        else:
            chi2, p_value, dof, expected = chi2_contingency(table)
        return p_value

    p_values = {
        'precision': perform_test(table_precision),
        'recall': perform_test(table_recall),
        'specificity': perform_test(table_specificity),
        'accuracy': perform_test(table_accuracy)
    }

    return p_values


def main():
    # Example default paths - update as needed
    base_folder_path = os.path.join(os.getcwd(), 'bounding_boxes')
    histology_path = os.path.join(os.getcwd(), "video_output_fullDs", "hystology.json")
    output_folder = os.path.join(base_folder_path, 'metrics_output')
    os.makedirs(output_folder, exist_ok=True)

    # Load data
    data = load_json_files(base_folder_path)
    histology_data = load_histology_data(histology_path)

    # Models to process
    model_keys = ['GPT', 'Gemini', 'CAD']

    metrics_model_bbox = {}
    metrics_model_no_bbox = {}
    histology_metrics_model = {}
    per_lesion_metrics_model = {}

    # Process each model
    for model_key in model_keys:
        metrics = process_json_data(data, histology_data, model_key, output_folder, iou_threshold=0.5)
        metrics_model_no_bbox[model_key] = metrics['metrics_no_bbox']
        metrics_model_bbox[model_key] = metrics['metrics_bbox']
        histology_metrics_model[model_key] = metrics['histology_metrics']
        per_lesion_metrics_model[model_key] = metrics['per_lesion_metrics']

    # Save results
    save_metrics(metrics_model_no_bbox, output_folder, 'metrics_no_bbox')
    save_metrics(metrics_model_bbox, output_folder, 'metrics_bbox')
    save_metrics(per_lesion_metrics_model, output_folder, 'per_lesion_metrics')

    print("Metrics evaluation completed.")

    # Statistical comparisons
    comparisons = [('CAD', 'GPT'), ('CAD', 'Gemini'), ('GPT', 'Gemini')]

    # Compare per-frame metrics without bbox
    p_values_per_frame = {}
    for model1, model2 in comparisons:
        p_values = compare_per_frame_metrics(metrics_model_no_bbox[model1], metrics_model_no_bbox[model2])
        comparison_key = f"{model1}_vs_{model2}"
        p_values_per_frame[comparison_key] = p_values

    output_file_path = os.path.join(output_folder, 'statistical_results_frame_no_bbox.json')
    with open(output_file_path, 'w') as f:
        json.dump(p_values_per_frame, f, indent=4)
    print(f"Per-frame metrics p-values saved to {output_file_path}")

    # Mean IoU p-values
    mean_iou_p_values = {}
    for model1, model2 in [('CAD', 'Gemini')]:
        if metrics_model_bbox[model1] != 'None' and metrics_model_bbox[model2] != 'None':
            ious_model1 = metrics_model_bbox[model1]['overall_metrics']['iou_per_frame']
            ious_model2 = metrics_model_bbox[model2]['overall_metrics']['iou_per_frame']
            stat, p_value = mannwhitneyu(ious_model1, ious_model2, alternative='two-sided')
            comparison_key = f"{model1}_vs_{model2}"
            mean_iou_p_values[comparison_key] = {'mean_iou_p_value': p_value}

    output_file_path = os.path.join(output_folder, 'statistical_results_mean_iou.json')
    with open(output_file_path, 'w') as f:
        json.dump(mean_iou_p_values, f, indent=4)
    print(f"Mean IoU p-values saved to {output_file_path}")

    # Per-frame bbox metrics comparison
    p_values_per_frame_bbox = {}
    for model1, model2 in [('CAD', 'Gemini')]:
        if metrics_model_bbox[model1] != 'None' and metrics_model_bbox[model2] != 'None':
            p_values = compare_per_frame_metrics(metrics_model_bbox[model1], metrics_model_bbox[model2])
            comparison_key = f"{model1}_vs_{model2}"
            p_values_per_frame_bbox[comparison_key] = p_values

    output_file_path = os.path.join(output_folder, 'statistical_results_frame_bbox.json')
    with open(output_file_path, 'w') as f:
        json.dump(p_values_per_frame_bbox, f, indent=4)
    print(f"Per-frame bbox metrics p-values saved to {output_file_path}")

    # Per-lesion metrics comparison
    p_values_per_lesion = {}
    for model1, model2 in comparisons:
        p_values = compare_per_lesion_metrics(per_lesion_metrics_model[model1], per_lesion_metrics_model[model2])
        comparison_key = f"{model1}_vs_{model2}"
        p_values_per_lesion[comparison_key] = p_values

    output_file_path = os.path.join(output_folder, 'statistical_results_per_lesion.json')
    with open(output_file_path, 'w') as f:
        json.dump(p_values_per_lesion, f, indent=4)
    print(f"Per-lesion metrics p-values saved to {output_file_path}")

    # Histology accuracy comparison between models (Mann-Whitney U test)
    comparisons = [('GPT', 'Gemini')]
    p_values_histology = {}
    for model1, model2 in comparisons:
        bootstrap_accuracies_model1 = histology_metrics_model[model1]['bootstrap_accuracies']
        bootstrap_accuracies_model2 = histology_metrics_model[model2]['bootstrap_accuracies']
        stat, p_value = mannwhitneyu(bootstrap_accuracies_model1, bootstrap_accuracies_model2, alternative='two-sided')
        comparison_key = f"{model1}_vs_{model2}"
        p_values_histology[comparison_key] = {'histology_accuracy_p_value': p_value}

    output_file_path = os.path.join(output_folder, 'bootstrap_p_values_histology_accuracy.json')
    with open(output_file_path, 'w') as f:
        json.dump(p_values_histology, f, indent=4)
    print(f"Comparison of histology accuracy between models saved to {output_file_path}")

    # Visualization of results
    output_images_folder = os.path.join(output_folder, 'results_plots')
    os.makedirs(output_images_folder, exist_ok=True)

    # Define paths for case without bounding boxes
    metrics_file_paths = os.path.join(output_folder, 'metrics_no_bbox.json')
    p_values_file = os.path.join(output_folder, 'statistical_results_frame_no_bbox.json')

    # load metrics and p-values
    with open(metrics_file_paths, 'r') as f:
        metrics = json.load(f)

    with open(p_values_file, 'r') as f:
        p_values = json.load(f)

    group_precision = [metrics['CAD']['overall_metrics']['precision'],
                       metrics['GPT']['overall_metrics']['precision'],
                       metrics['Gemini']['overall_metrics']['precision']]
    group_recall = [metrics['CAD']['overall_metrics']['recall'],
                    metrics['GPT']['overall_metrics']['recall'],
                    metrics['Gemini']['overall_metrics']['recall']]
    group_specificity = [metrics['CAD']['overall_metrics']['specificity'],
                         metrics['GPT']['overall_metrics']['specificity'],
                         metrics['Gemini']['overall_metrics']['specificity']]
    group_accuracy = [metrics['CAD']['overall_metrics']['accuracy'],
                      metrics['GPT']['overall_metrics']['accuracy'],
                      metrics['Gemini']['overall_metrics']['accuracy']]
    values = [group_precision, group_recall, group_specificity, group_accuracy]

    group_precision_ci = [metrics['CAD']['overall_metrics']['precision_95%_CI'],
                          metrics['GPT']['overall_metrics']['precision_95%_CI'],
                          metrics['Gemini']['overall_metrics']['precision_95%_CI']]
    group_recall_ci = [metrics['CAD']['overall_metrics']['recall_95%_CI'],
                       metrics['GPT']['overall_metrics']['recall_95%_CI'],
                       metrics['Gemini']['overall_metrics']['recall_95%_CI']]
    group_specificity_ci = [metrics['CAD']['overall_metrics']['specificity_95%_CI'],
                            metrics['GPT']['overall_metrics']['specificity_95%_CI'],
                            metrics['Gemini']['overall_metrics']['specificity_95%_CI']]
    group_accuracy_ci = [metrics['CAD']['overall_metrics']['accuracy_95%_CI'],
                         metrics['GPT']['overall_metrics']['accuracy_95%_CI'],
                         metrics['Gemini']['overall_metrics']['accuracy_95%_CI']]
    ci_values = [group_precision_ci, group_recall_ci, group_specificity_ci, group_accuracy_ci]

    group_precision_p = [p_values['CAD_vs_GPT']['precision'],
                         p_values['CAD_vs_Gemini']['precision'],
                         p_values['GPT_vs_Gemini']['precision']]
    group_recall_p = [p_values['CAD_vs_GPT']['recall'],
                      p_values['CAD_vs_Gemini']['recall'],
                      p_values['GPT_vs_Gemini']['recall']]
    group_specificity_p = [p_values['CAD_vs_GPT']['specificity'],
                           p_values['CAD_vs_Gemini']['specificity'],
                           p_values['GPT_vs_Gemini']['specificity']]
    group_accuracy_p = [p_values['CAD_vs_GPT']['accuracy'],
                        p_values['CAD_vs_Gemini']['accuracy'],
                        p_values['GPT_vs_Gemini']['accuracy']]
    p_values_list = [group_precision_p, group_recall_p, group_specificity_p, group_accuracy_p]

    # In p values if there is nan set it to 1
    for i in range(len(p_values_list)):
        for j in range(len(p_values_list[i])):
            if p_values_list[i][j] is None or np.isnan(p_values_list[i][j]):
                p_values_list[i][j] = 1

    bar_labels = ['CAD', 'GPT', 'Gemini']
    group_labels = ['Precision', 'Sensitivity', 'Specificity', 'Accuracy']

    create_grouped_bar_plot_with_min_max_and_p_values(values, ci_values, p_values_list, group_labels=group_labels,
                                                      bar_labels=bar_labels, title='Per Frame Comparison',
                                                      ylabel='Metrics',
                                                      save_path=os.path.join(output_images_folder,
                                                                             'per_frame_comparison.png'),
                                                      color_choice='green')

    # Define paths for case with bounding boxes
    metrics_file_paths = os.path.join(output_folder, 'metrics_bbox.json')
    p_values_file = os.path.join(output_folder, 'statistical_results_frame_bbox.json')
    p_value_meanIoU_file = os.path.join(output_folder, 'statistical_results_mean_iou.json')

    # load metrics and p-values
    with open(metrics_file_paths, 'r') as f:
        metrics = json.load(f)

    with open(p_values_file, 'r') as f:
        p_values = json.load(f)

    with open(p_value_meanIoU_file, 'r') as f:
        p_values_mean_iou = json.load(f)

    # group_precision = [metrics['CAD']['overall_metrics']['precision'], metrics['Gemini']['overall_metrics']['precision']]
    # group_recall = [metrics['CAD']['overall_metrics']['recall'], metrics['Gemini']['overall_metrics']['recall']]
    # group_accuracy = [metrics['CAD']['overall_metrics']['accuracy'], metrics['Gemini']['overall_metrics']['accuracy']]
    group_mean_iou = [metrics['CAD']['overall_metrics']['mean_iou'], metrics['Gemini']['overall_metrics']['mean_iou']]
    # values = [group_precision, group_recall, group_accuracy, group_mean_iou]
    values = [group_mean_iou]

    # group_precision_ci = [metrics['CAD']['overall_metrics']['precision_95%_CI'], metrics['Gemini']['overall_metrics']['precision_95%_CI']]
    # group_recall_ci = [metrics['CAD']['overall_metrics']['recall_95%_CI'], metrics['Gemini']['overall_metrics']['recall_95%_CI']]
    # group_accuracy_ci = [metrics['CAD']['overall_metrics']['accuracy_95%_CI'], metrics['Gemini']['overall_metrics']['accuracy_95%_CI']]
    group_mean_iou_ci = [metrics['CAD']['overall_metrics']['mean_iou_95%_CI'],
                         metrics['Gemini']['overall_metrics']['mean_iou_95%_CI']]
    # ci_values = [group_precision_ci, group_recall_ci, group_accuracy_ci, group_mean_iou_ci]
    ci_values = [group_mean_iou_ci]

    # group_precision_p = [p_values['CAD_vs_Gemini']['precision']]
    # group_recall_p = [p_values['CAD_vs_Gemini']['recall']]
    # group_accuracy_p = [p_values['CAD_vs_Gemini']['accuracy']]
    group_mean_iou_p = [p_values_mean_iou['CAD_vs_Gemini']['mean_iou_p_value']]
    # p_values_list = [group_precision_p, group_recall_p, group_accuracy_p, group_mean_iou_p]
    p_values_list = [group_mean_iou_p]

    # In p values if there is nan set it to 1
    for i in range(len(p_values_list)):
        for j in range(len(p_values_list[i])):
            if p_values_list[i][j] is None or np.isnan(p_values_list[i][j]):
                p_values_list[i][j] = 1

    bar_labels = ['CAD', 'Gemini']
    # group_labels = ['Precision', 'Sensitivity', 'Accuracy', 'Mean IoU']
    group_labels = ['Mean IoU']

    create_grouped_bar_plot_with_min_max_and_p_values(values, ci_values, p_values_list, group_labels=group_labels,
                                                      bar_labels=bar_labels, title='Segmentation Comparison',
                                                      ylabel='Metrics',
                                                      save_path=os.path.join(output_images_folder,
                                                                             'per_frame_comparison_bbox.png'),
                                                      color_choice='orange')

    # Define paths for case per lesion
    metrics_file_paths = os.path.join(output_folder, 'per_lesion_metrics.json')
    p_values_file = os.path.join(output_folder, 'statistical_results_per_lesion.json')

    # load metrics and p-values
    with open(metrics_file_paths, 'r') as f:
        metrics = json.load(f)

    with open(p_values_file, 'r') as f:
        p_values = json.load(f)

    group_precision = [metrics['CAD']['precision'], metrics['GPT']['precision'], metrics['Gemini']['precision']]
    group_recall = [metrics['CAD']['recall'], metrics['GPT']['recall'], metrics['Gemini']['recall']]
    group_specificity = [metrics['CAD']['specificity'], metrics['GPT']['specificity'], metrics['Gemini']['specificity']]
    group_accuracy = [metrics['CAD']['accuracy'], metrics['GPT']['accuracy'], metrics['Gemini']['accuracy']]
    values = [group_precision, group_recall, group_specificity, group_accuracy]

    group_precision_ci = [metrics['CAD']['precision_95%_CI'], metrics['GPT']['precision_95%_CI'],
                          metrics['Gemini']['precision_95%_CI']]
    group_recall_ci = [metrics['CAD']['recall_95%_CI'], metrics['GPT']['recall_95%_CI'],
                       metrics['Gemini']['recall_95%_CI']]
    group_specificity_ci = [metrics['CAD']['specificity_95%_CI'], metrics['GPT']['specificity_95%_CI'],
                            metrics['Gemini']['specificity_95%_CI']]
    group_accuracy_ci = [metrics['CAD']['accuracy_95%_CI'], metrics['GPT']['accuracy_95%_CI'],
                         metrics['Gemini']['accuracy_95%_CI']]
    ci_values = [group_precision_ci, group_recall_ci, group_specificity_ci, group_accuracy_ci]

    group_precision_p = [p_values['CAD_vs_GPT']['precision'], p_values['CAD_vs_Gemini']['precision'],
                         p_values['GPT_vs_Gemini']['precision']]
    group_recall_p = [p_values['CAD_vs_GPT']['recall'], p_values['CAD_vs_Gemini']['recall'],
                      p_values['GPT_vs_Gemini']['recall']]
    group_specificity_p = [p_values['CAD_vs_GPT']['specificity'], p_values['CAD_vs_Gemini']['specificity'],
                           p_values['GPT_vs_Gemini']['specificity']]
    group_accuracy_p = [p_values['CAD_vs_GPT']['accuracy'], p_values['CAD_vs_Gemini']['accuracy'],
                        p_values['GPT_vs_Gemini']['accuracy']]
    p_values_list = [group_precision_p, group_recall_p, group_specificity_p, group_accuracy_p]

    # In p values if there is nan set it to 1
    for i in range(len(p_values_list)):
        for j in range(len(p_values_list[i])):
            if p_values_list[i][j] is None or np.isnan(p_values_list[i][j]):
                p_values_list[i][j] = 1

    bar_labels = ['CAD', 'GPT', 'Gemini']
    group_labels = ['Precision', 'Sensitivity', 'Specificity', 'Accuracy']

    create_grouped_bar_plot_with_min_max_and_p_values(values, ci_values, p_values_list, group_labels=group_labels,
                                                      bar_labels=bar_labels, title='Per Lesion Comparison',
                                                      ylabel='Metrics',
                                                      save_path=os.path.join(output_images_folder,
                                                                             'per_lesion_comparison.png'),
                                                      color_choice='blue')

    print("Results visualization completed.")

    # Overal graph
    # load metrics
    per_lesion_path = os.path.join(output_folder, 'per_lesion_metrics.json')
    per_frame_path = os.path.join(output_folder, 'metrics_no_bbox.json')
    per_frame_bbox_path = os.path.join(output_folder, 'metrics_bbox.json')

    with open(per_lesion_path, 'r') as f:
        per_lesion_metrics = json.load(f)

    with open(per_frame_path, 'r') as f:
        per_frame_metrics = json.load(f)

    with open(per_frame_bbox_path, 'r') as f:
        per_frame_bbox_metrics = json.load(f)

    # load p-values
    p_values_path = os.path.join(output_folder, 'statistical_results_per_lesion.json')
    p_values_frame_path = os.path.join(output_folder, 'statistical_results_frame_no_bbox.json')
    p_values_frame_bbox_path = os.path.join(output_folder, 'statistical_results_frame_bbox.json')
    p_values_mean_iou_path = os.path.join(output_folder, 'statistical_results_mean_iou.json')

    with open(p_values_path, 'r') as f:
        p_values = json.load(f)

    with open(p_values_frame_path, 'r') as f:
        p_values_frame = json.load(f)

    with open(p_values_frame_bbox_path, 'r') as f:
        p_values_frame_bbox = json.load(f)

    with open(p_values_mean_iou_path, 'r') as f:
        p_values_mean_iou = json.load(f)

    group_lesion_sensitivity = [
        per_lesion_metrics['CAD']['recall'],
        per_lesion_metrics['GPT']['recall'],
        per_lesion_metrics['Gemini']['recall']
    ]

    group_lesion_specificity = [
        per_lesion_metrics['CAD']['specificity'],
        per_lesion_metrics['GPT']['specificity'],
        per_lesion_metrics['Gemini']['specificity']
    ]

    group_lesion_accuracy = [
        per_lesion_metrics['CAD']['accuracy'],
        per_lesion_metrics['GPT']['accuracy'],
        per_lesion_metrics['Gemini']['accuracy']
    ]

    group_frame_sensitivity = [
        per_frame_metrics['CAD']['overall_metrics']['recall'],
        per_frame_metrics['GPT']['overall_metrics']['recall'],
        per_frame_metrics['Gemini']['overall_metrics']['recall']
    ]

    group_frame_specificity = [
        per_frame_metrics['CAD']['overall_metrics']['specificity'],
        per_frame_metrics['GPT']['overall_metrics']['specificity'],
        per_frame_metrics['Gemini']['overall_metrics']['specificity']
    ]

    group_frame_accuracy = [
        per_frame_metrics['CAD']['overall_metrics']['accuracy'],
        per_frame_metrics['GPT']['overall_metrics']['accuracy'],
        per_frame_metrics['Gemini']['overall_metrics']['accuracy']
    ]

    group_frame_bbox_iou = [
        per_frame_bbox_metrics['CAD']['overall_metrics']['mean_iou'],
        per_frame_bbox_metrics['Gemini']['overall_metrics']['mean_iou']
    ]

    group_lesion_sensitivity_ci = [
        per_lesion_metrics['CAD']['recall_95%_CI'],
        per_lesion_metrics['GPT']['recall_95%_CI'],
        per_lesion_metrics['Gemini']['recall_95%_CI']
    ]

    group_lesion_specificity_ci = [
        per_lesion_metrics['CAD']['specificity_95%_CI'],
        per_lesion_metrics['GPT']['specificity_95%_CI'],
        per_lesion_metrics['Gemini']['specificity_95%_CI']
    ]

    group_lesion_accuracy_ci = [
        per_lesion_metrics['CAD']['accuracy_95%_CI'],
        per_lesion_metrics['GPT']['accuracy_95%_CI'],
        per_lesion_metrics['Gemini']['accuracy_95%_CI']
    ]

    group_frame_sensitivity_ci = [
        per_frame_metrics['CAD']['overall_metrics']['recall_95%_CI'],
        per_frame_metrics['GPT']['overall_metrics']['recall_95%_CI'],
        per_frame_metrics['Gemini']['overall_metrics']['recall_95%_CI']
    ]

    group_frame_specificity_ci = [
        per_frame_metrics['CAD']['overall_metrics']['specificity_95%_CI'],
        per_frame_metrics['GPT']['overall_metrics']['specificity_95%_CI'],
        per_frame_metrics['Gemini']['overall_metrics']['specificity_95%_CI']
    ]

    group_frame_accuracy_ci = [
        per_frame_metrics['CAD']['overall_metrics']['accuracy_95%_CI'],
        per_frame_metrics['GPT']['overall_metrics']['accuracy_95%_CI'],
        per_frame_metrics['Gemini']['overall_metrics']['accuracy_95%_CI']
    ]

    group_frame_bbox_iou_ci = [
        per_frame_bbox_metrics['CAD']['overall_metrics']['mean_iou_95%_CI'],
        per_frame_bbox_metrics['Gemini']['overall_metrics']['mean_iou_95%_CI']
    ]

    group_lesion_sensitivity_p = [
        p_values['CAD_vs_GPT']['recall'],
        p_values['CAD_vs_Gemini']['recall'],
        p_values['GPT_vs_Gemini']['recall']
    ]

    group_lesion_specificity_p = [
        p_values['CAD_vs_GPT']['specificity'],
        p_values['CAD_vs_Gemini']['specificity'],
        p_values['GPT_vs_Gemini']['specificity']
    ]

    group_lesion_accuracy_p = [
        p_values['CAD_vs_GPT']['accuracy'],
        p_values['CAD_vs_Gemini']['accuracy'],
        p_values['GPT_vs_Gemini']['accuracy']
    ]

    group_frame_sensitivity_p = [
        p_values['CAD_vs_GPT']['recall'],
        p_values['CAD_vs_Gemini']['recall'],
        p_values['GPT_vs_Gemini']['recall']
    ]

    group_frame_specificity_p = [
        p_values['CAD_vs_GPT']['specificity'],
        p_values['CAD_vs_Gemini']['specificity'],
        p_values['GPT_vs_Gemini']['specificity']
    ]

    group_frame_accuracy_p = [
        p_values['CAD_vs_GPT']['accuracy'],
        p_values['CAD_vs_Gemini']['accuracy'],
        p_values['GPT_vs_Gemini']['accuracy']
    ]

    group_frame_bbox_iou_p = [
        p_values_mean_iou['CAD_vs_Gemini']['mean_iou_p_value']
    ]

    values = [
        group_lesion_sensitivity,
        group_lesion_specificity,
        group_lesion_accuracy,
        group_frame_sensitivity,
        group_frame_specificity,
        group_frame_accuracy,
        group_frame_bbox_iou
    ]

    ci_values = [
        group_lesion_sensitivity_ci,
        group_lesion_specificity_ci,
        group_lesion_accuracy_ci,
        group_frame_sensitivity_ci,
        group_frame_specificity_ci,
        group_frame_accuracy_ci,
        group_frame_bbox_iou_ci
    ]

    p_values_list = [
        group_lesion_sensitivity_p,
        group_lesion_specificity_p,
        group_lesion_accuracy_p,
        group_frame_sensitivity_p,
        group_frame_specificity_p,
        group_frame_accuracy_p,
        group_frame_bbox_iou_p
    ]

    # In p values if there is nan set it to 1
    for i in range(len(p_values_list)):
        for j in range(len(p_values_list[i])):
            if p_values_list[i][j] is None or np.isnan(p_values_list[i][j]):
                p_values_list[i][j] = 1

    bar_labels = ['CAD', 'GPT', 'Gemini']
    group_labels = ['Per Lesion Sensitivity', 'Per Lesion Specificity', 'Per Lesion Accuracy', 'Per Frame Sensitivity',
                    'Per Frame Specificity', 'Per Frame Accuracy', 'Mean IoU']

    create_grouped_bar_plot_with_min_max_and_p_values(values, ci_values, p_values_list, group_labels=group_labels,
                                                      bar_labels=bar_labels, title='Overall Comparison',
                                                      save_path=os.path.join(output_images_folder,
                                                                             'overall_comparison.png'),
                                                      color_choice='grayscale')

    # overall comparison with per frame sensitivity and specificity and per lesion sensitivity
    group_frame_sensitivity = [
        per_frame_metrics['CAD']['overall_metrics']['recall'],
        per_frame_metrics['GPT']['overall_metrics']['recall'],
        per_frame_metrics['Gemini']['overall_metrics']['recall']
    ]

    group_frame_sensitivity_ci = [
        per_frame_metrics['CAD']['overall_metrics']['recall_95%_CI'],
        per_frame_metrics['GPT']['overall_metrics']['recall_95%_CI'],
        per_frame_metrics['Gemini']['overall_metrics']['recall_95%_CI']
    ]

    group_frame_sensitivity_p = [
        p_values_frame['CAD_vs_GPT']['recall'],
        p_values_frame['CAD_vs_Gemini']['recall'],
        p_values_frame['GPT_vs_Gemini']['recall']
    ]

    group_frame_specificity = [
        per_frame_metrics['CAD']['overall_metrics']['specificity'],
        per_frame_metrics['GPT']['overall_metrics']['specificity'],
        per_frame_metrics['Gemini']['overall_metrics']['specificity']
    ]

    group_frame_specificity_ci = [
        per_frame_metrics['CAD']['overall_metrics']['specificity_95%_CI'],
        per_frame_metrics['GPT']['overall_metrics']['specificity_95%_CI'],
        per_frame_metrics['Gemini']['overall_metrics']['specificity_95%_CI']
    ]

    group_frame_specificity_p = [
        p_values_frame['CAD_vs_GPT']['specificity'],
        p_values_frame['CAD_vs_Gemini']['specificity'],
        p_values_frame['GPT_vs_Gemini']['specificity']
    ]

    group_lesion_sensitivity = [
        per_lesion_metrics['CAD']['recall'],
        per_lesion_metrics['GPT']['recall'],
        per_lesion_metrics['Gemini']['recall']
    ]

    group_lesion_sensitivity_ci = [
        per_lesion_metrics['CAD']['recall_95%_CI'],
        per_lesion_metrics['GPT']['recall_95%_CI'],
        per_lesion_metrics['Gemini']['recall_95%_CI']
    ]

    group_lesion_sensitivity_p = [
        p_values['CAD_vs_GPT']['recall'],
        p_values['CAD_vs_Gemini']['recall'],
        p_values['GPT_vs_Gemini']['recall']
    ]

    values = [
        group_frame_sensitivity,
        group_frame_specificity,
        group_lesion_sensitivity
    ]

    ci_values = [
        group_frame_sensitivity_ci,
        group_frame_specificity_ci,
        group_lesion_sensitivity_ci
    ]

    p_values_list = [
        group_frame_sensitivity_p,
        group_frame_specificity_p,
        group_lesion_sensitivity_p
    ]

    # In p values if there is nan set it to 1
    for i in range(len(p_values_list)):
        for j in range(len(p_values_list[i])):
            if p_values_list[i][j] is None or np.isnan(p_values_list[i][j]):
                p_values_list[i][j] = 1

    bar_labels = ['CAD', 'GPT', 'Gemini']
    group_labels = ['Per Frame Sensitivity', 'Per Frame Specificity', 'Per Lesion Sensitivity']

    create_grouped_bar_plot_with_min_max_and_p_values(values, ci_values, p_values_list, group_labels=group_labels,
                                                      bar_labels=bar_labels, title='Overall Comparison',
                                                      save_path=os.path.join(output_images_folder,
                                                                             'overall_comparison_NEW.png'),
                                                      color_choice='grayscale')


if __name__ == '__main__':
    main()
