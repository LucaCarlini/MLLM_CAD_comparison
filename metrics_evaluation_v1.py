import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2_contingency, fisher_exact, bootstrap
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
                size=histology_data.get(case_id, {}).get('Size', 'Unknown'),
                # remove 'mm' from the size
                size = size[0].replace('mm', '')
                # convert to float
                size = float(size)
                # split size into categories
                if size <= 3:
                    size_category = 'diminutive'
                elif size <= 5:
                    size_category = 'small'
                elif size <=10:
                    size_category = 'medium'
                else:
                    size_category = 'large'

                frame_data = {
                    'frame_number': frame_number,
                    'annotations_bbox': frame['Annotation'].get('Bounding boxes', []),
                    'detections': None,
                    'pred_bbox': None,
                    'histology_pred': None,
                    'histology_simpl_gt': histology_data.get(case_id, {}).get('Simplified diagnosis', 'Unknown'),
                    'histology_real_gt': histology_data.get(case_id, {}).get('Pathological diagnosis', 'Unknown'),
                    'polyp_size_gt': size,
                    'polyp_size_category': size_category
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
                    'polyp_size_gt': None,
                    'polyp_size_category': None
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

    return {
        'metrics_no_bbox': metrics_no_bbox,
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
    table_specificity = np.array([[tn1, fn1], [tn2, fn2]])
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
    table_specificity = np.array([[tn1, fn1], [tn2, fn2]])
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
        per_lesion_metrics_model[model_key] = metrics['per_lesion_metrics']

    # Save results
    save_metrics(metrics_model_no_bbox, output_folder, 'metrics_no_bbox')
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
    # create a json file with the p-value
    with open(output_file_path, 'w') as f:
        json.dump(p_values_per_frame, f, indent=4)
    print(f"Per-frame metrics p-values saved to {output_file_path}")

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



    # Visualization of results
    output_images_folder = os.path.join(output_folder, 'results_plots')
    os.makedirs(output_images_folder, exist_ok=True)



    # load metrics
    per_lesion_path = os.path.join(output_folder, 'per_lesion_metrics.json')
    per_frame_path = os.path.join(output_folder, 'metrics_no_bbox.json')
    per_frame_bbox_path = os.path.join(output_folder, 'metrics_bbox.json')

    with open(per_lesion_path, 'r') as f:
        per_lesion_metrics = json.load(f)

    with open(per_frame_path, 'r') as f:
        per_frame_metrics = json.load(f)



    # load p-values
    p_values_path = os.path.join(output_folder, 'statistical_results_per_lesion.json')
    p_values_frame_path = os.path.join(output_folder, 'statistical_results_frame_no_bbox.json')


    with open(p_values_path, 'r') as f:
        p_values = json.load(f)

    with open(p_values_frame_path, 'r') as f:
        p_values_frame = json.load(f)


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
