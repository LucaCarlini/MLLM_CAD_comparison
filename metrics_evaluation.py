import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm
import pandas as pd
import seaborn as sns

from bar_plot import create_grouped_bar_plot_with_min_max_and_p_values

def load_json_files(base_folder_path):
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

def compute_proportion_confidence_interval(successes, n, confidence=0.95):
    if n == 0:
        return (0.0, 0.0)
    return proportion_confint(successes, n, alpha=1 - confidence, method='beta')

def evaluate_frames_detection(video_data):
    global_tp = global_fp = global_fn = global_tn = 0

    size_categories = ['diminutive', 'small', 'medium', 'large']
    size_category_counts_frame = {cat: {'tp':0, 'fp':0, 'fn':0, 'tn':0} for cat in size_categories}

    for case_id, case_data in video_data.items():
        category = case_data['category']
        for frame_data in case_data['frames']:
            has_annotation = len(frame_data['annotations_bbox']) > 0 and any(
                bbox != [0, 0, 0, 0] for bbox in frame_data['annotations_bbox']
            )
            has_detection = frame_data['detections']
            size_category = frame_data['size_category'] if category == 'Positive' else None

            if has_annotation and has_detection:
                global_tp += 1
                if size_category in size_categories:
                    size_category_counts_frame[size_category]['tp'] += 1
            elif has_annotation and not has_detection:
                global_fn += 1
                if size_category in size_categories:
                    size_category_counts_frame[size_category]['fn'] += 1
            elif not has_annotation and has_detection:
                global_fp += 1
            else:
                global_tn += 1

    overall_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0.0
    overall_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0.0
    overall_specificity = global_tn / (global_tn + global_fp) if (global_tn + global_fp) > 0 else 0.0

    precision_ci = compute_proportion_confidence_interval(global_tp, global_tp + global_fp) if (global_tp + global_fp) > 0 else (0.0, 0.0)
    recall_ci = compute_proportion_confidence_interval(global_tp, global_tp + global_fn) if (global_tp + global_fn) > 0 else (0.0, 0.0)
    specificity_ci = compute_proportion_confidence_interval(global_tn, global_tn + global_fp) if (global_tn + global_fp) > 0 else (0.0, 0.0)

    size_category_metrics_frame = {}
    for cat, counts in size_category_counts_frame.items():
        cat_tp = counts['tp']
        cat_fp = counts['fp']
        cat_fn = counts['fn']
        cat_tn = counts['tn']

        cat_precision = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0
        cat_recall = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0
        cat_specificity = cat_tn / (cat_tn + cat_fp) if (cat_tn + cat_fp) > 0 else 0

        cat_precision_ci = compute_proportion_confidence_interval(cat_tp, cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else (0.0, 0.0)
        cat_recall_ci = compute_proportion_confidence_interval(cat_tp, cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else (0.0, 0.0)
        cat_specificity_ci = compute_proportion_confidence_interval(cat_tn, cat_tn + cat_fp) if (cat_tn + cat_fp) > 0 else (0.0, 0.0)

        size_category_metrics_frame[cat] = {
            'precision': cat_precision,
            'precision_95%_CI': cat_precision_ci,
            'recall': cat_recall,
            'recall_95%_CI': cat_recall_ci,
            'specificity': cat_specificity,
            'specificity_95%_CI': cat_specificity_ci,
            'counts': counts
        }

    overall_metrics = {
        'tp_total': global_tp,
        'fp_total': global_fp,
        'fn_total': global_fn,
        'tn_total': global_tn,
        'precision': overall_precision,
        'precision_95%_CI': precision_ci,
        'recall': overall_recall,
        'recall_95%_CI': recall_ci,
        'specificity': overall_specificity,
        'specificity_95%_CI': specificity_ci
    }

    return {
        'overall_metrics': overall_metrics,
        'size_category_metrics_frame': size_category_metrics_frame
    }

def process_json_data(data, histology_data, model_key, output_folder, iou_threshold=0.5):
    video_data = {}

    for case_data in tqdm(data, desc=f"Processing {model_key} data"):
        case_id = case_data['details']['Case']
        category = case_data['details']['Category']
        case_ref = case_id + '_' + category

        if case_ref not in video_data:
            video_data[case_ref] = {
                'category': category,
                'total_frames': len(case_data['frames']),
                'frames': []
            }

        for frame in case_data['frames']:
            frame_number = frame['Frame']
            if category == 'Positive':
                size_str = histology_data.get(case_id, {}).get('Size', 'Unknown')
                size_str = size_str.replace('mm', '')
                size = float(size_str) if size_str.isdigit() else 0.0

                if size <= 3:
                    size_category = 'diminutive'
                elif size <= 5:
                    size_category = 'small'
                elif size <= 10:
                    size_category = 'medium'
                else:
                    size_category = 'large'

                frame_data = {
                    'frame_number': frame_number,
                    'annotations_bbox': frame['Annotation'].get('Bounding boxes', []),
                    'detections': None,
                    'size_category': size_category
                }
            else:
                frame_data = {
                    'frame_number': frame_number,
                    'annotations_bbox': [],
                    'detections': None,
                    'size_category': None
                }

            # Extract model detections
            if model_key == 'CAD':
                bbox = frame[model_key].get('Bounding boxes', [])
                frame_data['detections'] = bool(bbox and bbox != [[0, 0, 0, 0]])
            elif model_key in ['GPT', 'Gemini']:
                frame_data['detections'] = frame[model_key]['Lesion detected'] == 'True'

            video_data[case_ref]['frames'].append(frame_data)

    metrics_no_bbox = evaluate_frames_detection(video_data)
    per_lesion_metrics = compute_per_lesion_metrics(video_data)

    return {
        'metrics_no_bbox': metrics_no_bbox,
        'per_lesion_metrics': per_lesion_metrics,
    }

def compute_per_lesion_metrics(video_data):
    tp = fp = fn = tn = 0
    size_categories = ['diminutive', 'small', 'medium', 'large']
    size_category_counts = {cat: {'tp':0, 'fp':0, 'fn':0, 'tn':0} for cat in size_categories}

    for case_id, case_data in video_data.items():
        category = case_data['category']
        total_video_frames = case_data['total_frames']
        lesion_detected = sum(frame['detections'] for frame in case_data['frames']) > (total_video_frames / 2)

        size_category = None
        if category == 'Positive':
            size_category = case_data['frames'][0]['size_category']

        if category == 'Positive':
            if lesion_detected:
                tp += 1
                if size_category in size_categories:
                    size_category_counts[size_category]['tp'] += 1
            else:
                fn += 1
                if size_category in size_categories:
                    size_category_counts[size_category]['fn'] += 1
        elif category == 'Negative':
            if lesion_detected:
                fp += 1
            else:
                tn += 1

    total_cases = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    precision_ci = compute_proportion_confidence_interval(tp, tp + fp) if (tp + fp) > 0 else (0.0, 0.0)
    recall_ci = compute_proportion_confidence_interval(tp, tp + fn) if (tp + fn) > 0 else (0.0, 0.0)
    specificity_ci = compute_proportion_confidence_interval(tn, tn + fp) if (tn + fp) > 0 else (0.0, 0.0)

    size_category_metrics = {}
    for cat, counts in size_category_counts.items():
        cat_tp = counts['tp']
        cat_fp = counts['fp']
        cat_fn = counts['fn']
        cat_tn = counts['tn']

        cat_precision = cat_tp / (cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else 0
        cat_recall = cat_tp / (cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else 0
        cat_specificity = cat_tn / (cat_tn + cat_fp) if (cat_tn + cat_fp) > 0 else 0

        cat_precision_ci = compute_proportion_confidence_interval(cat_tp, cat_tp + cat_fp) if (cat_tp + cat_fp) > 0 else (0.0, 0.0)
        cat_recall_ci = compute_proportion_confidence_interval(cat_tp, cat_tp + cat_fn) if (cat_tp + cat_fn) > 0 else (0.0, 0.0)
        cat_specificity_ci = compute_proportion_confidence_interval(cat_tn, cat_tn + cat_fp) if (cat_tn + cat_fp) > 0 else (0.0, 0.0)

        size_category_metrics[cat] = {
            'precision': cat_precision,
            'precision_95%_CI': cat_precision_ci,
            'recall': cat_recall,
            'recall_95%_CI': cat_recall_ci,
            'specificity': cat_specificity,
            'specificity_95%_CI': cat_specificity_ci,
            'counts': counts
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
        'size_category_metrics': size_category_metrics
    }

    return per_lesion_metrics

def save_metrics(metrics, output_folder, filename):
    output_path = os.path.join(output_folder, f"{filename}.json")
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {output_path}")

def compare_metrics(tp1, fp1, fn1, tn1, tp2, fp2, fn2, tn2):
    table_precision = np.array([[tp1, fp1], [tp2, fp2]])
    table_recall = np.array([[tp1, fn1], [tp2, fn2]])
    table_specificity = np.array([[tn1, fn1], [tn2, fn2]])

    def perform_test(table):
        if np.any(table < 5):
            _, p_value = fisher_exact(table)
        else:
            chi2, p_value, dof, expected = chi2_contingency(table)

        return p_value

    p_values = {
        'precision': perform_test(table_precision),
        'recall': perform_test(table_recall),
        'specificity': perform_test(table_specificity)
    }
    return p_values

def compare_per_frame_metrics(metrics_model1, metrics_model2):
    overall1 = metrics_model1['overall_metrics']
    overall2 = metrics_model2['overall_metrics']

    tp1, fp1, fn1, tn1 = overall1['tp_total'], overall1['fp_total'], overall1['fn_total'], overall1['tn_total']
    tp2, fp2, fn2, tn2 = overall2['tp_total'], overall2['fp_total'], overall2['fn_total'], overall2['tn_total']

    return compare_metrics(tp1, fp1, fn1, tn1, tp2, fp2, fn2, tn2)

def compare_per_lesion_metrics(metrics_model1, metrics_model2):
    tp1, fp1, fn1, tn1 = metrics_model1['total_tp'], metrics_model1['total_fp'], metrics_model1['total_fn'], metrics_model1['total_tn']
    tp2, fp2, fn2, tn2 = metrics_model2['total_tp'], metrics_model2['total_fp'], metrics_model2['total_fn'], metrics_model2['total_tn']

    return compare_metrics(tp1, fp1, fn1, tn1, tp2, fp2, fn2, tn2)


def compare_models_for_sizes(size_metrics_all_models):
    """
    Compare models for each size category on recall.
    Recall = TP/(TP+FN)
    Comparisons: CAD vs GPT, CAD vs Gemini, GPT vs Gemini
    """
    comparisons = [('CAD', 'GPT'), ('CAD', 'Gemini'), ('GPT', 'Gemini')]
    size_categories = ['diminutive', 'small', 'medium', 'large']

    p_values_per_size = {}
    for cat in size_categories:
        p_values_per_size[cat] = {}

        def get_tp_fn(model, cat):
            cat_data = size_metrics_all_models[model].get(cat, {})
            counts = cat_data.get('counts', {})
            tp = counts.get('tp', 0)
            fn = counts.get('fn', 0)
            return tp, fn

        for (m1, m2) in comparisons:
            tp1, fn1 = get_tp_fn(m1, cat)
            tp2, fn2 = get_tp_fn(m2, cat)

            table = np.array([[tp1, fn1], [tp2, fn2]])

            if np.any(table < 5):
                _, p_value = fisher_exact(table)
                #print(f"Fisher's Exact Test used for size category '{cat}', comparison {m1} vs {m2}: Table {table}")
            else:
                chi2, p_value, dof, expected = chi2_contingency(table)
                #print(f"Chi-Square Test used for size category '{cat}', comparison {m1} vs {m2}: Table {table}")

            comparison_key = f"{m1}_vs_{m2}"
            p_values_per_size[cat][comparison_key] = p_value

    return p_values_per_size


def compare_sizes_for_model(size_metrics_one_model):
    """
    Compare size categories within the same model based on recall (TP/(TP+FN)).
    Comparisons: diminutive vs small, diminutive vs medium, diminutive vs large,
                 small vs medium, small vs large, medium vs large.
    """
    categories = ['diminutive', 'small', 'medium', 'large']
    p_values = {}
    comparisons = [
        ('diminutive', 'small'),
        ('diminutive', 'medium'),
        ('diminutive', 'large'),
        ('small', 'medium'),
        ('small', 'large'),
        ('medium', 'large')
    ]

    for (cat1, cat2) in comparisons:
        tp1 = size_metrics_one_model.get(cat1, {}).get('counts', {}).get('tp', 0)
        fn1 = size_metrics_one_model.get(cat1, {}).get('counts', {}).get('fn', 0)
        tp2 = size_metrics_one_model.get(cat2, {}).get('counts', {}).get('tp', 0)
        fn2 = size_metrics_one_model.get(cat2, {}).get('counts', {}).get('fn', 0)

        table = np.array([[tp1, fn1], [tp2, fn2]])
        if np.any(table < 5):
            _, p = fisher_exact(table)
        else:
            chi2, p, dof, exp = chi2_contingency(table)

        comparison_key = f"{cat1}_vs_{cat2}"
        p_values[comparison_key] = p

    return p_values


def main():
    base_folder_path = os.path.join(os.getcwd(), 'bounding_boxes')
    histology_path = os.path.join(os.getcwd(), "video_output_fullDs", "hystology.json")
    output_folder = os.path.join(base_folder_path, 'metrics_output')
    os.makedirs(output_folder, exist_ok=True)

    with open(histology_path, 'r') as file:
        histology_data = json.load(file)

    data = load_json_files(base_folder_path)

    model_keys = ['GPT', 'Gemini', 'CAD']
    metrics_model_no_bbox = {}
    per_lesion_metrics_model = {}
    size_category_metrics_model = {}
    size_category_metrics_frame_model = {}

    # Process each model
    for model_key in model_keys:
        metrics = process_json_data(data, histology_data, model_key, output_folder, iou_threshold=0.5)
        metrics_model_no_bbox[model_key] = metrics['metrics_no_bbox']
        per_lesion_metrics_model[model_key] = metrics['per_lesion_metrics']
        size_category_metrics_model[model_key] = metrics['per_lesion_metrics']['size_category_metrics']
        size_category_metrics_frame_model[model_key] = metrics['metrics_no_bbox']['size_category_metrics_frame']

    # Save results
    save_metrics(metrics_model_no_bbox, output_folder, 'metrics_no_bbox')
    save_metrics(per_lesion_metrics_model, output_folder, 'per_lesion_metrics')
    save_metrics(size_category_metrics_model, output_folder, 'size_category_metrics_lesion')
    save_metrics(size_category_metrics_frame_model, output_folder, 'size_category_metrics_frame')

    print("Metrics evaluation completed.")

    # Statistical comparisons between models (per frame)
    comparisons = [('CAD', 'GPT'), ('CAD', 'Gemini'), ('GPT', 'Gemini')]
    p_values_per_frame = {}
    for model1, model2 in comparisons:
        p_values = compare_per_frame_metrics(metrics_model_no_bbox[model1], metrics_model_no_bbox[model2])
        comparison_key = f"{model1}_vs_{model2}"
        p_values_per_frame[comparison_key] = p_values

    output_file_path = os.path.join(output_folder, 'statistical_results_frame_no_bbox.json')
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

    # Load size category metrics
    lesion_size_path = os.path.join(output_folder, 'size_category_metrics_lesion.json')
    frame_size_path = os.path.join(output_folder, 'size_category_metrics_frame.json')

    with open(lesion_size_path, 'r') as f:
        lesion_size_metrics = json.load(f)

    with open(frame_size_path, 'r') as f:
        frame_size_metrics = json.load(f)

    # Compare models for each size category at lesion-level and frame-level
    # The dictionary structure: { 'CAD': {...}, 'GPT': {...}, 'Gemini': {...} }
    # Each is {size_category: {counts: {tp, fn, ...}, ...}}
    size_p_values_lesion_models = compare_models_for_sizes(lesion_size_metrics)
    size_p_values_frame_models = compare_models_for_sizes(frame_size_metrics)

    output_file_path = os.path.join(output_folder, 'statistical_results_size_recall_lesion_models.json')
    with open(output_file_path, 'w') as f:
        json.dump(size_p_values_lesion_models, f, indent=4)
    print(f"Size category recall p-values (lesion-level, model comparisons) saved to {output_file_path}")

    output_file_path = os.path.join(output_folder, 'statistical_results_size_recall_frame_models.json')
    with open(output_file_path, 'w') as f:
        json.dump(size_p_values_frame_models, f, indent=4)
    print(f"Size category recall p-values (frame-level, model comparisons) saved to {output_file_path}")

    # Visualization of overall results (as before)
    output_images_folder = os.path.join(output_folder, 'results_plots')
    os.makedirs(output_images_folder, exist_ok=True)

    per_lesion_path = os.path.join(output_folder, 'per_lesion_metrics.json')
    per_frame_path = os.path.join(output_folder, 'metrics_no_bbox.json')

    with open(per_lesion_path, 'r') as f:
        per_lesion_metrics = json.load(f)

    with open(per_frame_path, 'r') as f:
        per_frame_metrics = json.load(f)

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
                                                      color_choice='grayscale', apply_bonferroni_correction=True, bonferroni_div=3)

    # bar plot for size categories

    # load size category metrics
    size_category_metrics_path = os.path.join(output_folder, 'size_category_metrics_frame.json')
    with open(size_category_metrics_path, 'r') as f:
        size_category_metrics = json.load(f)

    p_values_size_path = os.path.join(output_folder, 'statistical_results_size_recall_frame_models.json')
    with open(p_values_size_path, 'r') as f:
        p_values_size = json.load(f)

    group_size_sensitivity_diminutive = [
        size_category_metrics['CAD'].get('diminutive', {}).get('recall', 0),
        size_category_metrics['GPT'].get('diminutive', {}).get('recall', 0),
        size_category_metrics['Gemini'].get('diminutive', {}).get('recall', 0)
    ]

    group_size_sensitivity_small = [
        size_category_metrics['CAD'].get('small', {}).get('recall', 0),
        size_category_metrics['GPT'].get('small', {}).get('recall', 0),
        size_category_metrics['Gemini'].get('small', {}).get('recall', 0)
    ]

    group_size_sensitivity_medium = [
        size_category_metrics['CAD'].get('medium', {}).get('recall', 0),
        size_category_metrics['GPT'].get('medium', {}).get('recall', 0),
        size_category_metrics['Gemini'].get('medium', {}).get('recall', 0)
    ]

    group_size_sensitivity_large = [
        size_category_metrics['CAD'].get('large', {}).get('recall', 0),
        size_category_metrics['GPT'].get('large', {}).get('recall', 0),
        size_category_metrics['Gemini'].get('large', {}).get('recall', 0)
    ]

    group_size_sensitivity = [group_size_sensitivity_diminutive, group_size_sensitivity_small, group_size_sensitivity_medium, group_size_sensitivity_large]

    group_size_sensitivity_ci_diminutive = [
        size_category_metrics['CAD'].get('diminutive', {}).get('recall_95%_CI', [0, 0]),
        size_category_metrics['GPT'].get('diminutive', {}).get('recall_95%_CI', [0, 0]),
        size_category_metrics['Gemini'].get('diminutive', {}).get('recall_95%_CI', [0, 0])
    ]

    group_size_sensitivity_ci_small = [
        size_category_metrics['CAD'].get('small', {}).get('recall_95%_CI', [0, 0]),
        size_category_metrics['GPT'].get('small', {}).get('recall_95%_CI', [0, 0]),
        size_category_metrics['Gemini'].get('small', {}).get('recall_95%_CI', [0, 0])
    ]

    group_size_sensitivity_ci_medium = [
        size_category_metrics['CAD'].get('medium', {}).get('recall_95%_CI', [0, 0]),
        size_category_metrics['GPT'].get('medium', {}).get('recall_95%_CI', [0, 0]),
        size_category_metrics['Gemini'].get('medium', {}).get('recall_95%_CI', [0, 0])
    ]

    group_size_sensitivity_ci_large = [
        size_category_metrics['CAD'].get('large', {}).get('recall_95%_CI', [0, 0]),
        size_category_metrics['GPT'].get('large', {}).get('recall_95%_CI', [0, 0]),
        size_category_metrics['Gemini'].get('large', {}).get('recall_95%_CI', [0, 0])
    ]

    group_size_sensitivity_ci = [group_size_sensitivity_ci_diminutive, group_size_sensitivity_ci_small, group_size_sensitivity_ci_medium, group_size_sensitivity_ci_large]

    group_size_sensitivity_p_diminutive = [
        p_values_size['diminutive']['CAD_vs_GPT'],
        p_values_size['diminutive']['CAD_vs_Gemini'],
        p_values_size['diminutive']['GPT_vs_Gemini']
    ]

    group_size_sensitivity_p_small = [
        p_values_size['small']['CAD_vs_GPT'],
        p_values_size['small']['CAD_vs_Gemini'],
        p_values_size['small']['GPT_vs_Gemini']
    ]

    group_size_sensitivity_p_medium = [
        p_values_size['medium']['CAD_vs_GPT'],
        p_values_size['medium']['CAD_vs_Gemini'],
        p_values_size['medium']['GPT_vs_Gemini']
    ]

    group_size_sensitivity_p_large = [
        p_values_size['large']['CAD_vs_GPT'],
        p_values_size['large']['CAD_vs_Gemini'],
        p_values_size['large']['GPT_vs_Gemini']
    ]

    group_size_sensitivity_p = [group_size_sensitivity_p_diminutive, group_size_sensitivity_p_small, group_size_sensitivity_p_medium, group_size_sensitivity_p_large]



    group_labels = [
        'Small\n(≤3mm)', 
        'Diminutive\n(>3mm & ≤5mm)', 
        'Medium\n(>5mm & ≤10mm)', 
        'Large\n(>10mm)'
    ]
    bar_labels = ['CAD', 'GPT', 'Gemini']
    title = 'Size Category Sensitivity'
    offsets = [0, 4, 0, 4]

    create_grouped_bar_plot_with_min_max_and_p_values(group_size_sensitivity, group_size_sensitivity_ci, group_size_sensitivity_p, group_labels=group_labels, bar_labels=bar_labels, title=title, save_path=os.path.join(output_images_folder, 'size_category_sensitivity.png'), color_choice='grayscale', offsets=offsets, apply_bonferroni_correction=True, bonferroni_div=12)

    # After calculating metrics for all models:
    model_keys = ['GPT', 'Gemini', 'CAD']
    # We already have size_category_metrics_frame_model from the original code.
    # Now we compare sizes within the same model:
    p_values_sizes_same_model = {}
    for model in model_keys:
        p_values_sizes_same_model[model] = compare_sizes_for_model(size_category_metrics_frame_model[model])

    # Save these results:
    with open(os.path.join(output_folder, 'statistical_results_size_categories.json'), 'w') as f:
        json.dump(p_values_sizes_same_model, f, indent=4)
    print("P-values for size category comparisons within each model saved.")

    # Visualization of size category comparisons within each model
    group_size_sensitivity_cad = [
        size_category_metrics_frame_model['CAD'].get('diminutive', {}).get('recall', 0),
        size_category_metrics_frame_model['CAD'].get('small', {}).get('recall', 0),
        size_category_metrics_frame_model['CAD'].get('medium', {}).get('recall', 0),
        size_category_metrics_frame_model['CAD'].get('large', {}).get('recall', 0)
    ]

    group_size_sensitivity_gpt = [
        size_category_metrics_frame_model['GPT'].get('diminutive', {}).get('recall', 0),
        size_category_metrics_frame_model['GPT'].get('small', {}).get('recall', 0),
        size_category_metrics_frame_model['GPT'].get('medium', {}).get('recall', 0),
        size_category_metrics_frame_model['GPT'].get('large', {}).get('recall', 0)
    ]

    group_size_sensitivity_gemini = [
        size_category_metrics_frame_model['Gemini'].get('diminutive', {}).get('recall', 0),
        size_category_metrics_frame_model['Gemini'].get('small', {}).get('recall', 0),
        size_category_metrics_frame_model['Gemini'].get('medium', {}).get('recall', 0),
        size_category_metrics_frame_model['Gemini'].get('large', {}).get('recall', 0)
    ]

    group_size_sensitivity = [group_size_sensitivity_cad, group_size_sensitivity_gpt, group_size_sensitivity_gemini]

    group_size_sensitivity_ci_cad = [
        size_category_metrics_frame_model['CAD'].get('diminutive', {}).get('recall_95%_CI', [0, 0]),
        size_category_metrics_frame_model['CAD'].get('small', {}).get('recall_95%_CI', [0, 0]),
        size_category_metrics_frame_model['CAD'].get('medium', {}).get('recall_95%_CI', [0, 0]),
        size_category_metrics_frame_model['CAD'].get('large', {}).get('recall_95%_CI', [0, 0])
    ]

    group_size_sensitivity_ci_gpt = [
        size_category_metrics_frame_model['GPT'].get('diminutive', {}).get('recall_95%_CI', [0, 0]),
        size_category_metrics_frame_model['GPT'].get('small', {}).get('recall_95%_CI', [0, 0]),
        size_category_metrics_frame_model['GPT'].get('medium', {}).get('recall_95%_CI', [0, 0]),
        size_category_metrics_frame_model['GPT'].get('large', {}).get('recall_95%_CI', [0, 0])
    ]

    group_size_sensitivity_ci_gemini = [
        size_category_metrics_frame_model['Gemini'].get('diminutive', {}).get('recall_95%_CI', [0, 0]),
        size_category_metrics_frame_model['Gemini'].get('small', {}).get('recall_95%_CI', [0, 0]),
        size_category_metrics_frame_model['Gemini'].get('medium', {}).get('recall_95%_CI', [0, 0]),
        size_category_metrics_frame_model['Gemini'].get('large', {}).get('recall_95%_CI', [0, 0])
    ]

    group_size_sensitivity_ci = [group_size_sensitivity_ci_cad, group_size_sensitivity_ci_gpt, group_size_sensitivity_ci_gemini]

    group_size_sensitivity_p_cad = [
        p_values_sizes_same_model['CAD']['diminutive_vs_small'],
        p_values_sizes_same_model['CAD']['diminutive_vs_medium'],
        p_values_sizes_same_model['CAD']['diminutive_vs_large'],
        p_values_sizes_same_model['CAD']['small_vs_medium'],
        p_values_sizes_same_model['CAD']['small_vs_large'],
        p_values_sizes_same_model['CAD']['medium_vs_large']
    ]

    group_size_sensitivity_p_gpt = [
        p_values_sizes_same_model['GPT']['diminutive_vs_small'],
        p_values_sizes_same_model['GPT']['diminutive_vs_medium'],
        p_values_sizes_same_model['GPT']['diminutive_vs_large'],
        p_values_sizes_same_model['GPT']['small_vs_medium'],
        p_values_sizes_same_model['GPT']['small_vs_large'],
        p_values_sizes_same_model['GPT']['medium_vs_large']
    ]

    group_size_sensitivity_p_gemini = [
        p_values_sizes_same_model['Gemini']['diminutive_vs_small'],
        p_values_sizes_same_model['Gemini']['diminutive_vs_medium'],
        p_values_sizes_same_model['Gemini']['diminutive_vs_large'],
        p_values_sizes_same_model['Gemini']['small_vs_medium'],
        p_values_sizes_same_model['Gemini']['small_vs_large'],
        p_values_sizes_same_model['Gemini']['medium_vs_large']
    ]

    group_size_sensitivity_p = [group_size_sensitivity_p_cad, group_size_sensitivity_p_gpt, group_size_sensitivity_p_gemini]

    group_labels = ['CAD', 'GPT', 'Gemini']
    bar_labels = ['Diminutive', 'Small', 'Medium', 'Large']

    create_grouped_bar_plot_with_min_max_and_p_values(group_size_sensitivity, group_size_sensitivity_ci, group_size_sensitivity_p, group_labels=group_labels, bar_labels=bar_labels, title='Size Category Sensitivity', save_path=os.path.join(output_images_folder, 'size_category_sensitivity_models.png'), color_choice='grayscale', apply_bonferroni_correction=True, bonferroni_div=18)




if __name__ == '__main__':
    main()
