# GitHub Repository Summary: Comparative Analysis of GPT, Gemini, and CAD Models in Colonoscopy

This repository contains scripts used for generating videos, extracting bounding boxes, and evaluating performance metrics for a comparative analysis between GPT, Gemini, and CAD models in colonoscopy image analysis. These tools facilitate the evaluation of model outputs, including polyp detection, histological classification, and bounding box localization.

---

## **Repository Structure**

### **1. Video Generation**
- **Scripts:**
  - `create_video_entire_dataset.py`: Generates videos from a colonoscopy dataset, optionally including bounding boxes.
  - `create_video_entire_dataset_noborder.py`: Similar to the previous script but generates videos without visual borders.

### **2. Bounding Box Extraction**
- **Scripts:**
  - `detect_bbox.py`: Detects bounding boxes in colonoscopy frames using visual features and OCR for frame numbering.
  - `get_colonoscopic_image_position.py`: Identifies colonoscopy-specific regions in video frames using color segmentation.

### **3. Model-Specific Video Analysis**
- **Scripts:**
  - `GPT_video_analysis.py`: Analyzes colonoscopy videos using OpenAI’s GPT model and extracts lesion-related metrics.
  - `Gemini_video_analysis.py`: Analyzes videos using Google Generative AI’s Gemini model with detailed histological classification.

### **4. Metric Evaluation and Visualization**
- **Scripts:**
  - `metrics_evaluation.py`: Evaluates bounding box and lesion detection metrics, computes IoU, and generates summary statistics.
  - `bar_plot.py`: Generates comparative bar plots with statistical significance markers.

### **5. Data Summarization**
- **Script:**
  - `summarization.py`: Extracts, resizes, and summarizes bounding boxes from model outputs, adjusting for different output resolutions.

---

## **Key Features**
- **Video Generation**: Automatic video creation from colonoscopy datasets with bounding box overlay support.
- **Bounding Box Detection**: Frame-level bounding box extraction using OCR and color segmentation.
- **Model Integration**: Direct support for GPT, Gemini, and CAD-based video analysis.
- **Evaluation Metrics**: IoU, precision, recall, F1 score, histological classification metrics, and statistical tests.
- **Visualization**: Comprehensive plots showing performance comparisons.

---

## **Requirements**
Refer to `requirements.txt`for the complete list of dependencies, including:
- Core Libraries: `opencv-python`, `numpy`, `matplotlib`, `scipy`, `pandas`
- AI/ML Libraries: `google-generativeai`, `requests`
- Additional Tools: `pytesseract`, `moviepy`, `tqdm`

---

## **How to Use**
1. **Setup**: Clone the repository and install the required libraries.
2. **Generate Videos**: Use video generation scripts to create colonoscopy videos.
3. **Run Model Inference**: Perform analysis using GPT, Gemini, and CAD models.
4. **Evaluate Metrics**: Use evaluation scripts for detailed performance metrics and visualization.

---

This repository serves as a comprehensive evaluation framework for comparative research in colonoscopy image analysis. Explore the code, run the models, and gain insights into the strengths and limitations of each approach.