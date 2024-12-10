import os
import json
import argparse
import re
import cv2
import numpy as np
from moviepy.editor import VideoFileClip

# Scaling factors for GPT (from 1024x1024) and Gemini (from 1000x1000) to 1158x1008
GPT_SCALE_X = 1158 / 1000
GPT_SCALE_Y = 1008 / 1000
GEMINI_SCALE_X = 1158 / 1000
GEMINI_SCALE_Y = 1008 / 1000



def parse_args():
    parser = argparse.ArgumentParser(description='Analyze bounding box predictions from video summary files.')
    parser.add_argument('--category', type=str, choices=['Positive', 'Negative'], required=True, help='Choose between Positive or Negative videos.')
    parser.add_argument('--case', type=str, help='Specific case number to analyze.')
    #parser.add_argument('--analyze_all', action='store_true', help='Flag to analyze all cases both positive and negative.')
    parser.add_argument('--only_5videos', action='store_true', help='Limit extraction to the first 5 videos of the category.')
    parser.add_argument('--debug', action='store_true', help='Show debug information.')
    return parser.parse_args()

def rescale_bbox(bbox, scale_x, scale_y):
    """Rescale a bounding box according to the given scale factors."""
    x1, y1, x2, y2 = bbox
    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)
    return [x1, y1, x2, y2]

def load_annotations(annotation_file):
    annotations = {}
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
        for frame_number, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            filename_and_bbox = line.split(' ', maxsplit=1)
            if len(filename_and_bbox) != 2:
                #print(f"Missing bounding box: {line}")
                filename_and_bbox.append('')
            filename = filename_and_bbox[0]
            bbox_str = filename_and_bbox[1]
            bbox_parts = bbox_str.split(',')
            if len(bbox_parts) >= 5:
                x1, y1, x2, y2, label = bbox_parts[:5]
                x1 = int(x1)
                x2 = int(x2)
                y1 = int(y1)
                y2 = int(y2)
                label = int(label)
                if frame_number % 30 == 0:


                    annotations[str(frame_number // 30 + 1)] = {
                        'Bounding boxes': [[x1, y1, x2, y2]],
                        'Label': label
                    }
    return annotations

def load_cad_annotations(cad_file):
    cad_annotations = {}
    with open(cad_file, 'r') as f:
        lines = f.readlines()
        for frame_number, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ', maxsplit=1)
            if len(parts) < 2:
                filename = parts[0]
                bboxes = []
            else:
                filename, bbox_str = parts
                bbox_parts = bbox_str.split(' ')
                bboxes = []
                for bbox in bbox_parts:
                    x1, y1, x2, y2 = map(int, bbox.split(','))
                    bboxes.append([x1, y1, x2, y2])
            if frame_number % 30 == 0:

                cad_annotations[str(frame_number // 30 + 1)] = {
                    'Bounding boxes': bboxes
                }

    return cad_annotations

def parse_summary_gemini(summary_text):
    frames_data = {}
    time_taken = None

    if not summary_text:
        return frames_data

    # Split summary into blocks for each frame
    frames = summary_text.strip().split('\n\n')


    for frame in frames:
        lines = frame.strip().split('\n')

        if len(lines) >= 6:
            frame_line = lines[0]
            time_line = lines[1]
            lesion_detected_line = lines[2]
            bbox_line = lines[3]
            histology_line = lines[4]
            findings_line = lines[5]




            # Extract the frame number
            frame_match = re.search(r'Frame number: (\d+)', frame_line)
            if frame_match:
                frame_number = frame_match.group(1)

                # Extraxt if lesion detected
                lesion_detected = re.search(r'Lesion detected: (.*)', lesion_detected_line)
                lesion_detected = lesion_detected.group(1) if lesion_detected else "N/A"

                # Extract the histology
                histology_match = re.search(r'Histology lesion type: (.*)', histology_line)
                histology = histology_match.group(1) if histology_match else "N/A"



                if histology == 'N/A':
                    print(histology_line)
                # Extract the frame time
                frame_time_match = re.search(r'Frame time: (.*)', time_line)
                frame_time = frame_time_match.group(1) if frame_time_match else "N/A"

                # Extract the bounding box and rescale for 1158x1008
                bbox_match = re.search(r'Bounding box coordinates: (\[.*?\])', bbox_line)
                if bbox_match:
                    bbox_str = bbox_match.group(1)
                    try:
                        bbox = json.loads(bbox_str)
                        # Rescale from 1000x1000 to 1158x1008
                        bbox = rescale_bbox(bbox, GEMINI_SCALE_X, GEMINI_SCALE_Y)
                    except json.JSONDecodeError:
                        bbox = [0, 0, 0, 0]  # In case of malformed data
                else:
                    bbox = [0, 0, 0, 0]  # Default if no bbox is found

                # Extract findings
                findings_match = re.search(r'Findings: (.*)', findings_line)
                findings = findings_match.group(1) if findings_match else "N/A"

                # Store the extracted information
                frames_data[frame_number] = {
                    'Lesion detected': lesion_detected,
                    'Histology': histology,
                    'Bounding boxes': [bbox],
                    'Findings': findings,
                    'Frame time': frame_time,

                }



    # Extract the time taken if available
    time_match = re.search(r'Time taken: ([\d.]+) seconds', summary_text)
    if time_match:
        time_taken = float(time_match.group(1))



    time_taken_match = re.search(r'Time taken: ([\d.]+) seconds', summary_text)
    length_match = re.search(r'Video time lenght: ([\d\.]+) seconds', summary_text)
    if time_taken_match and length_match:
        time_taken = float(time_taken_match.group(1))
        video_duration = float(length_match.group(1))
        time_per_frame = time_taken / video_duration

    return frames_data, time_per_frame

def parse_summary_gpt(summary_text):
    frames_data = {}
    time_taken = None

    if not summary_text:
        return frames_data

    # Split summary into blocks for each frame
    frames = summary_text.strip().split('\n\n')

    for frame in frames:
        lines = frame.strip().split('\n')

        # Ensure at least  lines for each frame: Frame number, Lesion detected, bbox, histology, findings
        if len(lines) >= 4:
            frame_line = lines[0]
            lesion_detected_line = lines[1]
            #bbox_line = lines[2]
            histology_line = lines[2]
            findings_line = lines[3]


            # Extract the frame number
            frame_match = re.search(r'Frame number: (\d+)', frame_line)
            if frame_match:
                frame_number = frame_match.group(1)

                # Extraxt if lesion detected
                lesion_detected = re.search(r'Lesion detected: (.*)', lesion_detected_line)
                print(lesion_detected)
                lesion_detected = lesion_detected.group(1) if lesion_detected else "N/A"


                # Extract the histology
                histology_match = re.search(r'Histology lesion type: (.*)', histology_line)
                histology = histology_match.group(1) if histology_match else "N/A"


                # Extract the bounding box and rescale for 1158x1008
                # bbox_match = re.search(r'Bounding box coordinates: (\[.*?\])', bbox_line)
                # if bbox_match:
                #     bbox_str = bbox_match.group(1)
                #     try:
                #         bbox = json.loads(bbox_str)
                #         # Rescale from 1024x1024 to 1158x1008
                #         bbox = rescale_bbox(bbox, GPT_SCALE_X, GPT_SCALE_Y)
                #     except json.JSONDecodeError:
                #         bbox = [0, 0, 0, 0]  # In case of malformed data
                # else:
                #     bbox = [0, 0, 0, 0]  # Default if no bbox is found

                # Extract findings
                findings_match = re.search(r'Findings: (.*)', findings_line)
                findings = findings_match.group(1) if findings_match else "N/A"

                # Store the extracted information
                frames_data[frame_number] = {
                    'Lesion detected': lesion_detected,
                    'Histology': histology,
                    #'Bounding boxes': [bbox],
                    'Findings': findings,

                }


    # Extract the time taken if available
    time_match = re.search(r'Time taken: ([\d.]+) seconds', summary_text)
    if time_match:
        time_taken = float(time_match.group(1))

    time_taken_match = re.search(r'Time taken: ([\d.]+) seconds', summary_text)
    video_duration_match = re.search(r'Video duration: (\d+) frames', summary_text)
    if time_taken_match and video_duration_match:
        time_taken = float(time_taken_match.group(1))
        video_duration = int(video_duration_match.group(1))
        time_per_frame = time_taken / video_duration


    return frames_data, time_per_frame

def get_video_duration(video_file):
    clip = VideoFileClip(video_file)
    duration = clip.duration
    return duration

def draw_bounding_boxes(frame, annotations, gpt_data, gemini_data, cad_data):
    def draw_bbox(frame, bbox, color, label):
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for bbox in annotations.get('Bounding boxes', []):
            draw_bbox(frame, bbox, (0, 255, 0), 'Annotation')
    for bbox in gpt_data.get('Bounding boxes', []):
            draw_bbox(frame, bbox, (255, 0, 0), 'GPT')
    for bbox in gemini_data.get('Bounding boxes', []):
            draw_bbox(frame, bbox, (0, 0, 255), 'Gemini')
    for bbox in cad_data.get('Bounding boxes', []):
            draw_bbox(frame, bbox, (0, 255, 255), 'CAD')
    return frame



def main():
    args = parse_args()
    dataset_folder = r"C:\Users\lcarl\Desktop\GPT_4_colon\video_output_fullDs"


    category_folder = os.path.join(dataset_folder, args.category)
    output_folder = r'bounding_boxes'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    category_output_folder = os.path.join(output_folder, args.category)
    if not os.path.exists(category_output_folder):
        os.makedirs(category_output_folder)

    if args.case:
        cases = [args.case]
    else:
        cases = [d for d in os.listdir(category_folder) if os.path.isdir(os.path.join(category_folder, d))]

    cases = sorted(cases, key=lambda x: int(x[4:]))

    case_frames_dict = {}

    for case_name in cases:
        if args.only_5videos and int(case_name[4:]) > 5:
            break

        case_folder = os.path.join(category_folder, case_name)
        annotations = {}
        gpt_data = {}
        gemini_data = {}
        cad_data = {}
        gpt_time = None
        gemini_time = None
        video_duration_seconds = None
        video_file_path = None  # Store video file path


        for file in os.listdir(case_folder):
            file_path = os.path.join(case_folder, file)
            if file == 'image_annotations.txt':
                annotations = load_annotations(file_path)
            elif file.startswith('video_summary_gpt') and file.endswith('.txt'):
                with open(file_path, 'r') as f:
                    gpt_data, gpt_time = parse_summary_gpt(f.read())
            elif file.startswith('video_summary_gemini-1.5-pro-002') and file.endswith('.txt'):
                with open(file_path, 'r') as f:
                    gemini_data, gemini_time = parse_summary_gemini(f.read())
                    # Print the number of frames with histology 'None'
                    histology_none = 0

            elif file == f'{case_name}_video30seconds.mp4':
                video_duration_seconds = get_video_duration(file_path)
                video_file_path = file_path  # Store the video file path
            elif file == f'{case_name}_bbox_30fps.txt':
                cad_data = load_cad_annotations(file_path)

        if video_duration_seconds:
            max_frame_number = min(30, int(video_duration_seconds))
            case_frames = []

            if gpt_time is None:
                gpt_time = 0
            if gemini_time is None:
                gemini_time = 0

            case_info = {
                "details": {
                    "Case": case_name,
                    "Category": args.category,
                    "Video Duration (s)": video_duration_seconds,
                    "GPT time per frame (s)": gpt_time,
                    "Gemini time per second of video (s)": gemini_time,

                },
                "frames": []
            }

            for i in range(1, max_frame_number + 1):
                frame_number = i

                annotation = annotations.get(str(frame_number), {'Bounding boxes': [], 'Label': 'N/A'})
                gpt = gpt_data.get(str(frame_number), {'Lesion detected': 'N/A', 'Histology': 'N/A', 'Findings': 'N/A'})
                gemini = gemini_data.get(str(frame_number), {'Lesion detected': 'N/A', 'Histology': 'N/A', 'Bounding boxes': [], 'Findings': 'N/A', 'Frame time': 'N/A'})
                cad = cad_data.get(str(frame_number), {'Bounding boxes': []})

                frame_data = {
                    "Frame": frame_number,
                    "Annotation": annotation,
                    "GPT": gpt,
                    "Gemini": gemini,
                    "CAD": cad
                }
                case_info["frames"].append(frame_data)

            case_frames_dict[case_name] = case_info

    for case_name, case_data in case_frames_dict.items():
        output_file = os.path.join(category_output_folder, f'{case_name}_{args.category}_bounding_boxes.json')
        if os.path.exists(output_file):
            os.remove(output_file)
        with open(output_file, 'w') as file:
            json.dump(case_data, file, indent=4)

        if args.debug and video_file_path:
            # Open the video file
            cap = cv2.VideoCapture(video_file_path)
            if not cap.isOpened():
                print(f"Error opening video file {video_file_path}")
                continue

            for frame_data in case_data["frames"]:
                frame_number = frame_data["Frame"]
                annotation = frame_data["Annotation"]
                gpt = frame_data["GPT"]
                gemini = frame_data["Gemini"]
                cad = frame_data["CAD"]

                # Set the video to the correct frame (assuming 1 frame per second)
                cap.set(cv2.CAP_PROP_POS_MSEC, (frame_number - 1) * 1000)

                ret, frame = cap.read()
                if not ret:
                    print(f"Error reading frame {frame_number}")
                    continue

                # Resize the frame to match the target size (1158, 1008)
                target_size = (1158, 1008)
                resized_frame = cv2.resize(frame, target_size)

                # Draw bounding boxes
                resized_frame = draw_bounding_boxes(resized_frame, annotation, gpt, gemini, cad)

                # Add legend to the frame
                cv2.putText(resized_frame, 'Legend:', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(resized_frame, 'Green: Annotation', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                #cv2.putText(resized_frame, 'Blue: GPT', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(resized_frame, 'Red: Gemini', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(resized_frame, 'Yellow: CAD', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Resize for better display results
                resized_frame = cv2.resize(resized_frame, (target_size[0] // 2, target_size[1] // 2))

                # Display resized frame
                cv2.imshow(f'Frame {frame_number}', resized_frame)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    cap.release()
                    return
                cv2.destroyAllWindows()

            cap.release()



if __name__ == '__main__':
    main()