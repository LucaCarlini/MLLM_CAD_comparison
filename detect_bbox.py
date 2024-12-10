import cv2
import argparse
import numpy as np
import os
import json
import pytesseract
import skimage.feature as skf
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def parse_arguments():
    """
    Parse the input arguments for the script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process video frames and detect bounding boxes."
    )
    parser.add_argument(
        '--folder_path',
        type=str,
        default='video_output_fullDs/Positive/Case100',
        help="Path to the folder containing the video and frame name text file."
    )
    parser.add_argument(
        '--coords_path',
        type=str,
        default='video_output_fullDs/frame_coordinates.json',
        help="Path to the JSON file containing the coordinates of the region of interest."
    )
    parser.add_argument(
        '--min_area',
        type=int,
        default=100,
        help="Minimum area for a bounding box to be considered valid."
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help="Show debugging images."
    )
    parser.add_argument(
        '--all_cases',
        action='store_true',
        help="Process all cases in Positive and Negative folders."
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help="Overwrite existing bounding box files."
    )
    return parser.parse_args()

def find_files(folder_path):
    """
    Find the video and text files in the specified folder.

    Args:
        folder_path (str): Path to the folder containing video and text files.

    Returns:
        tuple: (video_path, txt_path) where both are strings.

    Raises:
        FileNotFoundError: If required files are not found.
    """
    video_path = None
    txt_path = None

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.mp4') and file_name.startswith('CAD') and '30fps' in file_name:
            video_path = os.path.join(folder_path, file_name)
        elif file_name.endswith('image_filenames.txt'):
            txt_path = os.path.join(folder_path, file_name)

    if not video_path or not txt_path:
        if not video_path:
            print(f"Video file not found in {folder_path}")
        if not txt_path:
            print(f"Text file not found in {folder_path}")
        raise FileNotFoundError("Required files (video and text file) not found in the specified folder.")

    print(f"Found video file: {video_path}")
    return video_path, txt_path

def load_frame_names(txt_path):
    """
    Load frame names from a text file.

    Args:
        txt_path (str): Path to the text file containing frame names.

    Returns:
        list: A list of frame names.
    """
    with open(txt_path, 'r') as file:
        frame_names = [line.strip() for line in file]
    return frame_names

def load_frame_coordinates(json_path):
    """
    Load frame coordinates from a JSON file.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        dict: Dictionary containing the frame coordinates and text bounding box info.
    """
    with open(json_path, 'r') as file:
        coordinates = json.load(file)
    return coordinates

def process_video(video_path, video_data, min_area, debug=False):
    """
    Process the video to detect bounding boxes in frames after a detected transition in numbers.

    Args:
        video_path (str): Path to the video file.
        video_data (dict): Dictionary with frame coordinates and text bounding box data.
        min_area (int): Minimum area for a valid bounding box.
        debug (bool): If True, show debugging images.

    Returns:
        dict: A dictionary mapping frame names to a list of bounding boxes.
    """
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return None

    fps_video = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps_video}, Total Frames: {total_frames}")

    frame_bboxes = {}
    start_counting = False
    prev_number = None
    frame_idx = 0

    # Coordinates for number region extraction
    text_bbox = video_data['Frame_text_bbox']
    x1, y1, x2, y2 = int(text_bbox['x1']), int(text_bbox['y1']), int(text_bbox['x2']), int(text_bbox['y2'])

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Extract the number region from the frame using provided coordinates
        number_region = frame[y1-5:y2, x1:x2]
        number_region_debug = frame[y1-5:y2, x1:x2+200]  # Wider region for debug visualization

        # Preprocess number region for OCR
        gray = cv2.cvtColor(number_region, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        if debug:
            cv2.imshow('Preprocessed Number Region', thresh)
            cv2.imshow('Original Number Region', number_region_debug)
            cv2.waitKey(1)

        # Detect the number using OCR
        config = '--psm 6 -c tessedit_char_whitelist=0123456789'
        number_text = pytesseract.image_to_string(thresh, config=config)
        try:
            number = int(number_text.strip())
        except ValueError:
            number = None

        # Detect transition from 1 to 2 to start counting frames
        if prev_number == 1 and number == 2 and not start_counting:
            start_counting = True
            print("Detected transition from 1 to 2. Starting frame processing.")
            frame_idx = 1

        prev_number = number

        if start_counting:
            frame_name = f"frame_{frame_idx:06d}"
            x, y, width, height = video_data['Frame_coordinates'].values()

            # Ensure that the cropping coordinates do not exceed the frame dimensions
            x_end = min(x + width, frame.shape[1])
            y_end = min(y + height, frame.shape[0])
            cropped_frame = frame[y:y_end, x:x_end]

            # Resize to match dataset size
            resized_frame = cv2.resize(cropped_frame, (1158, 1008))

            # Detect bounding boxes in the resized frame
            bboxes = detect_bounding_boxes(resized_frame, frame_idx, min_area, debug)
            frame_bboxes[frame_name] = bboxes
            frame_idx += 1

    video_capture.release()
    if debug:
        cv2.destroyAllWindows()

    return frame_bboxes

def detect_bounding_boxes(frame, frame_idx, min_area=100, debug=False):
    """
    Detect bounding boxes in the given frame based on a target color.

    Args:
        frame (numpy.ndarray): The frame in which to detect bounding boxes.
        frame_idx (int): Current frame index.
        min_area (int): Minimum area for a bounding box.
        debug (bool): If True, show debug windows.

    Returns:
        list: A list of bounding boxes, each represented as [x1, y1, x2, y2].
    """
    # Define the target color and convert to HSV
    rgb_color = np.array([86, 246, 137])
    rgb_to_hsv = cv2.cvtColor(np.uint8([[rgb_color]]), cv2.COLOR_RGB2HSV)
    hsv_color = rgb_to_hsv[0][0]

    # Define color range in HSV space
    color_interval = np.array([10, 75, 75])
    lower_color = np.clip(hsv_color - color_interval, 0, 255)
    upper_color = np.clip(hsv_color + color_interval, 0, 255)

    # Create a mask to isolate the target regions
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)

    # Apply morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []

    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if approximated contour is a quadrilateral
        if len(approx) == 4:
            is_rectangle = True
            for i in range(4):
                p1 = approx[i][0]
                p2 = approx[(i + 1) % 4][0]
                p3 = approx[(i + 2) % 4][0]

                v1 = p2 - p1
                v2 = p3 - p2

                # Compute angle between sides
                cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

                # Check if angle is close to 90 degrees
                if angle < 70 or angle > 110:
                    is_rectangle = False
                    break

            if is_rectangle:
                x, y, w, h = cv2.boundingRect(approx)
                area = cv2.contourArea(approx)
                aspect_ratio = float(w) / h

                # Filter out rectangles that are too small or too line-like
                if area > min_area and 0.3 < aspect_ratio < 3.0:
                    bboxes.append([x, y, x + w, y + h])

    if debug:
        frame_with_bbox = frame.copy()
        mask_with_bbox = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            print(f"Frame idx: {frame_idx}     Bounding Box: {bbox}")
            cv2.rectangle(frame_with_bbox, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(mask_with_bbox, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow('Frame with Bounding Box', cv2.resize(frame_with_bbox, (640, 480)))
        cv2.imshow('Processed Mask', cv2.resize(mask_with_bbox, (640, 480)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()

    return bboxes

def save_bounding_boxes(output_path, frame_bboxes):
    """
    Save bounding boxes to a text file.

    Args:
        output_path (str): File path to save bounding boxes.
        frame_bboxes (dict): Dictionary mapping frame names to bounding boxes.
    """
    with open(output_path, 'w') as file:
        for frame_name, bboxes in frame_bboxes.items():
            if bboxes:
                bbox = bboxes[0]
                bbox_str = ','.join(map(str, bbox))
                file.write(f"{frame_name} {bbox_str}\n")
            else:
                file.write(f"{frame_name}\n")

def extract_bboxes_at_30fps(frame_bboxes_50fps, frame_names_30fps, fps_video, fps_target):
    """
    Extract bounding boxes corresponding to 30 FPS frames from 50 FPS bounding box data.

    Args:
        frame_bboxes_50fps (dict): Bounding boxes at 50 FPS.
        frame_names_30fps (list): List of frame names at 30 FPS.
        fps_video (int): Original video FPS (50).
        fps_target (int): Target FPS (30).

    Returns:
        dict: A dictionary mapping 30 FPS frame names to their bounding boxes.
    """
    frame_bboxes_30fps = {}
    frame_bboxes_50fps_keys = list(frame_bboxes_50fps.keys())

    for idx_30fps, frame_name_30fps in enumerate(frame_names_30fps):
        frame_idx_30fps = idx_30fps + 1
        frame_idx_50fps = ((frame_idx_30fps - 1) * fps_video) // fps_target + 1
        idx_50fps_in_list = frame_idx_50fps - 1

        if idx_50fps_in_list >= len(frame_bboxes_50fps_keys):
            print(f"Warning: Frame index {frame_idx_50fps} exceeds the number of frames at 50 fps.")
            break

        frame_name_50fps = frame_bboxes_50fps_keys[idx_50fps_in_list]
        bboxes = frame_bboxes_50fps[frame_name_50fps]
        frame_bboxes_30fps[frame_name_30fps] = bboxes

    return frame_bboxes_30fps

def process_case(folder_path, coords_path, min_area, debug, overwrite):
    """
    Process a single case, extracting bounding boxes at 50 FPS and then at 30 FPS.

    Args:
        folder_path (str): Path to the case folder.
        coords_path (str): Path to the coordinates JSON file.
        min_area (int): Minimum area for bounding boxes.
        debug (bool): If True, show debug images.
        overwrite (bool): If True, overwrite existing files.
    """
    video_path, txt_path = find_files(folder_path)
    frame_names = load_frame_names(txt_path)
    video_data = load_frame_coordinates(coords_path)

    last_folder_name = os.path.basename(os.path.normpath(folder_path))
    output_filename_50fps = f"{last_folder_name}_bbox_50fps.txt"
    output_path_50fps = os.path.join(folder_path, output_filename_50fps)

    # If 50 FPS bounding boxes exist and overwrite is False, load them. Otherwise, process video.
    if os.path.exists(output_path_50fps) and not overwrite:
        print(f"Bounding boxes at 50 fps already detected for {last_folder_name}. Skipping 50 fps processing...")
        frame_bboxes_50fps = {}
        with open(output_path_50fps, 'r') as file:
            for line in file:
                parts = line.strip().split()
                frame_name = parts[0]
                if len(parts) > 1:
                    bbox = list(map(int, parts[1].split(',')))
                    frame_bboxes_50fps[frame_name] = [bbox]
                else:
                    frame_bboxes_50fps[frame_name] = []
    else:
        frame_bboxes_50fps = process_video(video_path, video_data, min_area, debug)
        if frame_bboxes_50fps:
            save_bounding_boxes(output_path_50fps, frame_bboxes_50fps)
            print(f"Bounding boxes at 50 fps saved to {output_path_50fps}")
        else:
            print("No bounding boxes to save at 50 fps.")

    # Extract bounding boxes at 30 FPS
    output_filename_30fps = f"{last_folder_name}_bbox_30fps.txt"
    output_path_30fps = os.path.join(folder_path, output_filename_30fps)
    if os.path.exists(output_path_30fps) and not overwrite:
        print(f"Bounding boxes at 30 fps already extracted for {last_folder_name}. Skipping 30 fps extraction...")
    else:
        fps_video = 50
        fps_target = 30
        frame_bboxes_30fps = extract_bboxes_at_30fps(frame_bboxes_50fps, frame_names, fps_video, fps_target)
        if frame_bboxes_30fps:
            with open(output_path_30fps, 'w') as file:
                for frame_name, bboxes in frame_bboxes_30fps.items():
                    if bboxes:
                        bbox = bboxes[0]
                        bbox_str = ','.join(map(str, bbox))
                        file.write(f"{frame_name} {bbox_str}\n")
                    else:
                        file.write(f"{frame_name}\n")
            print(f"Bounding boxes at 30 fps saved to {output_path_30fps}")
        else:
            print("No bounding boxes to save at 30 fps.")

def main():
    """
    Main entry point for the script. Processes single or multiple cases depending on arguments.
    """
    args = parse_arguments()

    if args.all_cases:
        base_folders = ['video_output_fullDs/Positive', 'video_output_fullDs/Negative']
        for base_folder in base_folders:
            case_folders = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
            case_folders = sorted(case_folders, key=lambda x: int(re.search(r'\d+', x).group()))
            for case_folder in case_folders:
                folder_path = os.path.join(base_folder, case_folder)
                process_case(folder_path, args.coords_path, args.min_area, args.debug, overwrite=args.overwrite)
    else:
        process_case(args.folder_path, args.coords_path, args.min_area, args.debug, overwrite=args.overwrite)

if __name__ == '__main__':
    main()
