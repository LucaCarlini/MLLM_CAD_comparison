import os
import cv2
import argparse
import numpy as np
import subprocess
import json

def parse_args():
    """
    Parse command-line arguments for dataset processing and video generation.
    """
    parser = argparse.ArgumentParser(description='Process a dataset and generate videos with optional bounding boxes.')
    parser.add_argument(
        '--dataset_folder',
        type=str,
        default=r'D:\SUNdatabase',
        help='Path to the root dataset folder.'
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        default='video_output_fullDs',
        help='Path to save the output video files.'
    )
    parser.add_argument(
        '--draw_bboxes',
        action='store_true',
        help='Draw bounding boxes on the images to test the bbox extraction algorithm.'
    )
    return parser.parse_args()

def load_images_from_folder(folder):
    """
    Load images from the specified folder.

    Args:
        folder (str): Path to the folder containing images.

    Returns:
        tuple: (images, filenames)
            images (list): List of loaded images as NumPy arrays.
            filenames (list): List of corresponding image filenames.
    """
    images = []
    filenames = []
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(supported_extensions):
            img_path = os.path.join(folder, filename)
            frame = cv2.imread(img_path)
            if frame is not None:
                images.append(frame)
                filenames.append(filename)
            else:
                print(f"Warning: Could not read image {img_path}")
    return images, filenames

def read_annotations(annotation_file):
    """
    Read annotations from the annotation file.

    The annotation file should have lines in the format:
    filename x1,y1,x2,y2,class_label

    Args:
        annotation_file (str): Path to the annotation text file.

    Returns:
        dict: Dictionary with keys as filenames and values as dicts containing:
              'bbox': (x1, y1, x2, y2)
              'class': class_label
    """
    annotations = {}
    with open(annotation_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split()
                filename = parts[0]
                bbox_and_class = parts[1]
                bbox_values = bbox_and_class.split(',')

                # Expecting (x1, y1, x2, y2, class_label)
                if len(bbox_values) == 5:
                    x1, y1, x2, y2, class_label = map(int, bbox_values)
                    annotations[filename] = {'bbox': (x1, y1, x2, y2), 'class': class_label}
                else:
                    print(f"Invalid annotation format in line: {line}")
    return annotations

def save_image_filenames(filenames, output_folder):
    """
    Save the list of image filenames to a text file.

    Args:
        filenames (list): List of image filenames.
        output_folder (str): Output directory to save the text file.
    """
    txt_filename = os.path.join(output_folder, 'image_filenames.txt')
    with open(txt_filename, 'w') as file:
        for name in filenames:
            file.write(f"{name}\n")
    print(f"Image filenames saved to: {txt_filename}")

def save_image_annotations(filenames, annotations, output_folder):
    """
    Save image annotations to a text file. If no annotations are found for an image, write just the filename.

    Args:
        filenames (list): List of image filenames.
        annotations (dict): Dictionary of annotations for each filename.
        output_folder (str): Output directory to save the text file.
    """
    txt_filename = os.path.join(output_folder, 'image_annotations.txt')
    with open(txt_filename, 'w') as file:
        for name in filenames:
            if name in annotations:
                bbox = annotations[name]['bbox']
                class_label = annotations[name]['class']
                bbox_str = ','.join(map(str, bbox))
                file.write(f"{name} {bbox_str},{class_label}\n")
            else:
                file.write(f"{name}\n")
    print(f"Image annotations saved to: {txt_filename}")

def process_images(images, filenames, annotations, output_folder, coordinates_path,
                   img_width, img_height, pos_x, pos_y,
                   screen_width, screen_height,
                   category, case, draw_bboxes):
    """
    Process images by resizing, placing them within a frame, optionally drawing bounding boxes, and
    adding frame counters, category, and case text. Saves processed frames as .png files.

    Args:
        images (list): List of images as NumPy arrays.
        filenames (list): Corresponding filenames for the images.
        annotations (dict): Annotations dictionary.
        output_folder (str): Directory to save processed frames.
        coordinates_path (str): Path to the JSON file containing frame coordinates.
        img_width (int): Width to resize the image to.
        img_height (int): Height to resize the image to.
        pos_x (int): X position to place the resized image in the frame.
        pos_y (int): Y position to place the resized image in the frame.
        screen_width (int): Width of the final frame.
        screen_height (int): Height of the final frame.
        category (str): Category name (e.g., 'Positive' or 'Negative').
        case (str): Case identifier.
        draw_bboxes (bool): If True, draw bounding boxes on the frames.
    """
    if not images:
        print("No images to process.")
        return

    for idx, (img, filename) in enumerate(zip(images, filenames)):
        original_height, original_width = img.shape[:2]
        scale_x = img_width / original_width
        scale_y = img_height / original_height

        print(f"Processing image {idx + 1}/{len(images)}: {filename}")

        # Resize the image
        try:
            resized_img = cv2.resize(img, (img_width, img_height))
        except cv2.error as e:
            print(f"Error resizing image {idx + 1}: {e}")
            continue

        # Create a blank frame and place the resized image
        frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        frame[pos_y:pos_y + resized_img.shape[0], pos_x:pos_x + resized_img.shape[1]] = resized_img

        # Optionally draw bounding boxes if enabled and annotation exists
        if draw_bboxes and filename in annotations:
            x1, y1, x2, y2 = annotations[filename]['bbox']
            adjusted_x1 = int(x1 * scale_x) + pos_x
            adjusted_y1 = int(y1 * scale_y) + pos_y
            adjusted_x2 = int(x2 * scale_x) + pos_x
            adjusted_y2 = int(y2 * scale_y) + pos_y
            cv2.rectangle(frame, (adjusted_x1, adjusted_y1), (adjusted_x2, adjusted_y2), (0, 0, 255), thickness=2)

        # Add frame counter text
        frame_counter_text = f"{idx + 1}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        color = (0, 255, 0)
        thickness = 3
        text_x, text_y = 50, 50
        cv2.putText(frame, frame_counter_text, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

        # Update coordinates JSON with the position of the text for the first frame
        if idx == 0:
            (text_width, text_height), baseline = cv2.getTextSize(frame_counter_text, font, font_scale, thickness)
            bbox_frame_text = (text_x, text_y - text_height, text_x + text_width, text_y + baseline)
            with open(coordinates_path, 'r+') as file:
                data = json.load(file)
                data['Frame_text_bbox'] = {
                    'x1': bbox_frame_text[0],
                    'y1': bbox_frame_text[1],
                    'x2': bbox_frame_text[2],
                    'y2': bbox_frame_text[3]
                }
                file.seek(0)
                json.dump(data, file, indent=4)
                file.truncate()

        # Add category and case text
        cv2.putText(frame, category, (text_x, text_y + 100), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(frame, case, (text_x, text_y + 200), font, font_scale, color, thickness, cv2.LINE_AA)

        # Save the processed frame
        frame_filename = os.path.join(output_folder, f"frame_{idx:05d}.png")
        cv2.imwrite(frame_filename, frame)

    print("Frames processed and data saved.")

def create_mp4_video(output_folder, output_video_path, input_fps, output_fps):
    """
    Create an MP4 video from the saved frames using FFmpeg.

    Args:
        output_folder (str): Directory containing the processed frames.
        output_video_path (str): Path to save the final MP4 video.
        input_fps (int): Original input frames per second.
        output_fps (int): Desired output frames per second.
    """
    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(input_fps),
        '-i', os.path.join(output_folder, 'frame_%05d.png'),
        '-filter:v', f'fps={output_fps}',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_video_path
    ]
    try:
        subprocess.run(cmd, check=True)
        print(f"Video saved to: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while creating video with FFmpeg: {e}")

def clean_up_frames(output_folder):
    """
    Delete temporary frame images after video creation.

    Args:
        output_folder (str): Directory containing the frame images.
    """
    frame_files = [f for f in os.listdir(output_folder) if f.startswith('frame_') and f.endswith('.png')]
    for f in frame_files:
        os.remove(os.path.join(output_folder, f))
    print("Temporary frame images deleted.")

def load_frame_coordinates(json_path):
    """
    Load frame coordinates from a JSON file.

    Args:
        json_path (str): Path to the JSON file containing frame coordinates.

    Returns:
        dict: Dictionary containing frame coordinates and other metadata.
    """
    with open(json_path, 'r') as file:
        coordinates = json.load(file)
    return coordinates

def main(args):
    # Load frame coordinates
    coordinates = load_frame_coordinates(os.path.join(args.output_folder, 'frame_coordinates.json'))
    img_width = coordinates['Frame_coordinates']['w']
    img_height = coordinates['Frame_coordinates']['h']
    pos_x = coordinates['Frame_coordinates']['x']
    pos_y = coordinates['Frame_coordinates']['y']
    input_fps = 30  # Assuming original FPS is 30
    output_fps = round(coordinates['fps'])
    screen_width = coordinates['Frame_resolution']['Width']
    screen_height = coordinates['Frame_resolution']['Height']

    dataset_folder = args.dataset_folder
    print(f"Processing dataset in folder: {dataset_folder}")

    for category in ['Positive', 'Negative']:
        category_folder = os.path.join(dataset_folder, category)
        if not os.path.isdir(category_folder):
            print(f"Category folder {category_folder} does not exist.")
            continue

        print(f"Processing category: {category}")

        # If category is Positive, annotations are expected
        if category == 'Positive':
            annotation_folder = os.path.join(category_folder, 'annotation_txt')
            if not os.path.isdir(annotation_folder):
                print(f"No annotation folder found in {category_folder}")
                continue

            cases = [
                d for d in os.listdir(category_folder)
                if os.path.isdir(os.path.join(category_folder, d)) and d != 'annotation_txt'
            ]
        else:
            cases = [
                d for d in os.listdir(category_folder)
                if os.path.isdir(os.path.join(category_folder, d))
            ]

        for case in sorted(cases):
            case_folder = os.path.join(category_folder, case)
            print(f"Processing {category}/{case}")

            images, filenames = load_images_from_folder(case_folder)
            if not images:
                print(f"No images found in {case_folder}")
                continue
            print(f"Loaded {len(images)} images from {case_folder}")

            # Load annotations for Positive category
            if category == 'Positive':
                annotation_file = os.path.join(category_folder, 'annotation_txt', f"{case}.txt")
                if os.path.isfile(annotation_file):
                    annotations = read_annotations(annotation_file)
                else:
                    print(f"Annotation file {annotation_file} not found.")
                    annotations = {}
            else:
                annotations = {}

            coordinates_path = os.path.join(args.output_folder, 'frame_coordinates.json')
            output_case_folder = os.path.join(args.output_folder, category, case)
            os.makedirs(output_case_folder, exist_ok=True)

            # Save filenames and annotations
            save_image_filenames(filenames, output_case_folder)
            save_image_annotations(filenames, annotations, output_case_folder)

            print("Processing images and generating video...")
            process_images(
                images,
                filenames,
                annotations,
                output_case_folder,
                coordinates_path,
                img_width=img_width,
                img_height=img_height,
                pos_x=pos_x,
                pos_y=pos_y,
                screen_width=screen_width,
                screen_height=screen_height,
                category=category,
                case=case,
                draw_bboxes=args.draw_bboxes
            )

            output_video_path = os.path.join(output_case_folder, f"{case}_video30fps.mp4")
            create_mp4_video(output_case_folder, output_video_path, input_fps, output_fps)

            # Clean up temporary frames
            clean_up_frames(output_case_folder)

if __name__ == '__main__':
    args = parse_args()
    main(args)
