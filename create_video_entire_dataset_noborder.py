import os
import cv2
import argparse
import subprocess

def parse_args():
    """
    Parse command-line arguments for dataset processing and video generation.
    """
    parser = argparse.ArgumentParser(description='Process a dataset and generate videos.')
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
        '--limit_30s',
        action='store_true',
        help='Limit the video duration to 30 seconds at 30 fps.'
    )
    return parser.parse_args()

def load_images_from_folder(folder, max_images=None):
    """
    Load images from the specified folder.

    Args:
        folder (str): Path to the folder containing images.
        max_images (int, optional): Maximum number of images to load. If None, load all images.

    Returns:
        list: A list of loaded image arrays.
    """
    images = []
    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(supported_extensions):
            img_path = os.path.join(folder, filename)
            frame = cv2.imread(img_path)
            if frame is not None:
                images.append(frame)
                if max_images is not None and len(images) >= max_images:
                    break
            else:
                print(f"Warning: Could not read image {img_path}")
    return images

def create_mp4_video(images, output_video_path, fps=30):
    """
    Create an MP4 video from a list of images using FFmpeg.

    This function temporarily saves the images as individual frames and then uses FFmpeg to
    encode them into an MP4 video.

    Args:
        images (list): List of image arrays.
        output_video_path (str): Path where the MP4 video will be saved.
        fps (int, optional): Frames per second for the output video. Default is 30.
    """
    if not images:
        print("No images provided for video creation.")
        return

    height, width, _ = images[0].shape
    temp_folder = 'temp_frames'
    os.makedirs(temp_folder, exist_ok=True)

    # Write each image to a temporary frame file
    for idx, img in enumerate(images):
        frame_filename = os.path.join(temp_folder, f"frame_{idx:05d}.png")
        cv2.imwrite(frame_filename, img)

    # Construct FFmpeg command
    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(fps),
        '-i', os.path.join(temp_folder, 'frame_%05d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_video_path
    ]

    # Run FFmpeg command
    try:
        subprocess.run(cmd, check=True)
        print(f"Video saved to: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while creating video with FFmpeg: {e}")

    # Clean up temporary frames
    for f in os.listdir(temp_folder):
        os.remove(os.path.join(temp_folder, f))
    os.rmdir(temp_folder)

def main(args):
    """
    Main function to process the dataset and generate videos.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    dataset_folder = args.dataset_folder
    output_folder = args.output_folder
    limit_30s = args.limit_30s

    # Process both Positive and Negative categories
    for category in ['Positive', 'Negative']:
        category_folder = os.path.join(dataset_folder, category)
        if not os.path.isdir(category_folder):
            print(f"Category folder {category_folder} does not exist.")
            continue

        print(f"Processing category: {category}")

        # List all case directories, excluding any with 'annotation' in the name
        cases = [
            d for d in os.listdir(category_folder)
            if os.path.isdir(os.path.join(category_folder, d)) and 'annotation' not in d
        ]

        # Sort cases by the numeric part of the folder name (e.g., "Case100" -> 100)
        cases = sorted(cases, key=lambda x: int(x[4:]))

        for case in cases:
            case_folder = os.path.join(category_folder, case)
            print(f"Processing {category}/{case}")

            # If limit_30s is True, load only the first 900 frames (30 fps * 30s = 900 frames)
            max_frames = 900 if limit_30s else None

            images = load_images_from_folder(case_folder, max_images=max_frames)
            if not images:
                print(f"No images found in {case_folder}")
                continue
            print(f"Loaded {len(images)} images from {case_folder}")

            # Create the output directory for this case
            output_case_folder = os.path.join(output_folder, category, case)
            os.makedirs(output_case_folder, exist_ok=True)

            # Determine the output video filename based on the duration limit
            if limit_30s:
                output_video_path = os.path.join(output_case_folder, f"{case}_video30seconds.mp4")
            else:
                output_video_path = os.path.join(output_case_folder, f"{case}_video_30fps_noborders.mp4")

            # Create the video from the loaded images
            create_mp4_video(images, output_video_path, fps=30)

    print("Processing complete.")

if __name__ == '__main__':
    args = parse_args()
    main(args)
