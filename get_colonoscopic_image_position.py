import cv2
import numpy as np
import os
import json
import argparse

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process video frames and detect bounding boxes.")
    parser.add_argument("--video_path", type=str, default="video_output_fullDs/Positive/case10/CAD_PC10.mp4",
                        help="Path to the video file.")
    return parser.parse_args()

def detect_colonoscopy_area(frame):
    """
    Detects the colonoscopy area in a frame by using color segmentation to find pink and red regions.

    Args:
        frame (numpy.ndarray): The input frame from the video.

    Returns:
        colonoscopy_mask (numpy.ndarray): A binary mask where the colonoscopy area is white and the rest is black.
        colonoscopy_area (numpy.ndarray): The extracted colonoscopy area from the frame.
    """
    # Convert the frame to the HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for pink and red hues in HSV
    # Note: Red color wraps around the hue values at 0 and 180 in HSV, so we need two ranges for red
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Define the range for pink hues
    lower_pink = np.array([140, 50, 50])
    upper_pink = np.array([170, 255, 255])

    # Create masks for red and pink areas
    mask_red1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    mask_pink = cv2.inRange(hsv_frame, lower_pink, upper_pink)

    # Combine the masks
    combined_mask = cv2.bitwise_or(mask_red1, mask_red2)
    combined_mask = cv2.bitwise_or(combined_mask, mask_pink)

    # Apply morphological operations to remove small noise and fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

    # Find contours from the mask
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found.")
        return None, None

    # Find the largest contour assuming it's the colonoscopy area
    largest_contour = max(contours, key=cv2.contourArea)

    # Create a mask for the largest contour
    colonoscopy_mask = np.zeros_like(cleaned_mask)
    cv2.drawContours(colonoscopy_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    # Extract the colonoscopy area using the mask
    colonoscopy_area = cv2.bitwise_and(frame, frame, mask=colonoscopy_mask)

    return colonoscopy_mask, colonoscopy_area

def main(args):
    """
    Main function to process a video and detect the colonoscopy area in the first frame.
    """
    # Load the video
    video_path = args.video_path
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read the first frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read the frame.")
        cap.release()
        return

    # get video fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")


    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    print(f"Frame Resolution: {frame_width} x {frame_height}")

    # Detect the colonoscopy area
    colonoscopy_mask, colonoscopy_area = detect_colonoscopy_area(frame)

    if colonoscopy_mask is None or colonoscopy_area is None:
        print("Failed to detect colonoscopy area.")
        cap.release()
        return

    # Optional: Get bounding box coordinates
    contours, _ = cv2.findContours(colonoscopy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    print(f"Bounding Box Coordinates (x, y, w, h): ({x}, {y}, {w}, {h})")

    # save these coordinates to a json in the folder video_output_fullDs after checking if the folder exists
    os.makedirs("video_output_fullDs", exist_ok=True)
    with open("video_output_fullDs/frame_coordinates.json", "w") as json_file:
        json.dump({
            "Frame_resolution": {"Width": frame_width, "Height": frame_height},
            "Frame_coordinates": {"x": x, "y": y, "w": w, "h": h},
            "fps": fps
        }, json_file)

        # Draw the bounding box on the frame for visualization
    frame_with_bbox = frame.copy()
    cv2.rectangle(frame_with_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the results
    cv2.imshow('Original Frame', cv2.resize(frame, (640, 360)))
    cv2.imshow('Colonoscopy Mask', cv2.resize(colonoscopy_mask, (640, 360)))
    cv2.imshow('Colonoscopy Area', cv2.resize(colonoscopy_area, (640, 360)))
    cv2.imshow('Frame with Bounding Box', cv2.resize(frame_with_bbox, (640, 360)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Release the video capture object
    cap.release()

if __name__ == "__main__":
    args = parse_args()
    main(args)
