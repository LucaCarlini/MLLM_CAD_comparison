import os
import base64
import requests
import json
import argparse
import time
import cv2

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Analyze video files using OpenAI GPT model.')
    parser.add_argument('--category', type=str, choices=['Positive', 'Negative'], required=True, help='Choose between Positive or Negative videos.')
    parser.add_argument('--case', type=str, help='Specify a single case number to analyze. If not provided, the entire dataset will be analyzed.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing video summary files.')
    parser.add_argument('--limit_30s', action='store_true', help='Limit extraction to the first 30 seconds of the video.')
    parser.add_argument('--only_5videos', action='store_true', help='Limit extraction to the first 5 videos of the category.')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature for the GPT model.')
    parser.add_argument('--api_key', type=str, required=True, help='OpenAI API key.')
    return parser.parse_args()

def extract_frames(video_path, output_folder):
    """Extract 1 frame per second from the video, starting from second 1 and including the last second."""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count // fps


    os.makedirs(output_folder, exist_ok=True)
    frame_paths = []
    print(f"Video duration: {duration} seconds")
    frame_number = 0
    for sec in range(0, duration + 1):  # Start from 1 to skip frame 0 and include the last second
        cap.set(cv2.CAP_PROP_POS_FRAMES, sec * fps)
        ret, frame = cap.read()
        frame_number = frame_number + 1
        if ret:
            frame_path = os.path.join(output_folder, f"frame_{sec:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            #print(f"Extracted frame {frame_number} at second: {sec}")
        else:
            break

    cap.release()
    return frame_paths

def encode_image(image_path):
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def main(args):
    api_key = args.api_key
    model = "gpt-4o-mini"
    temperature = args.temperature

    dataset_folder = r"C:\Users\lcarl\Desktop\GPT_4_colon\video_output_fullDs"
    category_folder = os.path.join(dataset_folder, args.category)

    if args.case:
        cases = [args.case]
    else:
        cases = [d for d in os.listdir(category_folder) if os.path.isdir(os.path.join(category_folder, d))]

    cases = sorted(cases, key=lambda x: int(x[4:]))

    for case_name in cases:

        if args.only_5videos:
            if int(case_name[4:]) > 5:
                break

        case_folder = os.path.join(category_folder, case_name)

        if args.limit_30s:
            video_file_name = os.path.join(case_folder, f"{case_name}_video30seconds.mp4")
        else:
            video_file_name = os.path.join(case_folder, f"{case_name}_video30fps_noborders.mp4")

        if not os.path.isfile(video_file_name):
            print(f"Video file {video_file_name} not found.")
            continue

        output_txt_filename = os.path.join(case_folder, f"video_summary_{model}_{case_name}.txt")
        output_json_filename = os.path.join(case_folder, f"video_summary_response_{model}_{case_name}.json")

        if not args.overwrite and (os.path.isfile(output_txt_filename) or os.path.isfile(output_json_filename)):
            print(f"Summary files for {case_name} already exist. Skipping...")
            continue

        print(f"Extracting frames from {video_file_name}...")
        frames_folder = os.path.join(case_folder, "frames")
        frame_paths = extract_frames(video_file_name, frames_folder)

        print(f"\nNumber of frames extracted: {len(frame_paths)}\n")
        frame_number = len(frame_paths)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        system_prompt = """You are an expert endoscopist reviewing each frame of a colonoscopic video. For every individual frame, provide the following details in this exact format:
                Frame number: [Number]
                Lesion detected: [True or False]
                Histology lesion type: [One of: "Adenoma", "Hyperplastic", "Serrated", "Invasive", "None"] (provide only if a lesion is detected; use "None" if no lesion is detected)
                Findings: [Description] (if no lesion detected, state "No lesion detected")

                Guidelines:
                1. Each frame is reviewed independently.
                2. The response for each frame must be concise and strictly follow the specified format.
                3. Avoid any additional commentary or follow-up beyond what is requested.
                4. Report frames in the exact order they appear in the video."""

        user_prompt = f"Analyze these {frame_number} colonoscopic video frames."




        base64_frames = [encode_image(frame_path) for frame_path in frame_paths]

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        user_prompt,
                        *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail":"high"}}, base64_frames),
                    ]

                },
            ],
            "temperature": temperature,
        }
        print("GPT analysis in progress...")
        start_time = time.time()
        response = requests.post("https://api.openai.com/v1/chat/completions",
                                 headers=headers,
                                 json=payload)
        end_time = time.time()
        time_interval = end_time - start_time
        result = response.json()

        response_data = {"responses": result, "time_taken": time_interval, "video_frames": frame_number}

        with open(output_json_filename, "w") as file:
            json.dump(response_data, file, indent=4)

        print("Removing extracted frames...")
        for frame_path in frame_paths:
            os.remove(frame_path)
        os.rmdir(frames_folder)

        print("Saving the response and time to a text file...")
        with open(output_txt_filename, "w") as file:
            file.write(result['choices'][0]['message']['content'].replace("                        ", "").strip())
            file.write("\n")
            file.write(f"Time taken: {time_interval} seconds\n")
            file.write(f"Video duration: {frame_number} frames")

        if not args.case:
            time.sleep(30)

if __name__ == '__main__':
    args = parse_args()
    main(args)