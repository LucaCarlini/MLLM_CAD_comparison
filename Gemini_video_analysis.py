import os
import google.generativeai as genai
import time
import json
import argparse
import cv2

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Analyze video files using Google Generative AI.')
    parser.add_argument('--category', type=str, choices=['Positive', 'Negative'], required=True, help='Choose between Positive or Negative videos.')
    parser.add_argument('--case', type=str, help='Specify a single case number to analyze. If not provided, the entire dataset will be analyzed.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing video summary files.')
    parser.add_argument('--limit_30s', action='store_true', help='Limit extraction to the first 30 seconds of the video.')
    parser.add_argument('--only_5videos', action='store_true', help='Limit extraction to the first 5 videos of the category.')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature for the Gemini model.')
    parser.add_argument('--api_key', type=str, required=True, help='API key for Google Generative AI.')
    return parser.parse_args()

def check_video_file(video_file):
    while video_file.state.name == "PROCESSING":
        print('.', end='')
        time.sleep(10)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        raise ValueError(video_file.state.name)



def main(args):
    genai.configure(api_key=args.api_key)

    temperature = args.temperature
    model_name = "gemini-1.5-pro-002"

    dataset_folder = r"C:\Users\lcarl\Desktop\GPT_4_colon\video_output_fullDs"
    category_folder = os.path.join(dataset_folder, args.category)

    if args.case:
        cases = [args.case]
    else:
        cases = [d for d in os.listdir(category_folder) if os.path.isdir(os.path.join(category_folder, d))]

    cases = sorted(cases, key=lambda x: int(x[4:]))

    for case_name in cases:
        if args.only_5videos and int(case_name[4:]) > 5:
            break

        case_folder = os.path.join(category_folder, case_name)

        if args.limit_30s:
            video_file_name = os.path.join(case_folder, f"{case_name}_video30seconds.mp4")
        else:
            video_file_name = os.path.join(case_folder, f"{case_name}_video30fps_noborders.mp4")


        if not os.path.isfile(video_file_name):
            print(f"Video file {video_file_name} not found.")
            continue

        output_txt_filename = os.path.join(case_folder, f"video_summary_{model_name}_{case_name}.txt")
        output_json_filename = os.path.join(case_folder, f"video_summary_{model_name}_response_{case_name}.json")

        if not args.overwrite and (os.path.isfile(output_txt_filename) or os.path.isfile(output_json_filename)):
            print(f"Summary files for {case_name} already exist. Skipping...")
            continue


        # print exact length of video
        cap = cv2.VideoCapture(video_file_name)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        print(f"Video duration: {duration} seconds")
        cap.release()
        #print(1/0)





        print(f"Uploading file {video_file_name}...")
        video_file = genai.upload_file(path=video_file_name)

        print(f"Completed upload: {video_file.uri}")

        check_video_file(video_file)

        system_prompt = """You are an expert endoscopist reviewing each frame of a colonoscopic video. For every individual frame, provide the following details in this exact format:
                Frame number: [Number]
                Frame time: [MM:SS]
                Lesion detected: [True or False]
                Bounding box coordinates: [x1, y1, x2, y2] (if no lesion detected, use [0, 0, 0, 0])
                Histology lesion type: [One of: "Adenoma", "Hyperplastic", "Serrated", "Invasive", "None"] (provide only if a lesion is detected; use "None" if no lesion is detected)
                Findings: [Description] (if no lesion detected, state "No lesion detected")

                Guidelines:
                1. Each frame is reviewed independently.
                2. The response for each frame must be concise and strictly follow the specified format.
                3. Avoid any additional commentary or follow-up beyond what is requested.
                4. Report frames in the exact order they appear in the video."""

        # system_prompt = """You are an expert endoscopist reviewing each frame of a colonoscopic video. For each frame, provide only the following details in this exact format:
        #         Frame number: [Number]
        #         Bounding box coordinates: [x1, y1, x2, y2] (around any detected lesion)
        #         Findings: A detailed description of the lesion's characteristics. If no lesion is detected, state "No lesion detected" and use bounding box as [0, 0, 0, 0].
        #         Ensure your responses are brief, strictly follow the structure, and do not include any additional commentary, follow-up, or diagnosis beyond the requested details.
        #         Provide the information for each frame in the same order as they appear in the video."""

        user_prompt = "Analyze this colonoscopic video."
        #if args.limit_30s:
        #    user_prompt += " First 30 seconds only."

        model = genai.GenerativeModel(model_name=model_name, system_instruction=system_prompt)
        config = genai.GenerationConfig(
            temperature=temperature,
        )
        print("Making LLM inference request...")
        # record the time it takes to generate the content

        start_time = time.time()

        response = model.generate_content([video_file, user_prompt], request_options={"timeout": 600}, generation_config=config)

        end_time = time.time()

        time_interval = end_time - start_time

        print(response.text)

        print("Saving the response and time to a text file...")
        with open(output_txt_filename, "w") as file:
            file.write(response.text)
            file.write("\n")
            file.write(f"Time taken: {time_interval} seconds\n")
            file.write(f"Video time lenght: {duration} seconds")

        # Save the response and time to a JSON file
        response_data = response.to_dict()
        response_data['time_taken'] = time_interval
        response_data['video_time_length'] = duration
        with open(output_json_filename, "w") as file:
            json.dump(response_data, file, indent=4)






        if not args.case:
            print(f"Waiting for 10 seconds before processing the next case...")
            time.sleep(30)

if __name__ == '__main__':
    args = parse_args()
    main(args)
