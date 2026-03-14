import json
import random
import os
import re
import cv2  # Requires: pip install opencv-python
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from multiprocessing import Pool
from typing import Dict, List, Any

# Root path for videos
VIDEO_ROOT = "./vsi-bench/scannet"
OG_FPS = 24.0
NEW_FPS = 2.0

def get_video_duration(relative_path: str) -> float:
    """
    Reads the video file to calculate total duration in seconds.
    Returns -1.0 if video cannot be read.
    """
    full_path = os.path.join(VIDEO_ROOT, relative_path)

    if not os.path.exists(full_path):
        if os.path.exists(full_path + ".mp4"):
            full_path += ".mp4"
        else:
            return -1.0

    try:
        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            return -1.0

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if fps > 0:
            duration = frame_count / fps
        else:
            duration = -1.0

        cap.release()
        return duration
    except Exception as e:
        print(f"Error reading video {full_path}: {e}")
        return -1.0

def frame_to_time(frame_idx: int, total_normalized_frames: int, real_duration: float) -> float:
    """
    Converts a normalized frame index (e.g., 16 out of 32)
    to actual time in seconds based on real video duration.
    """
    if total_normalized_frames == 0: return 0.0
    return (frame_idx / total_normalized_frames) * real_duration

def process_qa(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Transform a single camera_obj_abs_dist QA item into three question formats.
    Converts frame numbers to actual timestamps.
    """
    # 1. Get Video Info
    video_name = item.get("video_path", "").split("/")[-1]
    real_duration = get_video_duration(video_name)
    video_name = video_name.replace(".mp4", "")

    if real_duration < 0:
        return None

    # Randomly choose 2 out of 3 question types
    question_indices = random.sample([0, 1, 2], 2)

    question = item.get('question', "")
    if not question:
        return None

    results = []
    question_text = question
    answer_value = item.get('ground_truth', "")

    # 2. Extract frame numbers
    match = re.search(r'frame (\d+) of (\d+)', question_text)
    if not match:
        return None

    norm_frame_asked = int(match.group(1))
    norm_total_frames = int(match.group(2))

    # 3. Extract object name
    pattern = r"the nearest point of the\s+(.*?)\s+in frame"
    object_match = re.search(pattern, question_text)
    if not object_match:
        return None
    object_name = object_match.group(1)

    postfix = "\nPlease answer the question using a single word or phrase."

    if 0 in question_indices:
        # 1. Past ask Future
        ask_frame_idx = max(1, norm_frame_asked - random.randint(1, 5))

        question_time = frame_to_time(norm_frame_asked, norm_total_frames, real_duration)
        question_time_str = f"{question_time:.2f}s"

        ask_time = frame_to_time(ask_frame_idx, norm_total_frames, real_duration)

        results.append({
            "video_uid": video_name,
            "conversation": [
                {
                    "role": "user",
                    "content": f"What is the approximate distance (in meters) between the camera (or the person filming) and the nearest point of the {object_name} at {question_time_str}? {postfix}",
                    "time": ask_time
                },
                {
                    "role": "assistant",
                    "content": answer_value,
                    "time": question_time_str
                }
            ]
        })


    if 1 in question_indices:
        # 2. Current ask Current
        ask_time = frame_to_time(norm_frame_asked, norm_total_frames, real_duration)

        results.append({
            "video_uid": video_name,
            "conversation": [
                {
                    "role": "user",
                    "content": f"At current time, what is the approximate distance (in meters) between the camera (or the person filming) and the nearest point of the {object_name}? {postfix}",
                    "time": ask_time
                },
                {
                    "role": "assistant",
                    "content": answer_value,
                    "time": ask_time
                }
            ]
        })


    if 2 in question_indices:
        # 3. Future ask Past
        ask_frame_idx = min(norm_total_frames, norm_frame_asked + random.randint(1, 5))

        question_time = frame_to_time(norm_frame_asked, norm_total_frames, real_duration)
        question_time_str = f"{question_time:.2f}s"

        ask_time = frame_to_time(ask_frame_idx, norm_total_frames, real_duration)

        results.append({
            "video_uid": video_name,
            "conversation": [
                {
                    "role": "user",
                    "content": f"Looking back, what is the approximate distance (in meters) between the camera (or the person filming) and the nearest point of the {object_name} at {question_time_str}? {postfix}",
                    "time": ask_time
                },
                {
                    "role": "assistant",
                    "content": answer_value,
                    "time": ask_time
                }
            ]
        })

    return results

def process_dataset_parallel(dataset, question_type: str, output_path: str, num_workers: int = 4):
    """
    Process camera_obj_abs_dist questions in parallel and save to JSON.
    """
    filtered_data = dataset.filter(lambda x: x['question_type'] == question_type)

    data_list = [dict(item) for item in filtered_data]
    print(f"Processing {len(data_list)} items with {num_workers} workers...")

    with Pool(num_workers) as pool:
        results = pool.map(process_qa, data_list)

    final_qa_list = []
    for result in results:
        if result is not None:
            final_qa_list.extend(result)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(final_qa_list, f, indent=2)

    print(f"Processed {len(final_qa_list)} QA pairs. Saved to {output_path}")

# Main execution
if __name__ == "__main__":
    ds = load_dataset("Journey9ni/vstibench")
    ds = ds['test']
    question_type = 'camera_obj_abs_dist'

    output_file = "./datasets/vsi-bench/my_qa/test/camera_obj_abs_dist_reformatted.json"
    process_dataset_parallel(ds, question_type, output_file, num_workers=4)
