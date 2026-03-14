import json
import random
import os
import re
import cv2  # Requires: pip install opencv-python
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from multiprocessing import Pool
from typing import Dict, List, Any
from functools import partial

# Root path for videos
VIDEO_ROOT = "./datasets/vlm3r_videos"
FPS = 24.0
SEED = 42  # Set to None to disable reproducibility, or change to any integer for different reproducible sequences

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

def process_qa(item: Dict[str, Any], seed: int = None) -> List[Dict[str, Any]]:
    """
    Transform a single camera_obj_abs_dist QA item into three question formats.
    Converts frame numbers to actual timestamps.

    Args:
        item: The QA item to process
        seed: Optional seed for reproducibility. If provided, randomness is seeded based on the item.
    """

    # Set seed for reproducibility if provided
    if seed is not None:
        # Use seed + hash of scene_name to ensure different items get different random sequences
        scene_name = item.get('scene_name', '')
        item_seed = seed + hash(scene_name) % (2**31)
        random.seed(item_seed)

    # 1. Get Video Duration
    dataset_name = item.get('data_source', '')
    scene_rel_path = item.get("scene_name", "") + ".mp4"
    video_rel_path = os.path.join(dataset_name, "videos", scene_rel_path)
    real_duration = get_video_duration(video_rel_path)

    if real_duration < 0:
        return None

    # Randomly choose 1 out of 3 question types
    question_indices = random.sample([0, 1, 2], 1)
    
    conversations = item.get('conversations', [])
    if not conversations or len(conversations) < 2:
        return None

    results = []
    question_text = conversations[0]['value']
    answer_value = conversations[1]['value']
    
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
        # 1. Past ask Future (offset logic preserved)
        ask_frame_idx = max(1, norm_frame_asked - random.randint(1, 5))
        
        # Calculate time
        question_time = frame_to_time(norm_frame_asked, norm_total_frames, real_duration)
        question_time_str = f"{question_time:.2f}s"
        
        ask_time = frame_to_time(ask_frame_idx, norm_total_frames, real_duration)
        ask_time_str = f"{ask_time:.2f}s"
        
        results.append({
            "video_uid": item.get('scene_name', ''),
            "conversation": [
                {
                    "role": "user",
                    # Changed: "...in frame X..." -> "...at Xs..."
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
            "video_uid": item.get('scene_name', ''),
            "conversation": [
                {
                    "role": "user",
                    # Changed: "At current frame..." -> "At current time..."
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
        # 3. Future ask Past (offset logic preserved)
        ask_frame_idx = min(norm_total_frames, norm_frame_asked + random.randint(1, 5))
        
        # Calculate time
        question_time = frame_to_time(norm_frame_asked, norm_total_frames, real_duration)
        question_time_str = f"{question_time:.2f}s"
        
        ask_time = frame_to_time(ask_frame_idx, norm_total_frames, real_duration)
        ask_time_str = f"{ask_time:.2f}s"

        results.append({
            "video_uid": item.get('scene_name', ''),
            "conversation": [
                {
                    "role": "user",
                    # Changed: "...in frame X..." -> "...at Xs..."
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

def process_dataset_parallel(dataset, question_type: str, output_path: str, num_workers: int = 4, seed: int = None):
    """
    Process camera_obj_abs_dist questions in parallel and save to JSON.

    Args:
        dataset: The dataset to process
        question_type: The type of question to filter for
        output_path: Path to save the output JSON
        num_workers: Number of parallel workers
        seed: Optional seed for reproducibility
    """
    filtered_data = dataset.filter(lambda x: x['question_type'] == question_type)

    data_list = [dict(item) for item in filtered_data]
    print(f"Processing {len(data_list)} items with {num_workers} workers...")

    with Pool(num_workers) as pool:
        if seed is not None:
            # Use partial to pass seed to each worker
            process_fn = partial(process_qa, seed=seed)
            results = pool.map(process_fn, data_list)
        else:
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
    ds = load_dataset("Journey9ni/VLM-3R-DATA")
    ds = ds['train']
    question_type = 'camera_obj_abs_dist'

    output_file = "./datasets/vlm3r_ours/my_qa/train/camera_obj_abs_dist_reformatted.json"
    process_dataset_parallel(ds, question_type, output_file, num_workers=4, seed=SEED)