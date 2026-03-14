import json
import random
import os
import re
import cv2  # Requires: pip install opencv-python
from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from multiprocessing import Pool
from typing import Dict, List, Any
import math

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
        # Fallback: try checking if extension is missing (optional safety)
        if os.path.exists(full_path + ".mp4"):
            full_path += ".mp4"
        else:
            # print(f"Warning: Video file not found: {full_path}")
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
    return (frame_idx  / total_normalized_frames) * real_duration

def process_qa(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Transform a single camera_displacement QA item into three question formats.
    Calculates actual timestamps for BOTH the metadata 'time' field AND the question text.
    """

    # 1. Get Video Info
    video_name = item.get("video_path", "").split("/")[-1]
    real_duration = get_video_duration(video_name)
    video_name = video_name.replace(".mp4", "")
    
    # breakpoint()c

    # Skip if video is invalid
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
    
    # Extract frame numbers from question (e.g., "between frame 14 and frame 20 of 32")
    match = re.search(r'frame (\d+) and frame (\d+) of (\d+)', question_text)
    if not match:
        return None
    
    # Normalized frame indices (e.g., 0 to 32)
    norm_frame_start = int(match.group(1))
    norm_frame_end = int(match.group(2))
    norm_total_frames = int(match.group(3)) # Typically 32
    
    # Calculate textual times (rounded to 2 decimal places for readability)
    text_time_start = frame_to_time(norm_frame_start, norm_total_frames, real_duration)
    text_time_end = frame_to_time(norm_frame_end, norm_total_frames, real_duration)
    
    # Formatting helper for text
    t_start_str = f"{text_time_start:.2f}s"
    t_end_str = f"{text_time_end:.2f}s"
    
    postfix = "\nPlease answer the question using a single word or phrase."
    
    if 0 in question_indices:
        # 1. Past ask Future: Ask at frame_start (±5), about future frame_end
        ask_frame_idx = min(max(1, norm_frame_start + random.randint(-5, 5)), norm_frame_end)
        
        # Calculate timestamps for metadata
        ask_time = frame_to_time(ask_frame_idx, norm_total_frames, real_duration)
        target_time = frame_to_time(norm_frame_end, norm_total_frames, real_duration)
        
        results.append({
            "video_uid": video_name,
            "conversation": [
                {
                    "role": "user",
                    # Changed: frame X -> {t_start_str}
                    "content": f"How far (in meters) will the camera move between {t_start_str} and {t_end_str}? {postfix}",
                    "time": ask_time 
                },
                {
                    "role": "assistant",
                    "content": answer_value,
                    "time": target_time
                }
            ],
        })
    
    if 1 in question_indices:
        # 2. Current ask Current: Ask at frame_end, about frame_end
        ask_time = frame_to_time(norm_frame_end, norm_total_frames, real_duration)
        
        results.append({
            "video_uid": video_name,
            "conversation": [
                {
                    "role": "user",
                    # Changed: frame X -> {t_start_str}
                    "content": f"What is the camera displacement (in meters) from {t_start_str} to {t_end_str}? {postfix}",
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
        # 3. Future ask Past: Ask at frame > frame_end (±5), about past frame_start to frame_end
        ask_frame_idx = min(norm_total_frames, norm_frame_end + random.randint(1, 5))
        
        ask_time = frame_to_time(ask_frame_idx, norm_total_frames, real_duration)
        target_time = frame_to_time(ask_frame_idx, norm_total_frames, real_duration) 

        results.append({
            "video_uid": video_name,
            "conversation": [
                {
                    "role": "user",
                    # Changed: frame X -> {t_start_str}
                    "content": f"Looking back, how far did the camera move between {t_start_str} and {t_end_str} in meters? {postfix}",
                    "time": ask_time
                },
                {
                    "role": "assistant",
                    "content": answer_value,
                    "time": target_time
                }
            ]
        })
        
    return results

def process_dataset_parallel(dataset, question_type: str, output_path: str, num_workers: int = 4):
    """
    Process camera_displacement questions in parallel and save to JSON.
    """
    # Filter for camera_displacement question_type
    filtered_data = dataset.filter(lambda x: x['question_type'] == question_type)
    
    # Convert dataset to list of dictionaries for parallel processing
    data_list = [dict(item) for item in filtered_data]
    print(f"Processing {len(data_list)} items with {num_workers} workers...")
    
    # results = process_qa(data_list[0])
    # breakpoint()
    
    # Parallel processing
    with Pool(num_workers) as pool:
        results = pool.map(process_qa, data_list)
    
    # Flatten results and filter None values
    final_qa_list = []
    for result in results:
        if result is not None:
            final_qa_list.extend(result)
    
    # Save to JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(final_qa_list, f, indent=2)
    
    print(f"Processed {len(final_qa_list)} QA pairs. Saved to {output_path}")

# Main execution
if __name__ == "__main__":
    # ds = load_dataset("Journey9ni/VLM-3R-DATA")
    ds = load_dataset("Journey9ni/vstibench")

    ds = ds['test'] 
    # ds.unique('question_type')
    question_type = 'camera_displacement'
    
    output_file = "./datasets/vsi-bench/my_qa/test/camera_displacement_reformatted.json"
    process_dataset_parallel(ds, question_type, output_file, num_workers=4)
    
    
    # ['camera_obj_rel_dist_v2', 'camera_movement_direction', 'camera_obj_abs_dist', 'camera_obj_rel_dist_v3', 'obj_obj_relative_pos_nf', 'obj_obj_relative_pos_ud', 'camera_obj_rel_dist_v1', 'obj_obj_relative_pos_lr', 'camera_displacement']