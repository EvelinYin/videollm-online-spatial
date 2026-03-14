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
    Transform a single obj_obj_relative_pos QA item into three question formats.
    Replaces frame numbers with actual timestamps.
    """
    
    # 1. Get Video Info
    video_name = item.get("video_path", "").split("/")[-1]
    real_duration = get_video_duration(video_name)
    video_name = video_name.replace(".mp4", "")
    
    # breakpoint()
    
    if real_duration < 0:
        return None

    # Randomly choose 2 out of 3 question types
    question_indices = random.sample([0, 1, 2], 2)
    
    question = item.get('question', "")
    if not question:
        return None
    
    options = item.get('options', [])
    # breakpoint()
    results = []
    question_text = question + "\nOptions: " + "\n".join(options) + "\nAnswer with the option's letter from the given choices directly."
    answer_value = item.get('mc_answer', "")
    
    # 2. Extract frame numbers: "In frame X of Y"
    match = re.search(r'In frame (\d+) of (\d+)', question_text)
    if not match:
        return None
    
    norm_frame_asked = int(match.group(1))
    norm_total_frames = int(match.group(2))
    
    # 3. Extract Options Text
    options_match = re.search(r"Options:.*", question_text, re.DOTALL)
    if not options_match:
        return None
    options_text = options_match.group(0)
    
    # 4. Extract Object Names and Direction
    pattern = r"relative to (\w+), is"
    relative_object_match = re.search(pattern, question_text)
    if not relative_object_match:
        return None
    relative_object_name = relative_object_match.group(1)
    
    pattern = r"is (\w+) to the "
    asked_object_match = re.search(pattern, question_text)
    if not asked_object_match:
        return None
    asked_object_name = asked_object_match.group(1)
    
    # Extract text within square brackets (e.g., "[Left/Right]")
    bracket_pattern = r"\[.*?\]"
    bracket_matches_list = re.findall(bracket_pattern, question_text)
    if not bracket_matches_list:
        return None
    bracket_matches = bracket_matches_list[0]
    
    # 5. Pre-calculate Text String for target time
    target_time_val = frame_to_time(norm_frame_asked, norm_total_frames, real_duration)
    target_time_str = f"{target_time_val:.2f}s"

    if 0 in question_indices:
        # 1. Past ask Future: Ask before target
        ask_frame_idx = max(1, norm_frame_asked - random.randint(1, 5))
        ask_time = frame_to_time(ask_frame_idx, norm_total_frames, real_duration)
        
        results.append({
            "video_uid": video_name,
            "conversation": [
                {
                    "role": "user",
                    # Changed: "In future frame X..." -> "At Xs..."
                    "content": f"At {target_time_str}, will {asked_object_name} be to the {bracket_matches} relative to {relative_object_name}?\n{options_text}",
                    "time": ask_time
                },
                {
                    "role": "assistant",
                    "content": answer_value,
                    "time": target_time_val
                }
            ]
        })
    
    if 1 in question_indices:
        # 2. Current ask Current
        results.append({
            "video_uid": video_name,
            "conversation": [
                {
                    "role": "user",
                    # Changed: "In current frame..." -> "At current time..."
                    "content": f"At current time, relative to {relative_object_name}, is {asked_object_name} to the {bracket_matches}?\n{options_text}",
                    "time": target_time_val
                },
                {
                    "role": "assistant",
                    "content": answer_value,
                    "time": target_time_val
                }
            ]
        })
    
    if 2 in question_indices:
        # 3. Future ask Past: Ask after target
        ask_frame_idx = min(norm_total_frames, norm_frame_asked + random.randint(1, 5))
        ask_time = frame_to_time(ask_frame_idx, norm_total_frames, real_duration)

        results.append({
            "video_uid": video_name,
            "conversation": [
                {
                    "role": "user",
                    # Changed: "In frame X..." -> "at Xs..."
                    "content": f"Looking back, at {target_time_str}, relative to {relative_object_name}, is {asked_object_name} to the {bracket_matches}?\n{options_text}",
                    "time": ask_time
                },
                {
                    "role": "assistant",
                    "content": answer_value,
                    "time": ask_time # Assistant replies at the asking time
                }
            ]
        })
    
    return results

def process_dataset_parallel(dataset, question_type_list: list, output_path: str, num_workers: int = 4):
    """
    Process obj_obj_relative_pos questions in parallel and save to JSON.
    """
    # Filter for question types in the list
    filtered_data = dataset.filter(lambda x: x['question_type'] in question_type_list)
    
    # Convert dataset to list of dictionaries for parallel processing
    data_list = [dict(item) for item in filtered_data]
    
    print(f"Processing {len(data_list)} items with {num_workers} workers...")
    
    # results = process_qa(item=data_list[0])
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
    # Load your dataset
    ds = load_dataset("Journey9ni/vstibench")
    ds = ds['test'] 
    
    question_type = ['obj_obj_relative_pos_lr', 'obj_obj_relative_pos_nf', 'obj_obj_relative_pos_ud']
    
    # Process and save
    output_file = "./datasets/vsi-bench/my_qa/test/obj_obj_relative_pos_reformatted.json"
    process_dataset_parallel(ds, question_type, output_file, num_workers=4)