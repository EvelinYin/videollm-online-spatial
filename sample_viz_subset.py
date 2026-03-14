import json
import random

def create_subset_json(input_files, output_file, target_uids, count_per_file=5):
    combined_subset = []

    # Convert target list to a set for faster lookup
    allowed_uids = set(target_uids)

    print(f"Processing {len(input_files)} files...")

    for file_path in input_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 1. Filter data based on the allowed video_uids
            valid_entries = [
                item for item in data 
                if item.get("video_uid") in allowed_uids
            ]

            # 2. Randomly select 5 entries (or fewer if not enough exist)
            if len(valid_entries) >= count_per_file:
                selected = random.sample(valid_entries, count_per_file)
            else:
                # If there are fewer than 5 valid entries, take all of them
                print(f"Warning: '{file_path}' only has {len(valid_entries)} valid entries. Taking all.")
                selected = valid_entries

            combined_subset.extend(selected)

        except FileNotFoundError:
            print(f"Error: The file {file_path} was not found.")
        except json.JSONDecodeError:
            print(f"Error: The file {file_path} is not valid JSON.")

    # 3. Write the combined result to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_subset, f, indent=2)

    print(f"Success! {len(combined_subset)} total entries saved to '{output_file}'.")

# --- Configuration ---

# Replace these with your actual file names
# files_to_process = [
#     "/u/hyin2/videollm-online/datasets/vsi-bench/my_qa/test/camera_displacement_reformatted_full.json",
#     "/home/yin178/VLM-3R/data/my_processed/camera_obj_abs_dist_reformatted.json",
#     "/home/yin178/VLM-3R/data/my_processed/camera_obj_rel_dist_reformatted.json",
#     "/home/yin178/VLM-3R/data/my_processed/camera_movement_direction_reformatted.json",
#     "/home/yin178/VLM-3R/data/my_processed/obj_obj_relative_pos_reformatted.json"
# ]


files_to_process = [
    "/u/hyin2/videollm-online/datasets/vsi-bench/my_qa/test/camera_displacement_reformatted.json",
    "/u/hyin2/videollm-online/datasets/vsi-bench/my_qa/test/obj_obj_relative_pos_reformatted.json",
]

# The specific video_uids you want to include
allowed_scenes = [
    "scene0011_00", "scene0011_01", "scene0015_00", 
    "scene0019_00", "scene0019_01", "scene0025_00", 
    "scene0025_01", "scene0025_02", "scene0030_00",
    "scene0030_01", "scene0030_02", "scene0046_00",
    "scene0046_01", "scene0046_02", "scene0050_00",
    "scene0050_01", "scene0050_02"  
]

# Run the function
if __name__ == "__main__":
    create_subset_json(files_to_process, "./test_subset.json", allowed_scenes)