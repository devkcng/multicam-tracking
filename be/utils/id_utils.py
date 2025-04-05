from typing import List

def get_unique_cam_ids(file_path: str) -> List[int]:
    cam_ids = set()
    with open(file_path, 'r') as file:
        for line in file:
            data = line.strip().split()
            cam_id = int(data[0])  # Extracting the cam_id
            cam_ids.add(cam_id)
    
    return sorted(list(cam_ids))