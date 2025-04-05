from typing import List

def get_cam_ids_for_person(file_path: str, person_id: int) -> List[int]:
    cam_ids = set()
    with open(file_path, 'r') as file:
        # Skip the header row
        next(file)
        for line in file:
            data = line.strip().split()
            current_person_id = int(data[1])  # Assuming person_id is the second column
            if current_person_id == person_id:
                cam_id = int(data[0])  # Extracting the cam_id
                cam_ids.add(cam_id)
    
    return sorted(list(cam_ids))