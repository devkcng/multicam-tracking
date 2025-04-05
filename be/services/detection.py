import os
import cv2
from glob import glob
from ultralytics import YOLO
import numpy as np
import torch
from IPython.display import clear_output   #only jupyter notebook

device = 'cuda' if torch.cuda.is_available() else 'cpu'

scene_path = "/kaggle/input/dataset-acc2024/scene_039/"
output_text_dir = "output_text/"
os.makedirs(output_text_dir, exist_ok=True)

# Load YOLOv11 model
model = YOLO('/kaggle/input/yolov11l/pytorch/default/1/yolo11l.pt').to(device)
class_names = model.model.names  

camera_folders = glob(os.path.join(scene_path, "camera_*"))
print("Found camera folders:", camera_folders)  

def track_persons(detections, person_ids, frame_id):
    person_detections = [d for d in detections if d[5] == 0]  
    tracked = []

    for det in person_detections:
        x1, y1, x2, y2, score, _, _ = det 
        person_id = None
        
        for pid, last_det in person_ids.items():
            last_x1, last_y1, last_x2, last_y2 = last_det[:4]
            iou = compute_iou((x1, y1, x2, y2), (last_x1, last_y1, last_x2, last_y2))
            if iou > 0.5:
                person_id = pid
                break
        
        if person_id is None:
            person_id = len(person_ids) + 1
        
        tracked.append((person_id, x1, y1, x2, y2, score))
        person_ids[person_id] = (x1, y1, x2, y2, score)
    
    return tracked, person_ids
    
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    xx1, yy1, xx2, yy2 = box2

    inter_x1 = max(x1, xx1)
    inter_y1 = max(y1, yy1)
    inter_x2 = min(x2, xx2)
    inter_y2 = min(y2, yy2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (xx2 - xx1) * (yy2 - yy1)

    iou = inter_area / (area1 + area2 - inter_area)
    return iou

count_folder = 0
count_frame = []
for cam_folder in camera_folders:
    video_path = os.path.join(cam_folder, "video.mp4")
    if not os.path.exists(video_path):
        continue
    
    cam_id = os.path.basename(cam_folder).split('_')[1]
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        continue  # Skip unopenable videos
    
    frame_count = 0  # Reset for each camera
    cam_name = os.path.basename(cam_folder)
    cam_text_output_dir = os.path.join(output_text_dir, cam_name)
    os.makedirs(cam_text_output_dir, exist_ok=True)
    
    text_file_path = os.path.join(cam_text_output_dir, "detections.txt")
    person_ids = {}
    count_frame.append(frame_count)
    with open(text_file_path, "w") as f:
        f.write("cam_id,frame_id,person_id,x1,y1,x2,y2,score\n")
        
        while True:
            ret, frame = cap.read()
            if frame_count >= 300:
                break  # Stop at 300 frames or video end
            
            # Process the frame
            results = model(frame)
            clear_output(wait=True)
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    if cls == 0:  # Person class
                        detections.append([x1, y1, x2, y2, conf, cls, frame_count])
            
            tracked_detections, person_ids = track_persons(detections, person_ids, frame_count)
            
            # Write data with current frame_count (0–299)
            for det in tracked_detections:
                person_id, x1, y1, x2, y2, score = det
                f.write(f"{cam_id},{frame_count},{person_id},{x1},{y1},{x2},{y2},{score:.2f}\n")
            
            frame_count += 1  # Increment AFTER processing
    count_frame.append(frame_count)
    count_folder +=1        
    cap.release()
print(count_folder)
print("Person ID tracking and bounding box extraction completed for all cameras!")
