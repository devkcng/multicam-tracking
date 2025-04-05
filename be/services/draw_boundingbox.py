import subprocess
import cv2
import pandas as pd
import os

def draw_detections(video_path, detections_df, cam_id, output_video=None, max_frames=9000, person_id=None):
    """Draw bounding boxes on video frames for the given cam_id and optional person_id."""
    cap = cv2.VideoCapture(video_path)  # Removed CAP_FFMPEG flag
    if not cap.isOpened():
        print(f"⚠️ Video open failed: {video_path}")
        return

    # Filter detections for this camera ID and optional person_id
    df = detections_df[detections_df["cam_id"] == cam_id].copy()

    if person_id is not None:
        df = df[df['person_id'] == person_id]  # Filter by person_id

    # Compute x2, y2
    df['x2'] = df['x1'] + df['w']
    df['y2'] = df['y1'] + df['h']

    # Video writer setup
    writer = None
    if output_video:
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Changed codec to 'mp4v'
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
        if not writer.isOpened():  # Check if writer is properly opened
            print(f"⚠️ Failed to open VideoWriter for {output_video}")
            writer = None

    frame_count = 0
    batch_size = 3000  # Process in chunks to manage memory

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Get detections for the current frame (0-based)
        detections = df[df["frame_id"] == frame_count]

        # Draw bounding boxes
        for _, det in detections.iterrows():
            x1, y1, x2, y2 = map(int, det[['x1', 'y1', 'x2', 'y2']])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw person_id
            label = f"ID {det['person_id']}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Write frame
        if writer is not None:
            writer.write(frame)

        frame_count += 1
        
        if frame_count % batch_size == 0:
            print(f"⏩ Processed {frame_count} frames for Camera {cam_id}")
            
    cap.release()
    if writer is not None:
        writer.release()
    print(f"✅ Completed: {output_video or 'No output'} for Camera {cam_id}")



def draw_matching(id: int):
    video_urls = []
    camera_names = []
    # Load the full ground truth file once
    detections_path = "/kaggle/input/binomic/global_detection.txt"
    columns = ['cam_id', 'person_id', 'frame_id', 'x1', 'y1', 'w', 'h', 'vx', 'vy']
    df = pd.read_csv(detections_path, delim_whitespace=True, header=None, names=columns)

    # Convert frame_id from 1-based to 0-based
    df['frame_id'] = df['frame_id'] - 1

    # Iterate over a range of camera IDs and process each camera
    for cam_id in range(342, 351):
        camera_name = f"camera_{cam_id:04d}"
        video_path = f"/kaggle/input/dataset-acc2024/scene_039/{camera_name}/video.mp4"
        output_video = f"/kaggle/working/multicam-tracking/be/static/matching_id/{camera_name}/annotated_video.mp4"

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_video)
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

        # Call draw_detections with the specified person_id
        draw_detections(video_path, df, cam_id, output_video, person_id=id)

        # Check if the video file was created
        if os.path.exists(output_video):
            print(f"✅ Successfully created {output_video}")
        else:
            print(f"⚠️ Failed to create {output_video}")

        # Convert to H.264 format using ffmpeg
        input_video = output_video
        output_video_h264 = f"/kaggle/working/multicam-tracking/be/static/matching_id/{camera_name}/annotated_video_h264.mp4"
        
        command = [
            'ffmpeg', '-i', input_video, 
            '-vcodec', 'libx264', 
            '-pix_fmt', 'yuv420p', 
            '-profile:v', 'baseline', 
            '-level', '3.0', 
            '-movflags', '+faststart', 
            output_video_h264
        ]
        
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during video conversion for camera {camera_name}: {e}")
            continue

        # Add the URL of the processed video
        video_url = f"/static/matching_id/{camera_name}/annotated_video_h264.mp4"
        video_urls.append(video_url)
        camera_names.append(camera_name)
    return video_urls, camera_names
