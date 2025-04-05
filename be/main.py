from fastapi import FastAPI
from services.retrieval import *
from pydantic import BaseModel
from typing import List
from fastapi import UploadFile, File
import os
from fastapi.middleware.cors import CORSMiddleware

device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()

origins = [
    "http://localhost:5173",  # Replace with your frontend URL
    "https://your-frontend-domain.com"  # Add other allowed origins if needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    text: str


class SearchResult(BaseModel):
    camera_id: str
    frame: int
    time_seconds: float

faiss_engine = Retrieval(device)

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.post("/search/", response_model=List[SearchResult])
def search(request: SearchRequest):
    result = faiss_engine.search(request.text)
    return result


@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    save_path = "static/videos/unprocessed"
    os.makedirs(save_path, exist_ok=True)
    file_location = os.path.join(save_path, file.filename)
    with open(file_location, "wb") as f:
        f.write(await file.read())
    return {"message": f"Video {file.filename} uploaded successfully!"}

def list_videos():
    processed_path = "static/videos/processed"
    os.makedirs(processed_path, exist_ok=True)
    video_list = os.listdir(processed_path)
    return {"video_list": video_list}

@app.get("/processed-videos/")
def get_processed_videos():
    return list_videos()

# @app.get("/download-video/{video_name}")
# def download_video(video_name: str):
#     processed_path = "static/videos/processed"
#     file_path = os.path.join(processed_path, video_name)
#     if os.path.exists(file_path) and video_name.endswith(".mp4"):
#         return {"download_link": f"/static/videos/processed/{video_name}": 
#          }
#     else:
#         return {"error": "Video not found or invalid file type"}
@app.get("/get_video_info")
def get_video_info():
    # Full input path (from your example)
    full_path = "/kaggle/input/video-matching/matching_videos (1)/kaggle/working/output_video/camera_0342/annotated_video_h264.mp4"
    
    # Extract camera ID (0342) from the path
    cam_id = os.path.basename(os.path.dirname(full_path))  # camera_0342
    cam_id = cam_id.replace("camera_", "")  # "0342"
    
    # Return a path relative to your frontend (you can map this with nginx/static route etc.)
    relative_path = f"/video/camera_{cam_id}/annotated_video_h264.mp4"
    
    return JSONResponse(content={
        "cam_id": cam_id,
        "path": relative_path
    })