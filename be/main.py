from fastapi import FastAPI, HTTPException, Query, Request
from services.retrieval import *
from pydantic import BaseModel
from typing import List
from fastapi import UploadFile, File
import os
from fastapi.middleware.cors import CORSMiddleware
import re
from fastapi.responses import FileResponse
from utils.id_utils import get_cam_ids_for_person
from services.chatbot import LlavaNextVideoInference 
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

class ChatRequest(BaseModel):
    video_paths: List[str]
    question: str
    
class ChatResponse(BaseModel):
    answer: str
    chat_history: List[dict]

inference = None

faiss_engine = Retrieval(device)

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}

@app.post("/search/", response_model=List[SearchResult])
def search(request: SearchRequest):
    result = faiss_engine.search(request.text)
    return result


@app.post("/upload-video")
async def upload_videos(files: List[UploadFile] = File(...)):
    save_path = "static/videos/unprocessed"
    os.makedirs(save_path, exist_ok=True)

    uploaded_files = []
    for file in files:
        file_location = os.path.join(save_path, file.filename)
        with open(file_location, "wb") as f:
            f.write(await file.read())
        uploaded_files.append(file.filename)

    return {"message": "Videos uploaded successfully!", "files": uploaded_files}

BASE_VIDEO_DIR = "/kaggle/input/video-matching/matching_videos (1)/kaggle/working/output_video"

@app.get("/camera-ids")
def list_camera_ids(request: Request):
    if not os.path.exists(BASE_VIDEO_DIR):
        raise HTTPException(status_code=404, detail="Base directory not found.")
    
    camera_folders = [
        folder for folder in os.listdir(BASE_VIDEO_DIR)
        if os.path.isdir(os.path.join(BASE_VIDEO_DIR, folder)) and re.match(r"camera_\d{4}", folder)
    ]

    cam_ids = [folder.split("_")[1] for folder in camera_folders]
    sorted_cam_ids = sorted(cam_ids, key=lambda x: int(x))

    videos_info = []
    for cam_id in sorted_cam_ids:
        video_path = os.path.join(BASE_VIDEO_DIR, f"camera_{cam_id}", "annotated_video_h264.mp4")
        if os.path.isfile(video_path):
            video_url = request.url_for("get_video_by_cam_id", cam_id=cam_id)
            videos_info.append({"cam_id": cam_id, "url": video_url})

    return {"videos": videos_info}


@app.get("/video/{cam_id}")
def get_video_by_cam_id(cam_id: str):
    folder_name = f"camera_{cam_id}"
    video_path = os.path.join(BASE_VIDEO_DIR, folder_name, "annotated_video_h264.mp4")

    if not os.path.isfile(video_path):
        raise HTTPException(status_code=404, detail="Video file not found for this camera ID.")

    return FileResponse(video_path, media_type="video/mp4", filename=f"{cam_id}.mp4")

@app.get("/cam_ids", response_model=List[int])
async def get_cam_ids(person_id: int = Query(..., description="The person ID to search for")):
    file_path = '/kaggle/input/binomic/global_detection.txt'  # Path to your file
    cam_id_list = get_cam_ids_for_person(file_path, person_id)
    return cam_id_list
@app.post("/start_chat")
async def start_chat():
    global inference
    try:
        inference = LlavaNextVideoInference(video_paths=['input/video-matching/matching_videos (1)/kaggle/working/output_video/camera_0342/annotated_video_h264.mp4',
                                                         'input/video-matching/matching_videos (1)/kaggle/working/output_video/camera_0343/annotated_video_h264.mp4',
                                                         'input/video-matching/matching_videos (1)/kaggle/working/output_video/camera_0344/annotated_video_h264.mp4',
                                                         'input/video-matching/matching_videos (1)/kaggle/working/output_video/camera_0345/annotated_video_h264.mp4',])
        response = inference.infer(request.question)
        return ChatResponse(answer=response, chat_history=inference.get_chat_history())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for continued conversation
@app.post("/chat")
async def chat(request: ChatRequest):
    if not inference:
        raise HTTPException(status_code=400, detail="Please start a conversation first using /start_chat")

    try:
        # Use the question from the user and pass the chat history from the inference object
        response = inference.infer(request.question)
        return ChatResponse(answer=response, chat_history=inference.get_chat_history())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))