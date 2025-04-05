# from utils.processing_text import Translation 
from sentence_transformers import SentenceTransformer
import faiss
import json

import torch
from torch.nn import functional as F
import numpy as np
from easydict import EasyDict as edict

from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from transformers import CLIPProcessor, CLIPTokenizerFast
from transformers import AutoProcessor
from models.CLIP_VIP import CLIPModel, clip_loss
class Retrieval:
    def __init__(self, device, rerank_bin_file: str = None):
            # Ensure that bin_files and modes lists have the same length
            # Initialize re-ranking index if provided
            self.rerank_index = self.load_bin_file("/kaggle/input/binomic/faiss_nomic_cosine.bin") 
            # self.translate = Translation()
            
            self.device = device
            # Initialize SentenceTransformer model
            
            self.model_nomic = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device=self.device)
            # extraCfg = edict({
            #     "type": "ViP",
            #     "temporal_size": 12,
            #     "if_use_temporal_embed": 1,
            #     "logit_scale_init_value": 4.60,
            #     "add_cls_num": 3
            # })

            # clipconfig = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
            # clipconfig.vision_additional_config = extraCfg

            # checkpoint = torch.load("YOUR_PATH_TO/CLIP-ViP/pretrain_clipvip_base_32.pt")
            # cleanDict = { key.replace("clipmodel.", "") : value for key, value in checkpoint.items() }
            # self.model_clipvip =  CLIPModel(config=clipconfig)
            # self.model_clipvip.load_state_dict(cleanDict)
            # self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

    def load_bin_file(self, bin_file: str):
        return faiss.read_index(bin_file)

    def _read_json(self, file_json):
        with open(file_json, "r") as file:
            data = json.load(file)
        return data
    def search(self, text, k=50, fps=30):
        # text = self.translate(text)
        text_features = self.model_nomic.encode(text).reshape(1, -1)
        scores, idx_image = self.rerank_index.search(text_features, k=k)

        camera_start_id = 342
        camera_end_id = 350
        frame_step = 10
        max_frame = 9000
        num_frames_per_camera = (max_frame // frame_step) + 1  # 901

        seen_cameras = set()
        results = []

        for idx in idx_image[0]:  # idx_image shape: (1, k)
            cam_offset = idx // num_frames_per_camera
            frame_offset = idx % num_frames_per_camera

            cam_id = camera_start_id + cam_offset

            if cam_id < camera_start_id or cam_id > camera_end_id:
                continue  # skip unknown camera

            if cam_id in seen_cameras:
                continue  # skip duplicate camera results

            frame_num = frame_offset * frame_step
            timestamp_sec = frame_num / fps

            results.append({
                "camera_id": f"{cam_id:04d}",
                "time_sec": round(timestamp_sec, 2),
            })

            seen_cameras.add(cam_id)

            if len(seen_cameras) == (camera_end_id - camera_start_id + 1):
                break  # already got 1 result for each camera

        return results

