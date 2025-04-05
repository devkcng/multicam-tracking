from utils.processing_text import Translation 
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
    def __init__(self, bin_files: list, dict_json: str, device, modes: list, rerank_bin_file: str = None):
            # Ensure that bin_files and modes lists have the same length
            assert len(bin_files) == len(modes), "The number of bin_files must match the number of modes"
            self.indexes = [self.load_bin_file(f) for f in bin_files]
            # Initialize re-ranking index if provided
            self.rerank_index = self.load_bin_file(rerank_bin_file) if rerank_bin_file else None
            self.translate = Translation()
            self.dict_json = self._read_json(dict_json)
            self.modes = modes
            self.device = device
            # Initialize SentenceTransformer model
            
            self.model_nomic = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device=self.device)
            extraCfg = edict({
                "type": "ViP",
                "temporal_size": 12,
                "if_use_temporal_embed": 1,
                "logit_scale_init_value": 4.60,
                "add_cls_num": 3
            })

            clipconfig = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
            clipconfig.vision_additional_config = extraCfg

            checkpoint = torch.load("YOUR_PATH_TO/CLIP-ViP/pretrain_clipvip_base_32.pt")
            cleanDict = { key.replace("clipmodel.", "") : value for key, value in checkpoint.items() }
            self.model_clipvip =  CLIPModel(config=clipconfig)
            self.model_clipvip.load_state_dict(cleanDict)
            self.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

    def load_bin_file(self, bin_file: str):
        return faiss.read_index(bin_file)

    def _read_json(self, file_json):
        with open(file_json, "r") as file:
            data = json.load(file)
        return data
    def search(self, text):
        text = self.translate(text)
        text_features = self.model_nomic.encode(text).reshape(1, -1)
        self.rerank_index 
        
        return