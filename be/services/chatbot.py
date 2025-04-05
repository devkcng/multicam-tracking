import av
import torch
import numpy as np
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration

class VideoChatbot:
    def __init__(self, model_id="llava-hf/LLaVA-NeXT-Video-7B-hf", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        ).to(self.device)

        self.processor = LlavaNextVideoProcessor.from_pretrained(model_id)
        self.history = []  # Store chat history

    def _read_video_pyav(self, container, indices):
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    def _sample_frame_indices(self, total_frames, num_samples=8):
        return np.linspace(0, total_frames - 1, num_samples).astype(int)

    def chat(self, video_paths, user_message="What is happening in these videos?"):
        # Ensure we handle multiple video paths
        video_clips = []
        for video_path in video_paths:
            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            indices = self._sample_frame_indices(total_frames)
            clip = self._read_video_pyav(container, indices)
            video_clips.append(clip)  # Collect video clips

        # Add new user turn with video
        user_turn = {
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
                {"type": "video"},
            ],
        }
        self.history.append(user_turn)

        # Count how many {"type": "video"} in the full conversation
        video_count = sum(
            1 for message in self.history for item in message["content"] if item["type"] == "video"
        )

        # Ensure video_inputs matches the number of video tokens
        video_inputs = video_clips[:video_count]  # Slice to match

        # Build prompt
        prompt = self.processor.apply_chat_template(self.history, add_generation_prompt=True)

        # Tokenize with matched video inputs
        inputs = self.processor(
            text=prompt,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
        response_text = self.processor.decode(output[0][2:], skip_special_tokens=True)

        # Add assistant's response to history
        self.history.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response_text}]
        })

        return response_text
