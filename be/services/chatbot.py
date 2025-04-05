import av
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration
from torch.nn import DataParallel

class LlavaNextVideoInference:
    def __init__(self, model_id="llava-hf/LLaVA-NeXT-Video-7B-hf", dtype=torch.float16, video_paths=None, num_frames=8):
        self.model_id = model_id
        self.dtype = dtype
        self.video_paths = video_paths or []
        self.num_frames = num_frames

        # Load model and wrap with DataParallel to use multiple GPUs
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True
        )

        # Use DataParallel to distribute the model across available GPUs
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = DataParallel(self.model)

        self.processor = LlavaNextVideoProcessor.from_pretrained(self.model_id)

        # Initialize conversation history
        self.chat_history = []

        # Pre-process the videos before chat starts
        if self.video_paths:
            self.video_clips = self.process_multiple_videos(self.video_paths)

    def _decode_and_convert(self, frames):
        """ Convert frames to ndarray using parallel processing """
        with ThreadPoolExecutor() as executor:
            result = list(executor.map(lambda f: f.to_ndarray(format="rgb24"), frames))
        return np.stack(result)

    def _read_video_pyav(self, container, indices):
        """ Read frames from the video container using the provided indices """
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return self._decode_and_convert(frames)

    def _prepare_prompt(self):
        """ Prepare the chat prompt using conversation history """
        conversation = []
        for message in self.chat_history:
            if message["role"] == "user":
                conversation.append({"type": "text", "text": message["content"]})
            elif message["role"] == "assistant":
                conversation.append({"type": "text", "text": message["content"]})

        # Add the current user message
        conversation.append({"type": "text", "text": self.chat_history[-1]["content"]})

        # Add the video content
        conversation.append({"type": "video"})

        # Prepare prompt with the conversation history
        return self.processor.apply_chat_template([{"role": "user", "content": conversation}], add_generation_prompt=True)

    def process_video(self, video_path):
        """ Process a single video and return the video clip (extracted frames) """
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        clip = self._read_video_pyav(container, indices)
        return clip

    def process_multiple_videos(self, video_paths):
        """ Process multiple videos and return a list of clips """
        video_clips = []
        for video_path in video_paths:
            clip = self.process_video(video_path)
            video_clips.append(clip)
        return video_clips

    def infer(self, user_question):
        """ Perform inference on the pre-processed videos and return responses """
        # Ensure the videos are processed and ready for inference
        if not hasattr(self, 'video_clips'):
            raise ValueError("No videos provided for inference.")

        # Update chat history with the new user question
        self.chat_history.append({"role": "user", "content": user_question})

        # Prepare the prompt using the conversation history
        prompt = self._prepare_prompt()

        # Prepare inputs for the model
        inputs = self.processor(
            text=prompt,
            videos=self.video_clips,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Run the model to generate output for each video
        output = self.model.generate(**inputs, max_new_tokens=100, do_sample=False)
        answer = self.processor.decode(output[0][2:], skip_special_tokens=True)

        # Save model's response to history
        self.chat_history.append({"role": "assistant", "content": answer})

        return answer

    def get_chat_history(self):
        """ Return the conversation history """
        return self.chat_history
