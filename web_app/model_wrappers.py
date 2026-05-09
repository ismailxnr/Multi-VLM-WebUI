import os
import io
import base64
import httpx
from abc import ABC, abstractmethod
from PIL import Image

class BaseVLM(ABC):
    """Abstract base class for all VLM wrappers."""
    
    @abstractmethod
    def load(self, model_path: str):
        pass

    @abstractmethod
    def generate(self, pil_image: Image.Image, prompt: str) -> str:
        pass
    
    @abstractmethod
    def unload(self):
        pass


class RemoteVLMWrapper(BaseVLM):
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.client = httpx.Client(timeout=300.0)

    def load(self, model_path: str):
        response = self.client.post(f"{self.endpoint}/load", json={"model_path": model_path})
        if response.status_code != 200:
            raise RuntimeError(f"Failed to load model remotely: {response.text}")

    def generate(self, pil_image: Image.Image, prompt: str) -> str:
        # Convert PIL Image to Base64
        buffered = io.BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        response = self.client.post(f"{self.endpoint}/generate", json={
            "image_base64": img_str,
            "prompt": prompt
        })
        if response.status_code != 200:
            raise RuntimeError(f"Inference failed: {response.text}")
        
        return response.json().get("caption", "")

    def unload(self):
        try:
            response = self.client.post(f"{self.endpoint}/unload")
            if response.status_code != 200:
                print(f"Warning: Failed to unload remotely: {response.text}")
        except Exception as e:
            print(f"Warning: Failed to reach remote for unload: {e}")


def get_wrapper_for_family(family: str) -> BaseVLM:
    """Factory to get the right wrapper based on model family."""
    if family == "rsllava" or family == "llava":
        endpoint = os.environ.get("RSLLAVA_ENDPOINT", "http://localhost:8002")
        return RemoteVLMWrapper(endpoint)
    elif family == "qwenvl" or family == "qwen":
        endpoint = os.environ.get("QWEN_ENDPOINT", "http://localhost:8001")
        return RemoteVLMWrapper(endpoint)
    else:
        endpoint = os.environ.get("GENERIC_ENDPOINT", "http://localhost:8003")
        return RemoteVLMWrapper(endpoint)
