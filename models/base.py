from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseOCRModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def predict(self, image_path: str, prompt: str) -> str:
        """
        Perform OCR on the image with a specific prompt and return the result.
        """
        pass

    def get_info(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}


