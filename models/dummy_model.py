from models.base import BaseOCRModel

class DummyOCRModel(BaseOCRModel):
    def __init__(self):
        super().__init__(model_name="dummy-ocr")

    def predict(self, image_path: str, prompt: str) -> str:
        # In a real scenario, this would call an API or local model
        return "This is a dummy OCR result for " + image_path


