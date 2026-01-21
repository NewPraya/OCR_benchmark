import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from models.base import BaseOCRModel

# Load API key from .env
load_dotenv()

class GeminiOCRModel(BaseOCRModel):
    def __init__(self, model_id='gemini-2.0-flash-exp'):
        # ... standard initialization ...
        super().__init__(model_name=model_id)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        self.client = genai.Client(api_key=api_key)

    def predict(self, image_path: str, prompt: str) -> str:
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # Determine mime type based on extension
        mime_type = 'image/jpeg'
        if image_path.lower().endswith('.png'):
            mime_type = 'image/png'
        elif image_path.lower().endswith('.webp'):
            mime_type = 'image/webp'

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                ),
                prompt
            ]
        )

        return response.text if response.text else ""

