import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
from models.base import BaseOCRModel

# Load API key from .env
load_dotenv()

class QwenOCRModel(BaseOCRModel):
    def __init__(self, model_id='qwen3-vl-flash'):
        super().__init__(model_name=model_id)
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not found in environment variables.")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

    def predict(self, image_path: str, prompt: str) -> str:
        # Read and encode image to base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Determine mime type
        mime_type = 'image/jpeg'
        if image_path.lower().endswith('.png'):
            mime_type = 'image/png'
        elif image_path.lower().endswith('.webp'):
            mime_type = 'image/webp'

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            # extra_body is specific to DashScope for features like reasoning
            # but for standard OCR we might not need enable_thinking unless requested.
            # Keeping it simple for now to match Gemini's output.
        )

        return response.choices[0].message.content if response.choices else ""

