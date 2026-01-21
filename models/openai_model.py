import os
import base64
from openai import OpenAI
from dotenv import load_dotenv
from models.base import BaseOCRModel

# Load API key from .env
load_dotenv()

class OpenAIOCRModel(BaseOCRModel):
    def __init__(self, model_id='gpt-4o'):
        super().__init__(model_name=model_id)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        self.client = OpenAI(api_key=api_key)

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

        try:
            # 1. Try the new 'responses.create' API first for newer/experimental models
            if self.model_name.startswith(("gpt-5", "gpt-4.1", "o")):
                try:
                    response = self.client.responses.create(
                        model=self.model_name,
                        input=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "input_text", "text": prompt},
                                    {
                                        "type": "input_image",
                                        "image_url": f"data:{mime_type};base64,{base64_image}",
                                    },
                                ],
                            }
                        ],
                    )
                    if hasattr(response, 'output_text') and response.output_text:
                        return response.output_text
                except Exception:
                    pass

            # 2. Standard Chat Completions API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
            )
            
            if response.choices:
                return response.choices[0].message.content or ""
            
            return ""
        except Exception as e:
            print(f"  ❌ OpenAI API Error: {e}")
            return ""
