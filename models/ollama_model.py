import ollama
import os
from models.base import BaseOCRModel

class OllamaOCRModel(BaseOCRModel):
    def __init__(self, model_id='llama3.2-vision', host='http://127.0.0.1:11434'):
        super().__init__(model_name=model_id)
        
        # Force ignore proxies for local calls to prevent 502 errors
        os.environ['no_proxy'] = os.environ.get('no_proxy', '') + ',localhost,127.0.0.1'
        os.environ['NO_PROXY'] = os.environ.get('NO_PROXY', '') + ',localhost,127.0.0.1'
        
        env_host = os.getenv("OLLAMA_HOST", host).strip()
        if "0.0.0.0" in env_host:
            env_host = env_host.replace("0.0.0.0", "127.0.0.1")
        
        self.host = env_host
        # The official library can use the client or global methods. 
        # Using Client ensures we use the correct host if OLLAMA_HOST is set differently.
        self.client = ollama.Client(host=self.host)

    def predict(self, image_path: str, prompt: str) -> str:
        try:
            # Use absolute path for images to avoid any relative path issues
            abs_image_path = os.path.abspath(image_path)
            
            # Using official library
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [abs_image_path]
                    }
                ]
            )
            
            content = response.get('message', {}).get('content', '')
            
            # Remove thinking/reasoning blocks if present
            if "<thought>" in content and "</thought>" in content:
                content = content.split("</thought>")[-1].strip()
            elif "thought" in content.lower() and "\n\n" in content:
                parts = content.split("\n\n")
                if len(parts) > 1:
                    content = parts[-1].strip()
                    
            return content
            
        except Exception as e:
            print(f"  ❌ Ollama (Library) failed: {e}")
            return ""
