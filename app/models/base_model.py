import requests

class BaseModel:
    def __init__(self, model_name="mistral", temperature=0.35, ollama_url="http://localhost:11434"):
        self.model_name = model_name
        self.temperature = temperature
        self.ollama_url = ollama_url

    def generate_response(self, prompt, system_prompt=None, history=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            for msg in history:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "options": {
                "temperature": self.temperature,
            },
            "stream": False
        }
        try:
            response = requests.post(
                #f"{self.ollama_url}/api/chat",
                self.ollama_url,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            if "message" in data and "content" in data["message"]:
                return data["message"]["content"]
            else:
                return "[Error: Unexpected Ollama response format]"
        except Exception as e:
            return f"[Error communicating with Ollama: {e}]"