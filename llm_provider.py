import requests
from config import Config
import os

_config = None

def set_config(config: Config):
    global _config
    _config = config

def call_llm_endpoint(prompt: str) -> str:
    global _config
    if _config is None:
        raise ValueError("LLM config not set. Please call set_config() first.")

    headers = {}
    data = {"prompt": prompt}

    endpoint = _config.endpoint
    if _config.apiKey:
        headers["Authorization"] = f"Bearer {_config.apiKey}"

    if os.getenv("MODEL", "local").lower() == "chatgpt":
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "model": "gpt-4o-mini"
        }
        try:
            response = requests.post(endpoint, json=data, headers=headers)
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"]
            return result
        except Exception as e:
            return f"Error calling LLM endpoint: {e}"


    try:
        response = requests.post(endpoint, json=data, headers=headers)
        response.raise_for_status()
        result = response.json().get("result", "")
        return result
    except Exception as e:
        return f"Error calling LLM endpoint: {e}"

def llm(prompt: str) -> str:
    return call_llm_endpoint(prompt)
