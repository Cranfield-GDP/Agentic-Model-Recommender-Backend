import os
import logging
import uvicorn
from api import app
from config import Config
from llm_provider import set_config

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

model = os.getenv("MODEL", "chatgpt") 
try:
    config = Config(model)
    set_config(config)
    log.info(f"LLM config set with endpoint: {config.endpoint}")
except Exception as e:
    log.error(f"Error initializing configuration: {e}")
    exit(1)

if __name__ == "__main__":
    log.info(f"Starting FastAPI with model: {model}")
    uvicorn.run(app, host=config.fastApiHost, port=config.fastApiPort)
