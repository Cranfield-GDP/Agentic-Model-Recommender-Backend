import uvicorn
import logging
import os
from dotenv import load_dotenv
from api.api import app

from llm.config import Config
from llm.llm_provider import set_config

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


load_dotenv()
model = os.getenv("MODEL", "gemini")

try:
    config = Config(model)
    set_config(config)
    log.info(f"LLM config set with model: {config.model}")
except Exception as e:
    log.error(f"Error initializing configuration: {e}")
    exit(1)


if __name__ == "__main__":
    log.info(f"Starting FastAPI with model: {model}")
    uvicorn.run(app, host=config.fastApiHost, port=config.fastApiPort)