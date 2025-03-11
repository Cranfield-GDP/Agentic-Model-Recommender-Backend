import aiohttp
import os
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

endpoint = os.getenv("PIPELINE_ENDPOINT", "http://0.0.0.0:8001/load_model")


async def deploy_model_in_cloud(model_name: str, category: str, deployment: str, slice: str):
    try:
        reqBody = {"model_name": model_name, "task_name": category}
        log.info(f"Sending request to the deployer with model: {model_name} and category: {category}")
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, json=reqBody) as response:
                response_data = await response.json()
                log.info(f"The response code from the pipeline is {response.status} and the response is {response_data}")

                if deployment.lower() == "edge":
                    # TO-DO
                    # provide access to the edge zone
                    pass
    except Exception as e:
        log.error("An error while sending request to the deployer ", e)


