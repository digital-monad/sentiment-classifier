from fastapi import FastAPI, HTTPException
from apitally.fastapi import ApitallyMiddleware
from fastapi.logger import logger as fastapi_logger
import logging
from ..inference.infer import score
from ..schemas.rest import Request, Response



logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s") # Enable logging at the INFO level
logger = logging.getLogger(__name__)
fastapi_logger.handlers = logger.handlers
fastapi_logger.setLevel(logging.INFO)

app = FastAPI()

# Add middleware to let Apitally track the api endpoints
# to get metrics and logs
app.add_middleware(
    ApitallyMiddleware,
    client_id="72bfe24d-09e6-4433-98a7-6f2b71ebc269",
    env="dev",
)

# Provide a health check endpoint for services to check the status of the API and model
# For e.g. ECS, Kubernetes, it would only be satisfied in the case of returning {"status"}
@app.get("/health")
async def health():
    try:
        dummy_sentiment, dummy_score = score("This is a dummy text") # Check the model is worknig correctly
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Model unhealthy - {e}")
    return {"status": "healthy"}

# Endpoint to predict sentiment based on given text
@app.post("/predict")
async def predict(input: Request):
    fastapi_logger.info(f"Received request: {input}")
    sentiment, pred_score = score(input.text)
    fastapi_logger.info(f"Predicted sentiment: {sentiment}, score: {pred_score}")
    # Writing these to a db or streaming service would involve invoking the respective clients
    # through fastapi async background tasks to avoid blocking the main thread on I/O bound tasks
    return Response(sentiment=sentiment, api_version="0.1", model_version="0.1", score=pred_score)
