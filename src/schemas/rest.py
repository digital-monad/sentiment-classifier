from pydantic import BaseModel, Field
from typing import Optional, Literal

class Request(BaseModel):
    # Schema for the request body
    text: str = Field(..., min_length=1, max_length=5000, description="Review text to classify sentiment")
    asin: Optional[str] = Field(None, min_length=10, max_length=10, description="Product Id")
    parent_asin: Optional[str] = Field(None, min_length=10, max_length=10, description="Parent Product Id")

class Response(BaseModel):
    # Schema for the response body
    sentiment: Literal["negative", "neutral", "positive"] = Field(..., description="The predicted sentiment of the text.")
    api_version: str = Field(..., description="The version of the API that made the prediction.")
    model_version: str = Field(..., description="The version of the model that made the prediction.")
    score: float = Field(..., ge=0.0, description="Raw logit score of the prediction")
