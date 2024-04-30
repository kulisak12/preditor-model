import pydantic


class PredictionConfig(pydantic.BaseModel):
    """Configuration for the prediction algorithms.

    max_length: The maximum number of tokens generated during prediction.
    confidence: Higher confidence leads to longer suggestions.
    """
    max_length: int = pydantic.Field(10, ge=1)
    confidence: float = pydantic.Field(7.0, ge=1.0)
