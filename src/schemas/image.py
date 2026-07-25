"""OpenAI-compatible image generation schemas."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.const import IMAGE_MODEL_MAPPING


class ImageGenerationRequest(BaseModel):
    """Parameters accepted by the OpenAI Images generation endpoint."""

    model_config = ConfigDict(extra="allow")

    prompt: str = Field(min_length=1)
    model: str = "yuanbao-image"
    n: int = Field(default=4, ge=1, le=4)
    quality: Optional[Literal["standard", "hd", "auto", "low", "medium", "high"]] = None
    response_format: Literal["url", "b64_json"] = "url"
    size: Optional[str] = "1024x1024"
    style: Optional[Literal["vivid", "natural"]] = None
    user: Optional[str] = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if value not in IMAGE_MODEL_MAPPING:
            raise ValueError(f"model must be one of {list(IMAGE_MODEL_MAPPING.keys())}")
        return value

    @field_validator("size")
    @classmethod
    def validate_size(cls, value: Optional[str]) -> Optional[str]:
        if value is None or value == "auto":
            return value
        try:
            width, height = (int(part) for part in value.lower().split("x", maxsplit=1))
        except (TypeError, ValueError):
            raise ValueError("size must be 'auto' or WIDTHxHEIGHT") from None
        if width <= 0 or height <= 0:
            raise ValueError("size dimensions must be positive")
        return value
