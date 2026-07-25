"""OpenAI-compatible image generation routes."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from src.dependencies.auth import get_authorized_headers
from src.schemas.image import ImageGenerationRequest
from src.services.image import ImageGenerationError, generate_images

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/v1/images/generations")
async def image_generations(
    request: ImageGenerationRequest,
    headers: dict = Depends(get_authorized_headers),
):
    try:
        return await generate_images(request, headers)
    except ImageGenerationError as exc:
        logger.error("Yuanbao image generation failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Image generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
