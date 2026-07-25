"""Generate images through Yuanbao and adapt them to OpenAI Images responses."""

import asyncio
import base64
import logging
import time
from typing import Any, Dict, List

import httpx

from src.config import settings
from src.const import IMAGE_MODEL_MAPPING
from src.schemas.chat import YuanBaoChatCompletionRequest
from src.schemas.image import ImageGenerationRequest
from src.services.chat.completion import ChatCompletionError, create_completion_stream
from src.services.chat.conversation import create_conversation

logger = logging.getLogger(__name__)
_generation_lock = asyncio.Lock()


class ImageGenerationError(Exception):
    """Raised when Yuanbao finishes without returning usable images."""


def _build_prompt(request: ImageGenerationRequest) -> str:
    details = []
    if request.size and request.size != "auto":
        width, height = (int(part) for part in request.size.lower().split("x", maxsplit=1))
        if width == height:
            details.append("square composition")
        elif width > height:
            details.append("landscape composition")
        else:
            details.append("portrait composition")
    if request.style == "natural":
        details.append("natural visual style")
    elif request.style == "vivid":
        details.append("vivid visual style")
    if request.quality in {"hd", "high"}:
        details.append("high detail")
    if not details:
        return request.prompt
    return f"{request.prompt}\n\nImage requirements: {', '.join(details)}."


async def _download_as_base64(url: str) -> str:
    async with httpx.AsyncClient(follow_redirects=True, timeout=settings.image_download_timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
    return base64.b64encode(response.content).decode("ascii")


async def generate_images(request: ImageGenerationRequest, headers: Dict[str, str]) -> Dict[str, Any]:
    """Wait for Yuanbao's final image event and return an OpenAI-compatible payload."""

    async with _generation_lock:
        chat_id = await create_conversation(settings.agent_id, headers)
        model_info = IMAGE_MODEL_MAPPING[request.model]
        yuanbao_request = YuanBaoChatCompletionRequest(
            agent_id=settings.agent_id,
            chat_id=chat_id,
            prompt=_build_prompt(request),
            chat_model_id=model_info["model"],
            plugin="ImageHelper",
        )

        images: List[Dict[str, Any]] = []
        try:
            events = create_completion_stream(
                yuanbao_request,
                headers,
                should_remove_conversation=True,
                timeout=settings.image_generation_timeout,
            )
            async for event in events:
                if event.get("type") == "image":
                    images = event.get("images") or images
        except ChatCompletionError as exc:
            raise ImageGenerationError(str(exc)) from exc

        unique_images = []
        seen_urls = set()
        for image in images:
            url = image.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_images.append(image)
        if not unique_images:
            raise ImageGenerationError("Yuanbao completed the request without returning any image URLs")

        selected = unique_images[: request.n]
        if len(selected) < request.n:
            raise ImageGenerationError(
                f"Yuanbao returned {len(selected)} image(s), fewer than requested n={request.n}"
            )

        data: List[Dict[str, str]] = []
        if request.response_format == "b64_json":
            encoded = await asyncio.gather(*(_download_as_base64(image["url"]) for image in selected))
            data = [{"b64_json": value} for value in encoded]
        else:
            data = [{"url": image["url"]} for image in selected]

        logger.info("Generated %d Yuanbao image(s) in temporary conversation %s", len(data), chat_id)
        return {"created": int(time.time()), "data": data}
