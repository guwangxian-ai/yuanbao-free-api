"""OpenAI-compatible model and chat completion routes."""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from src.config import settings
from src.const import MODEL_MAPPING
from src.dependencies.auth import get_authorized_headers, require_api_key
from src.schemas.chat import ChatCompletionRequest, YuanBaoChatCompletionRequest
from src.services.chat.completion import ChatCompletionError, create_completion_stream
from src.services.chat.conversation import create_conversation
from src.services.chat.openai_adapter import (
    create_chat_completion,
    create_chat_completion_stream,
    create_completion_context,
)
from src.utils.chat import build_tool_prompt, get_model_info, parse_messages

logger = logging.getLogger(__name__)
router = APIRouter()


def _model_object(model_id: str) -> dict:
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "yuanbao",
    }


@router.get("/v1/models")
async def list_models(_: str = Depends(require_api_key)):
    return {
        "object": "list",
        "data": [_model_object(model_id) for model_id in MODEL_MAPPING],
    }


@router.get("/v1/models/{model_id}")
async def retrieve_model(model_id: str, _: str = Depends(require_api_key)):
    if model_id not in MODEL_MAPPING:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return _model_object(model_id)


def _apply_response_format(prompt: str, response_format: dict | None) -> str:
    if not response_format:
        return prompt
    response_type = response_format.get("type")
    if response_type == "json_object":
        return f"{prompt}\n\nRespond with one valid JSON object and no Markdown."
    if response_type == "json_schema":
        schema = response_format.get("json_schema") or {}
        return (
            f"{prompt}\n\nRespond with JSON that matches this schema and no Markdown:\n"
            f"{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}"
        )
    return prompt


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    headers: dict = Depends(get_authorized_headers),
):
    """Create an OpenAI-compatible streaming or non-streaming completion."""

    try:
        auto_created_conversation = not request.chat_id
        if not request.chat_id:
            request.chat_id = await create_conversation(settings.agent_id, headers)
            logger.info("Created temporary conversation %s", request.chat_id)

        model_info = get_model_info(request.model)
        if not model_info:
            raise HTTPException(status_code=400, detail=f"Unsupported model '{request.model}'")

        prompt = parse_messages(request.messages)
        prompt = build_tool_prompt(prompt, request.tools, request.tool_choice)
        prompt = _apply_response_format(prompt, request.response_format)

        yuanbao_request = YuanBaoChatCompletionRequest(
            agent_id=settings.agent_id,
            chat_id=request.chat_id,
            prompt=prompt,
            chat_model_id=model_info["model"],
            multimedia=request.multimedia,
            support_functions=model_info.get("support_functions"),
        )
        events = create_completion_stream(
            yuanbao_request,
            headers,
            request.should_remove_conversation or auto_created_conversation,
        )
        completion_id, created = create_completion_context()

        if request.stream:
            stream = create_chat_completion_stream(events, request, completion_id, created)
            return EventSourceResponse(stream, media_type="text/event-stream")

        return await create_chat_completion(events, request, completion_id, created)

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ChatCompletionError as e:
        logger.error("Yuanbao completion failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        logger.exception("Chat completion failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
