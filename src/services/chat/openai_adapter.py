"""OpenAI Chat Completions response adapter."""

import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, Optional, Tuple

from src.schemas.chat import ChatCompletionRequest
from src.utils.chat import parse_tool_calls

EMPTY_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def create_completion_context() -> Tuple[str, int]:
    return f"chatcmpl-{uuid.uuid4().hex}", int(time.time())


def _active_tool_names(request: ChatCompletionRequest) -> set[str]:
    if not request.tools or request.tool_choice == "none":
        return set()
    return {tool.function.name for tool in request.tools}


async def _collect_events(
    events: AsyncGenerator[Dict[str, Any], None],
) -> Tuple[str, str, Dict[str, int]]:
    content_parts = []
    finish_reason = "stop"
    usage = dict(EMPTY_USAGE)
    async for event in events:
        if event["type"] == "text":
            content_parts.append(event["content"])
        elif event["type"] == "finish":
            finish_reason = event["finish_reason"]
        elif event["type"] == "usage":
            usage = event["usage"]
    return "".join(content_parts), finish_reason, usage


async def create_chat_completion(
    events: AsyncGenerator[Dict[str, Any], None],
    request: ChatCompletionRequest,
    completion_id: str,
    created: int,
) -> Dict[str, Any]:
    content, finish_reason, usage = await _collect_events(events)
    tool_calls = parse_tool_calls(content, _active_tool_names(request))
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": None if tool_calls else content,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage,
    }


def _chunk(
    completion_id: str,
    created: int,
    model: str,
    delta: Dict[str, Any],
    finish_reason: Optional[str] = None,
) -> str:
    return json.dumps(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _usage_chunk(
    completion_id: str,
    created: int,
    model: str,
    usage: Dict[str, int],
) -> str:
    return json.dumps(
        {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [],
            "usage": usage,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


async def create_chat_completion_stream(
    events: AsyncGenerator[Dict[str, Any], None],
    request: ChatCompletionRequest,
    completion_id: str,
    created: int,
) -> AsyncGenerator[str, None]:
    yield _chunk(completion_id, created, request.model, {"role": "assistant", "content": ""})

    tool_names = _active_tool_names(request)
    usage = dict(EMPTY_USAGE)
    finish_reason = "stop"

    if tool_names:
        content, finish_reason, usage = await _collect_events(events)
        tool_calls = parse_tool_calls(content, tool_names)
        if tool_calls:
            deltas = [{"index": index, **tool_call} for index, tool_call in enumerate(tool_calls)]
            yield _chunk(completion_id, created, request.model, {"tool_calls": deltas})
            finish_reason = "tool_calls"
        elif content:
            yield _chunk(completion_id, created, request.model, {"content": content})
    else:
        async for event in events:
            if event["type"] == "text":
                yield _chunk(completion_id, created, request.model, {"content": event["content"]})
            elif event["type"] == "finish":
                finish_reason = event["finish_reason"]
            elif event["type"] == "usage":
                usage = event["usage"]

    yield _chunk(completion_id, created, request.model, {}, finish_reason)

    if request.stream_options and request.stream_options.get("include_usage"):
        yield _usage_chunk(completion_id, created, request.model, usage)
    yield "[DONE]"
