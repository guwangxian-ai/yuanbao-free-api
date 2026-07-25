"""Chat request and response conversion helpers."""

import json
import re
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from src.const import MODEL_MAPPING
from src.schemas.chat import ChatCompletionTool, Message

YUANBAO_MARKUP_PATTERN = re.compile(r"\[\]\(@mark_[^)]+\)")
YUANBAO_MARKUP_PREFIX = "[](@mark_"


def get_model_info(model_name: str) -> Optional[Dict]:
    return MODEL_MAPPING.get(model_name.lower())


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    text_parts: List[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"text", "input_text"}:
            text_parts.append(str(part.get("text", "")))
        elif part_type in {"image_url", "input_image"}:
            raise ValueError(
                "OpenAI image content parts are not supported yet; "
                "upload the file through /v1/upload and pass it in multimedia"
            )
    return "\n".join(part for part in text_parts if part)


def parse_messages(messages: List[Message]) -> str:
    """Serialize OpenAI messages, tool calls, and tool results for Yuanbao."""

    prompt_parts: List[str] = []
    for message in messages:
        content = _content_to_text(message.content)

        if message.role == "assistant" and message.tool_calls:
            calls = [
                {
                    "id": call.id,
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                }
                for call in message.tool_calls
            ]
            if content:
                prompt_parts.append(f"assistant: {content}")
            prompt_parts.append(f"assistant tool_calls: {json.dumps(calls, ensure_ascii=False)}")
            continue

        if message.role == "tool":
            label = f"tool call_id={message.tool_call_id or 'unknown'}"
            if message.name:
                label += f" name={message.name}"
            prompt_parts.append(f"{label}: {content}")
            continue

        prompt_parts.append(f"{message.role}: {content}")

    return "\n".join(prompt_parts)


def build_tool_prompt(
    prompt: str,
    tools: Optional[List[ChatCompletionTool]],
    tool_choice: Any,
) -> str:
    """Add a conservative JSON tool-calling contract for non-native providers."""

    if not tools or tool_choice == "none":
        return prompt

    tool_definitions = [tool.model_dump() for tool in tools]
    required_tool: Optional[str] = None
    if tool_choice == "required":
        required_tool = "any available tool"
    elif isinstance(tool_choice, dict):
        function = tool_choice.get("function") or {}
        required_tool = function.get("name")

    requirement = (
        f"You must call {required_tool}."
        if required_tool
        else "Call a tool only when it is needed; otherwise answer normally."
    )
    instructions = f"""

[OpenAI tool calling]
Available tools:
{json.dumps(tool_definitions, ensure_ascii=False, separators=(',', ':'))}

{requirement}
When calling tools, output only this JSON object and no surrounding prose or Markdown:
{{"tool_calls":[{{"name":"tool_name","arguments":{{"argument":"value"}}}}]}}
Use only listed tool names and arguments that satisfy the tool's JSON schema.
""".strip()
    return f"{prompt}\n\n{instructions}"


def parse_tool_calls(content: str, allowed_names: set[str]) -> Optional[List[Dict[str, Any]]]:
    """Parse the strict JSON tool envelope produced by the prompt adapter."""

    candidate = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
    if not (candidate.startswith("{") and candidate.endswith("}")):
        return None

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    raw_calls = payload.get("tool_calls") if isinstance(payload, dict) else None
    if not isinstance(raw_calls, list) or not raw_calls:
        return None

    parsed_calls: List[Dict[str, Any]] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            return None
        name = raw_call.get("name")
        if not isinstance(name, str) or name not in allowed_names:
            return None
        arguments = raw_call.get("arguments", {})
        if isinstance(arguments, str):
            arguments_json = arguments
        else:
            arguments_json = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        parsed_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": name, "arguments": arguments_json},
            }
        )
    return parsed_calls


def _normalize_finish_reason(reason: Any) -> str:
    if reason in {"length", "max_tokens"}:
        return "length"
    if reason in {"content_filter", "sensitive"}:
        return "content_filter"
    return "stop"


def _extract_generated_images(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    if event.get("type") != "replace":
        return []
    replacement = event.get("replace")
    if not isinstance(replacement, dict):
        return []

    images: List[Dict[str, Any]] = []
    for media in replacement.get("multimedias") or []:
        if not isinstance(media, dict) or media.get("mediaType") != "image":
            continue
        url = media.get("url") or media.get("downloadUrl") or media.get("resourceUrl")
        if not url or media.get("available") is False:
            continue
        images.append(
            {
                "url": str(url),
                "download_url": media.get("downloadUrl"),
                "resource_url": media.get("resourceUrl"),
                "width": media.get("width"),
                "height": media.get("height"),
                "media_id": media.get("mediaId"),
            }
        )
    return images


def _drain_yuanbao_text(buffer: str) -> tuple[str, str]:
    """Remove Yuanbao-only markup while retaining a possible split marker."""

    cleaned = YUANBAO_MARKUP_PATTERN.sub("", buffer)
    marker_start = cleaned.rfind(YUANBAO_MARKUP_PREFIX)
    if marker_start >= 0 and ")" not in cleaned[marker_start:]:
        return cleaned[:marker_start], cleaned[marker_start:]

    max_suffix = min(len(cleaned), len(YUANBAO_MARKUP_PREFIX) - 1)
    for suffix_length in range(max_suffix, 0, -1):
        suffix = cleaned[-suffix_length:]
        if YUANBAO_MARKUP_PREFIX.startswith(suffix):
            return cleaned[:-suffix_length], suffix
    return cleaned, ""


def _finalize_yuanbao_text(buffer: str) -> str:
    cleaned = YUANBAO_MARKUP_PATTERN.sub("", buffer)
    if YUANBAO_MARKUP_PREFIX.startswith(cleaned) or cleaned.startswith(YUANBAO_MARKUP_PREFIX):
        return ""
    return cleaned


async def process_response_stream(response: httpx.Response) -> AsyncGenerator[Dict[str, Any], None]:
    """Convert Yuanbao SSE lines into provider-neutral events."""

    if response.status_code >= 400:
        body = (await response.aread()).decode("utf-8", errors="replace")
        raise RuntimeError(f"Yuanbao API returned HTTP {response.status_code}: {body[:500]}")

    finished = False
    pending_text = ""
    async for line in response.aiter_lines():
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            trailing_text = _finalize_yuanbao_text(pending_text)
            if trailing_text:
                yield {"type": "text", "content": trailing_text}
            if not finished:
                yield {"type": "finish", "finish_reason": "stop"}
            return
        if not data.startswith("{"):
            continue

        try:
            event: Dict[str, Any] = json.loads(data)
        except json.JSONDecodeError:
            continue

        usage_info = event.get("tokenUsageInfo")
        if isinstance(usage_info, dict):
            yield {
                "type": "usage",
                "usage": {
                    "prompt_tokens": int(usage_info.get("promptTokens", 0)),
                    "completion_tokens": int(usage_info.get("completionTokens", 0)),
                    "total_tokens": int(usage_info.get("totalTokens", 0)),
                },
            }

        images = _extract_generated_images(event)
        if images:
            replacement = event.get("replace") or {}
            yield {
                "type": "image",
                "images": images,
                "asset_id": replacement.get("assetId"),
            }

        if event.get("stopReason"):
            trailing_text = _finalize_yuanbao_text(pending_text)
            if trailing_text:
                yield {"type": "text", "content": trailing_text}
            finished = True
            yield {
                "type": "finish",
                "finish_reason": _normalize_finish_reason(event["stopReason"]),
            }
            return

        if event.get("type") == "text" and event.get("msg"):
            pending_text += str(event["msg"])
            clean_text, pending_text = _drain_yuanbao_text(pending_text)
            if clean_text:
                yield {"type": "text", "content": clean_text}

    if not finished:
        trailing_text = _finalize_yuanbao_text(pending_text)
        if trailing_text:
            yield {"type": "text", "content": trailing_text}
        yield {"type": "finish", "finish_reason": "stop"}
