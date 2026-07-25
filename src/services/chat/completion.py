"""Tencent Yuanbao chat completion transport."""

import logging
from typing import Any, AsyncGenerator, Dict

import httpx

from src.schemas.chat import YuanBaoChatCompletionRequest
from src.services.chat.conversation import remove_conversation
from src.utils.chat import process_response_stream

CHAT_URL = "https://yuanbao.tencent.com/api/chat/{}"

DEFAULT_TIMEOUT = 60
logger = logging.getLogger(__name__)


class ChatCompletionError(Exception):
    """聊天完成异常"""

    pass


async def create_completion_stream(
    chat_request: YuanBaoChatCompletionRequest,
    headers: Dict[str, str],
    should_remove_conversation: bool = False,
    timeout: int = DEFAULT_TIMEOUT,
) -> AsyncGenerator[Dict[str, Any], None]:
    """创建聊天完成流

    Args:
        chat_request: 聊天请求
        headers: 认证请求头
        should_remove_conversation: 是否删除会话
        timeout: 超时时间

    Yields:
        str: SSE 格式的数据块

    Raises:
        ChatCompletionError: 聊天完成失败时抛出
    """
    multimedia = [m.model_dump() for m in chat_request.multimedia]
    body = {
        "model": "gpt_175B_0404",
        "prompt": chat_request.prompt,
        "plugin": "Adaptive",
        "displayPrompt": chat_request.prompt,
        "displayPromptType": 1,
        "options": {"imageIntention": {"needIntentionModel": True, "backendUpdateFlag": 2, "intentionStatus": True}},
        "multimedia": multimedia,
        "agentId": chat_request.agent_id,
        "supportHint": 1,
        "version": "v2",
        "chatModelId": chat_request.chat_model_id,
    }
    if chat_request.support_functions:
        body["supportFunctions"] = chat_request.support_functions

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                CHAT_URL.format(chat_request.chat_id),
                json=body,
                headers=headers,
                timeout=timeout,
            ) as response:
                async for event in process_response_stream(response):
                    yield event

    except Exception as e:
        raise ChatCompletionError(e)

    finally:
        if should_remove_conversation:
            try:
                await remove_conversation(chat_request.chat_id, headers)
            except Exception as e:
                logger.warning("Failed to remove temporary conversation %s: %s", chat_request.chat_id, e)
