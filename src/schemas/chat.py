"""OpenAI-compatible chat request schemas."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.const import MODEL_MAPPING
from src.schemas.common import Media


class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ChatCompletionTool(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(BaseModel):
    """OpenAI chat message, including tool-call history."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI Chat Completions request with Yuanbao extensions."""

    model_config = ConfigDict(extra="allow")

    messages: List[Message]
    model: str
    stream: bool = False
    stream_options: Optional[Dict[str, Any]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    n: int = 1
    user: Optional[str] = None
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[ChatCompletionTool]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None

    # Backward-compatible Yuanbao extensions. OpenAI clients can omit all of them.
    chat_id: Optional[str] = None
    should_remove_conversation: bool = False
    multimedia: List[Media] = Field(default_factory=list)

    @field_validator("messages")
    @classmethod
    def check_messages_not_empty(cls, value: List[Message]) -> List[Message]:
        if not value:
            raise ValueError("messages cannot be an empty list")
        return value

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        if value not in MODEL_MAPPING:
            raise ValueError(f"model must be one of {list(MODEL_MAPPING.keys())}")
        return value

    @field_validator("n")
    @classmethod
    def validate_n(cls, value: int) -> int:
        if value != 1:
            raise ValueError("only n=1 is supported")
        return value


class YuanBaoChatCompletionRequest(BaseModel):
    """Internal request sent to Tencent Yuanbao."""

    agent_id: str
    chat_id: str
    prompt: str
    chat_model_id: str
    multimedia: List[Media] = Field(default_factory=list)
    support_functions: Optional[List[str]] = None
    plugin: str = "Adaptive"
