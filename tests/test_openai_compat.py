import json
import unittest

from src.schemas.chat import ChatCompletionRequest, Message
from src.services.chat.openai_adapter import (
    create_chat_completion,
    create_chat_completion_stream,
    create_completion_context,
)
from src.utils.chat import build_tool_prompt, parse_messages, parse_tool_calls, process_response_stream


async def event_stream(*events):
    for event in events:
        yield event


class FakeYuanbaoResponse:
    status_code = 200

    def __init__(self, *lines):
        self.lines = lines

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class OpenAICompatibilityTests(unittest.IsolatedAsyncioTestCase):
    def test_request_defaults_to_non_streaming(self):
        request = ChatCompletionRequest(
            model="deepseek-v3",
            messages=[{"role": "user", "content": "hello"}],
        )
        self.assertFalse(request.stream)
        self.assertEqual(request.n, 1)

    def test_messages_preserve_roles_and_tool_history(self):
        messages = [
            Message(role="system", content="Be concise"),
            Message(role="assistant", content=None, tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path":"app.py"}'},
            }]),
            Message(role="tool", tool_call_id="call_1", content="file contents"),
        ]
        prompt = parse_messages(messages)
        self.assertIn("system: Be concise", prompt)
        self.assertIn("assistant tool_calls:", prompt)
        self.assertIn("tool call_id=call_1: file contents", prompt)

    def test_tool_prompt_and_parser(self):
        request = ChatCompletionRequest(
            model="deepseek-v3",
            messages=[{"role": "user", "content": "read app.py"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }],
        )
        prompt = build_tool_prompt("user: read app.py", request.tools, request.tool_choice)
        self.assertIn("read_file", prompt)

        calls = parse_tool_calls(
            '{"tool_calls":[{"name":"read_file","arguments":{"path":"app.py"}}]}',
            {"read_file"},
        )
        self.assertIsNotNone(calls)
        self.assertEqual(calls[0]["function"]["name"], "read_file")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"path": "app.py"})

    async def test_non_streaming_response(self):
        request = ChatCompletionRequest(
            model="deepseek-v3",
            messages=[{"role": "user", "content": "hello"}],
        )
        completion_id, created = create_completion_context()
        response = await create_chat_completion(
            event_stream(
                {"type": "text", "content": "hel"},
                {"type": "text", "content": "lo"},
                {"type": "usage", "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}},
                {"type": "finish", "finish_reason": "stop"},
            ),
            request,
            completion_id,
            created,
        )
        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(response["model"], "deepseek-v3")
        self.assertEqual(response["choices"][0]["message"]["content"], "hello")
        self.assertEqual(response["usage"]["total_tokens"], 3)

    async def test_non_streaming_tool_call_response(self):
        request = ChatCompletionRequest(
            model="deepseek-v3",
            messages=[{"role": "user", "content": "read app.py"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "read_file",
                    "parameters": {"type": "object"},
                },
            }],
        )
        completion_id, created = create_completion_context()
        response = await create_chat_completion(
            event_stream(
                {
                    "type": "text",
                    "content": '{"tool_calls":[{"name":"read_file","arguments":{"path":"app.py"}}]}',
                },
                {"type": "finish", "finish_reason": "stop"},
            ),
            request,
            completion_id,
            created,
        )
        choice = response["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        self.assertEqual(choice["message"]["tool_calls"][0]["function"]["name"], "read_file")

    async def test_streaming_response_uses_one_completion_id(self):
        request = ChatCompletionRequest(
            model="deepseek-v3",
            messages=[{"role": "user", "content": "hello"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        completion_id, created = create_completion_context()
        chunks = [
            chunk
            async for chunk in create_chat_completion_stream(
                event_stream(
                    {"type": "text", "content": "hello"},
                    {"type": "usage", "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}},
                    {"type": "finish", "finish_reason": "stop"},
                ),
                request,
                completion_id,
                created,
            )
        ]
        self.assertEqual(chunks[-1], "[DONE]")
        payloads = [json.loads(chunk) for chunk in chunks[:-1]]
        self.assertTrue(all(payload["id"] == completion_id for payload in payloads))
        self.assertEqual(payloads[-1]["choices"], [])
        self.assertEqual(payloads[-1]["usage"]["total_tokens"], 3)

    async def test_yuanbao_markup_is_removed_across_stream_chunks(self):
        response = FakeYuanbaoResponse(
            'data: {"type":"text","msg":"hello["}',
            'data: {"type":"text","msg":"](@mark_under"}',
            'data: {"type":"text","msg":"line=1)"}',
            'data: {"type":"meta","stopReason":"stop"}',
        )
        events = [event async for event in process_response_stream(response)]
        content = "".join(event["content"] for event in events if event["type"] == "text")
        self.assertEqual(content, "hello")
        self.assertEqual(events[-1], {"type": "finish", "finish_reason": "stop"})


if __name__ == "__main__":
    unittest.main()
