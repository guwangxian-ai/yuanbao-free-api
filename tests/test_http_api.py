import json
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from openai import OpenAI

from app import app
from src.dependencies.auth import get_authorized_headers, require_api_key
from src.routers import chat as chat_router


async def fake_headers():
    return {"x-uskey": "test"}


async def fake_api_key():
    return "test-key"


def fake_completion_stream(*_args, **_kwargs):
    async def events():
        yield {"type": "text", "content": "hello"}
        yield {"type": "usage", "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}}
        yield {"type": "finish", "finish_reason": "stop"}

    return events()


class OpenAIHTTPTests(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[get_authorized_headers] = fake_headers
        app.dependency_overrides[require_api_key] = fake_api_key
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        self.client.close()

    def test_models_endpoint(self):
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["object"], "list")

    def test_non_streaming_chat_completion(self):
        with (
            patch.object(chat_router, "create_conversation", AsyncMock(return_value="temporary-chat")),
            patch.object(chat_router, "create_completion_stream", side_effect=fake_completion_stream),
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v3",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                },
            )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "chat.completion")
        self.assertEqual(payload["choices"][0]["message"]["content"], "hello")

    def test_streaming_chat_completion(self):
        with (
            patch.object(chat_router, "create_conversation", AsyncMock(return_value="temporary-chat")),
            patch.object(chat_router, "create_completion_stream", side_effect=fake_completion_stream),
        ):
            response = self.client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-v3",
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": True,
                },
            )
        self.assertEqual(response.status_code, 200)
        data_lines = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
        self.assertEqual(data_lines[-1], "[DONE]")
        chunks = [json.loads(line) for line in data_lines[:-1]]
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_official_openai_sdk(self):
        sdk_http = TestClient(app)
        sdk = OpenAI(base_url="http://testserver/v1", api_key="test-key", http_client=sdk_http)
        try:
            with (
                patch.object(chat_router, "create_conversation", AsyncMock(return_value="temporary-chat")),
                patch.object(chat_router, "create_completion_stream", side_effect=fake_completion_stream),
            ):
                completion = sdk.chat.completions.create(
                    model="deepseek-v3",
                    messages=[{"role": "user", "content": "hello"}],
                )
            self.assertEqual(completion.model, "deepseek-v3")
            self.assertEqual(completion.choices[0].message.content, "hello")
        finally:
            sdk.close()

    def test_validation_error_uses_openai_envelope(self):
        response = self.client.post("/v1/chat/completions", json={"model": "deepseek-v3", "messages": []})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "invalid_request_error")

    def test_missing_api_key_uses_openai_envelope(self):
        app.dependency_overrides.pop(require_api_key, None)
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["type"], "authentication_error")


if __name__ == "__main__":
    unittest.main()
