import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from openai import OpenAI

from app import app
from src.dependencies.auth import get_authorized_headers, require_api_key
from src.routers import image as image_router
from src.schemas.image import ImageGenerationRequest
from src.services.image import generation


async def fake_headers():
    return {"x-uskey": "test"}


async def fake_api_key():
    return "test-key"


async def fake_image_result(*_args, **_kwargs):
    return {
        "created": 123,
        "data": [
            {"url": "https://example.test/1.png"},
            {"url": "https://example.test/2.png"},
        ],
    }


class ImageGenerationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_collects_and_slices_final_image_event(self):
        async def events(*_args, **_kwargs):
            yield {"type": "image", "images": [
                {"url": "https://example.test/1.png"},
                {"url": "https://example.test/2.png"},
                {"url": "https://example.test/3.png"},
                {"url": "https://example.test/4.png"},
            ]}
            yield {"type": "finish", "finish_reason": "stop"}

        request = ImageGenerationRequest(prompt="cat", n=2)
        with (
            patch.object(generation, "create_conversation", AsyncMock(return_value="temporary-chat")),
            patch.object(generation, "create_completion_stream", side_effect=events) as stream,
        ):
            response = await generation.generate_images(request, {"x-uskey": "test"})

        self.assertEqual(len(response["data"]), 2)
        self.assertEqual(response["data"][0]["url"], "https://example.test/1.png")
        sent_request = stream.call_args.args[0]
        self.assertEqual(sent_request.plugin, "ImageHelper")
        self.assertTrue(stream.call_args.kwargs["should_remove_conversation"])

    async def test_missing_image_event_is_an_error(self):
        async def events(*_args, **_kwargs):
            yield {"type": "finish", "finish_reason": "stop"}

        with (
            patch.object(generation, "create_conversation", AsyncMock(return_value="temporary-chat")),
            patch.object(generation, "create_completion_stream", side_effect=events),
        ):
            with self.assertRaises(generation.ImageGenerationError):
                await generation.generate_images(
                    ImageGenerationRequest(prompt="cat", n=1),
                    {"x-uskey": "test"},
                )

    async def test_base64_response_format_downloads_image(self):
        async def events(*_args, **_kwargs):
            yield {"type": "image", "images": [{"url": "https://example.test/1.png"}]}

        with (
            patch.object(generation, "create_conversation", AsyncMock(return_value="temporary-chat")),
            patch.object(generation, "create_completion_stream", side_effect=events),
            patch.object(generation, "_download_as_base64", AsyncMock(return_value="aW1hZ2U=")),
        ):
            response = await generation.generate_images(
                ImageGenerationRequest(prompt="cat", n=1, response_format="b64_json"),
                {"x-uskey": "test"},
            )
        self.assertEqual(response["data"], [{"b64_json": "aW1hZ2U="}])


class ImageHTTPTests(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[get_authorized_headers] = fake_headers
        app.dependency_overrides[require_api_key] = fake_api_key
        self.client = TestClient(app)

    def tearDown(self):
        app.dependency_overrides.clear()
        self.client.close()

    def test_images_endpoint(self):
        with patch.object(image_router, "generate_images", side_effect=fake_image_result):
            response = self.client.post(
                "/v1/images/generations",
                json={"model": "yuanbao-image", "prompt": "cat", "n": 2},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["data"]), 2)

    def test_official_openai_sdk_images(self):
        sdk_http = TestClient(app)
        sdk = OpenAI(base_url="http://testserver/v1", api_key="test-key", http_client=sdk_http)
        try:
            with patch.object(image_router, "generate_images", side_effect=fake_image_result):
                response = sdk.images.generate(model="yuanbao-image", prompt="cat", n=2)
            self.assertEqual(len(response.data), 2)
            self.assertEqual(response.data[0].url, "https://example.test/1.png")
        finally:
            sdk.close()

    def test_image_model_is_listed_and_retrievable(self):
        model_ids = {item["id"] for item in self.client.get("/v1/models").json()["data"]}
        self.assertIn("yuanbao-image", model_ids)
        self.assertEqual(self.client.get("/v1/models/yuanbao-image").status_code, 200)


if __name__ == "__main__":
    unittest.main()
