"""Smoke test using only the public OpenAI-compatible API."""

import os

from openai import OpenAI

client = OpenAI(
    base_url=os.getenv("YUANBAO_BASE_URL", "http://127.0.0.1:8000/v1"),
    api_key=os.getenv("YUANBAO_API_KEY", "sk-your-api-key-here"),
)

print("Models:")
for model in client.models.list().data:
    print(f"- {model.id}")

response = client.chat.completions.create(
    model="deepseek-v3",
    messages=[{"role": "user", "content": "只回复：非流式正常"}],
)
print("\nNon-streaming:", response.choices[0].message.content)

print("\nStreaming: ", end="", flush=True)
stream = client.chat.completions.create(
    model="deepseek-v3",
    messages=[{"role": "user", "content": "只回复：流式正常"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
