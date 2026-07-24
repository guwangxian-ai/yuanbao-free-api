# YuanBao Free API

将腾讯元宝网页能力封装为 OpenAI 兼容接口，可供 Cherry Studio、OpenAI SDK 或其他兼容客户端调用。

本仓库基于 [chenwr727/yuanbao-free-api](https://github.com/chenwr727/yuanbao-free-api) 调整，增加了以下实用修复：

- 提供 `GET /v1/models`，兼容 Cherry Studio 获取模型列表。
- 将元宝流式事件转换为标准 OpenAI SSE 文本，不再显示 `{"type":"text"...}` 原始数据。
- 修复新版元宝登录页的二维码定位与下载。
- Windows 控制台无法输出二维码字符时，仍会保存 `qrcode.png` 供扫码。

> 本项目仅供学习和研究。网页接口可能随时变化，频繁或不当使用可能导致账号受限。请遵守腾讯元宝的服务条款。

## 环境要求

- Python 3.10 或更高版本
- 可访问腾讯元宝的网络
- 微信，用于首次扫码登录
- Git

## 快速开始

### Windows PowerShell

```powershell
git clone https://github.com/guwangxian-ai/yuanbao-free-api.git
cd yuanbao-free-api

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m playwright install chromium

Copy-Item .env.example .env
python app.py
```

如果 PowerShell 禁止执行激活脚本，可以不激活虚拟环境，直接运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m playwright install chromium
Copy-Item .env.example .env
.\.venv\Scripts\python.exe app.py
```

### macOS / Linux

```bash
git clone https://github.com/guwangxian-ai/yuanbao-free-api.git
cd yuanbao-free-api

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m playwright install chromium

cp .env.example .env
python app.py
```

服务默认运行在 `http://127.0.0.1:8000`，API 文档位于 `http://127.0.0.1:8000/docs`。

## 配置

复制 `.env.example` 为 `.env`，然后按需修改：

```dotenv
# 这是本地 API 的访问密钥，可以自行更换；多个密钥用逗号分隔
API_KEYS=sk-your-api-key-here,sk-another-api-key

# 扫码等待时间，单位为毫秒；可选
LOGIN_TIMEOUT=300000
```

`.env` 不会被 Git 提交。修改密钥后需要重启服务。

## 扫码登录

首次启动时，程序会打开无头 Chromium 并生成登录二维码：

1. 等待项目目录中出现 `qrcode.png`。
2. 用微信扫描二维码并确认登录。
3. 看到日志 `扫码成功` 和 `Application startup complete` 后再调用 API。

二维码过期、登录态失效或需要换账号时，停止并重新启动服务，再扫描新生成的 `qrcode.png`。

## Cherry Studio 配置

在“OpenAI 兼容”服务中填写：

| 配置项 | 值 |
| --- | --- |
| API 地址 | `http://127.0.0.1:8000` |
| API 密钥 | `sk-your-api-key-here`，或你在 `.env` 中设置的密钥 |
| 模型 ID | `deepseek-v3` |

注意：

- API 地址不要填写 `/docs`。
- API 地址不要填写完整的 `/v1/chat/completions`。
- Cherry Studio 会自动拼接 `/v1/models` 和 `/v1/chat/completions`。

保存后点击“检测”或“获取模型列表”，再开始对话。

## API 测试

### 查看模型列表

浏览器打开：

```text
http://127.0.0.1:8000/v1/models
```

PowerShell：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/models `
  -Headers @{ Authorization = "Bearer sk-your-api-key-here" }
```

### 测试聊天

PowerShell：

```powershell
$body = @{
  model = "deepseek-v3"
  messages = @(
    @{ role = "user"; content = "你好，请用一句话介绍自己" }
  )
} | ConvertTo-Json -Depth 5

Invoke-WebRequest http://127.0.0.1:8000/v1/chat/completions `
  -Method POST `
  -Headers @{ Authorization = "Bearer sk-your-api-key-here" } `
  -ContentType "application/json" `
  -Body $body
```

curl：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-your-api-key-here" \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v3","messages":[{"role":"user","content":"你好"}]}'
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="sk-your-api-key-here",
)

stream = client.chat.completions.create(
    model="deepseek-v3",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## 支持的模型

| 模型 ID | 说明 |
| --- | --- |
| `deepseek-v3` | DeepSeek V3 |
| `deepseek-r1` | DeepSeek R1 |
| `deepseek-v3-search` | DeepSeek V3，启用联网搜索 |
| `deepseek-r1-search` | DeepSeek R1，启用联网搜索 |
| `hunyuan` | 腾讯混元 |
| `hunyuan-t1` | 腾讯混元 T1 |
| `hunyuan-search` | 腾讯混元，启用联网搜索 |
| `hunyuan-t1-search` | 腾讯混元 T1，启用联网搜索 |

模型由腾讯元宝网页端提供，实际可用性可能随账号权限和网页端调整而变化。

## 常见问题

### Cherry Studio 提示 `AI_APICallError: Not Found`

通常是 API 地址填写错误。应填写 `http://127.0.0.1:8000`，不能带 `/docs` 或 `/v1/chat/completions`。

### 返回 `401 Unauthorized`

Cherry Studio 中的密钥必须和 `.env` 中 `API_KEYS` 的某一项完全一致。修改 `.env` 后重启服务。

### 返回原始 `{"type":"text"}` JSON

请确认使用的是本仓库最新代码，并重启服务。此分支已将元宝内部事件转换为标准 OpenAI 流式文本。

### API 无响应或浏览器初始化失败

- 确认已运行 `python -m playwright install chromium`。
- 查看启动窗口中的错误日志。
- 删除旧的 `qrcode.png` 不是必需的，重启时会自动覆盖。
- 确认扫码后微信端已完成登录确认。

### 端口 8000 被占用

可以改用其他端口启动：

```powershell
python -m uvicorn app:app --host 0.0.0.0 --port 8001
```

客户端 API 地址相应改为 `http://127.0.0.1:8001`。

## 接口

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/v1/models` | 获取模型列表 |
| `POST` | `/v1/chat/completions` | OpenAI 兼容聊天流 |
| `POST` | `/v1/upload` | 上传图片或文件 |
| `GET` | `/docs` | FastAPI 在线接口文档 |

## 安全说明

- 不要提交 `.env`、`qrcode.png`、日志或浏览器登录数据。
- 不要将服务直接暴露到公网；如确需远程访问，请使用强密钥、HTTPS、访问控制和防火墙。
- API 密钥只保护这个本地代理，并不是腾讯元宝官方密钥。

## 致谢

- 原项目：[chenwr727/yuanbao-free-api](https://github.com/chenwr727/yuanbao-free-api)
- [Tencent YuanBao](https://yuanbao.tencent.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Playwright](https://playwright.dev/)

## License

沿用原项目的 MIT License。
