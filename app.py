"""YuanBao API Proxy 主应用"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.routers import chat, image, upload
from src.services.browser import browser_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """应用生命周期事件处理器"""
    logger.info("[Startup] 正在初始化浏览器...")
    try:
        await browser_manager.login()
        logger.info("[Startup] 浏览器初始化完成")
    except Exception as e:
        logger.error(f"[Startup] 浏览器初始化失败: {e}")

    yield

    logger.info("[Shutdown] 正在关闭浏览器...")
    try:
        await browser_manager.close()
        logger.info("[Shutdown] 浏览器已关闭")
    except Exception as e:
        logger.error(f"[Shutdown] 关闭浏览器失败: {e}")


app = FastAPI(title="YuanBao API Proxy", version="1.0.0", lifespan=lifespan)


def openai_error_response(status_code: int, message: str, error_type: str, code: str):
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": code,
            }
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    message = "; ".join(error.get("msg", "Invalid request") for error in exc.errors())
    return openai_error_response(400, message, "invalid_request_error", "invalid_request")


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    code = "invalid_request"
    error_type = "invalid_request_error"
    if exc.status_code in {401, 403}:
        code = "invalid_api_key"
        error_type = "authentication_error"
    elif exc.status_code == 404:
        code = "model_not_found"
    elif exc.status_code == 429:
        code = "rate_limit_exceeded"
        error_type = "rate_limit_error"
    elif exc.status_code >= 500:
        code = "upstream_error" if exc.status_code == 502 else "server_error"
        error_type = "api_error"
    return openai_error_response(exc.status_code, str(exc.detail), error_type, code)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def readiness():
    if not browser_manager.is_logged_in:
        return JSONResponse(status_code=503, content={"status": "not_ready", "reason": "yuanbao_login_required"})
    return {"status": "ready"}

app.include_router(chat.router)
app.include_router(image.router)
app.include_router(upload.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
