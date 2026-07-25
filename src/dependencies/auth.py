"""认证依赖模块"""

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.config import validate_api_key
from src.utils.common import generate_headers

bearer_scheme = HTTPBearer(auto_error=False)


async def require_api_key(
    authorization: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """Validate the local OpenAI-compatible bearer token."""
    if not authorization or not authorization.credentials:
        raise HTTPException(status_code=401, detail="need token")

    token = authorization.credentials

    if not validate_api_key(token):
        raise HTTPException(status_code=403, detail="invalid api_key")

    return token


async def get_authorized_headers(
    _: str = Depends(require_api_key),
):
    """Return fresh Yuanbao authentication headers after local API auth."""

    return await generate_headers()
