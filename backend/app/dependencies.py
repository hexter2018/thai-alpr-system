"""
FastAPI Dependencies
"""
from typing import Generator, Optional
from fastapi import Header, HTTPException, Depends
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

from .database import get_db, get_async_db
from .config import get_settings


# Database dependencies are already defined in database.py
# Re-export for convenience
def get_database() -> Generator[Session, None, None]:
    """Get synchronous database session"""
    return get_db()


async def get_async_database() -> AsyncSession:
    """Get asynchronous database session"""
    async for session in get_async_db():
        yield session


# API Key authentication
def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """
    Verify API key from header
    
    Args:
        x_api_key: API key from X-API-Key header
    
    Returns:
        API key if valid
    
    Raises:
        HTTPException: If API key is invalid
    """
    settings = get_settings()
    
    # If no API key configured, skip validation
    if not settings.API_KEY:
        return "no-key-configured"
    
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide X-API-Key header."
        )
    
    if x_api_key != settings.API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    
    return x_api_key


# Optional API key (for public endpoints)
def optional_api_key(x_api_key: Optional[str] = Header(None)) -> Optional[str]:
    """
    Optional API key verification
    
    Args:
        x_api_key: API key from header
    
    Returns:
        API key if provided and valid, None otherwise
    """
    settings = get_settings()
    
    if not settings.API_KEY:
        return None
    
    if x_api_key and x_api_key == settings.API_KEY:
        return x_api_key
    
    return None


# Rate limiting
class RateLimiter:
    """Simple rate limiter using Redis"""
    
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
    
    async def __call__(self, x_forwarded_for: Optional[str] = Header(None)):
        """Check rate limit"""
        from .services.redis_service import get_redis
        
        settings = get_settings()
        
        if not settings.RATE_LIMIT_ENABLED:
            return True
        
        # Get client IP
        client_ip = x_forwarded_for or "unknown"
        
        try:
            redis = await get_redis()
            
            # Rate limit key
            key = f"rate_limit:{client_ip}"
            
            # Get current count
            current = await redis.get_counter(key)
            
            if current >= self.max_requests:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds} seconds."
                )
            
            # Increment counter
            new_count = await redis.increment_counter(key)
            
            # Set expiry on first request
            if new_count == 1:
                await redis.client.expire(key, self.window_seconds)
            
            return True
            
        except HTTPException:
            raise
        except Exception:
            # On Redis failure, allow request
            return True


# Create rate limiter instance
rate_limiter = RateLimiter(max_requests=60, window_seconds=60)


# Pagination
class PaginationParams:
    """Pagination parameters"""
    
    def __init__(self, skip: int = 0, limit: int = 20):
        if skip < 0:
            raise HTTPException(status_code=400, detail="skip must be >= 0")
        if limit < 1 or limit > 100:
            raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
        
        self.skip = skip
        self.limit = limit


def get_pagination(skip: int = 0, limit: int = 20) -> PaginationParams:
    """Get pagination parameters"""
    return PaginationParams(skip, limit)


# Current user (placeholder for future auth)
class CurrentUser:
    """Current authenticated user"""
    
    def __init__(self, user_id: Optional[str] = None, username: Optional[str] = None):
        self.user_id = user_id or "system"
        self.username = username or "system"
        self.is_authenticated = user_id is not None


def get_current_user(
    x_user_id: Optional[str] = Header(None),
    x_username: Optional[str] = Header(None)
) -> CurrentUser:
    """
    Get current user from headers
    (Placeholder for future JWT/OAuth implementation)
    """
    return CurrentUser(user_id=x_user_id, username=x_username)