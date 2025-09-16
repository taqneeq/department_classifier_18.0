import time
import logging
import uuid
from typing import Callable
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    """Enhanced logging middleware with request tracking"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID for tracking
        request_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        # Log incoming request
        logger.info(
            f"Request {request_id}: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Add request ID to state for use in endpoints
        request.state.request_id = request_id
        
        try:
            # Process request
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log successful response
            logger.info(
                f"Response {request_id}: {response.status_code} "
                f"in {duration:.3f}s"
            )
            
            # Add useful headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{duration:.3f}s"
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Error {request_id}: {str(e)} "
                f"after {duration:.3f}s",
                exc_info=True
            )
            raise

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response

def setup_middleware(app: FastAPI) -> None:
    """Setup all middleware for the application"""
    from ..config import settings
    
    # CORS middleware (must be added first)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time"]
    )
    
    # Compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Custom middleware (order matters)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(LoggingMiddleware)
    
    logger.info("Middleware setup complete")
