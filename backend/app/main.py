import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from datetime import datetime

from .config import settings
from .api.routes import router
from .api.middleware import setup_middleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Taqneeq Department Classifier...")
    
    try:
        # Validate data files exist
        import os
        if not os.path.exists(settings.DEPARTMENTS_FILE):
            raise FileNotFoundError(f"Departments file not found: {settings.DEPARTMENTS_FILE}")
        
        if not os.path.exists(settings.QUESTIONS_FILE):
            raise FileNotFoundError(f"Questions file not found: {settings.QUESTIONS_FILE}")
        
        logger.info("✅ Data files validated")
        logger.info("🎯 Taqneeq Department Classifier ready!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down...")

# Create FastAPI app
app = FastAPI(
    title="Taqneeq Department Classifier",
    description="Intelligent department matching for Taqneeq techfest",
    version="2.0.0",
    lifespan=lifespan
)

# Setup middleware
setup_middleware(app)

# Include routes
app.include_router(router, prefix="/api/v1", tags=["Classification"])

@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Taqneeq Department Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }
            .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
            .endpoint { background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 3px; }
            a { color: #667eea; text-decoration: none; }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🎯 Taqneeq Department Classifier</h1>
            <p>Intelligent department matching system</p>
        </div>
        
        <div class="section">
            <h2>🚀 API Endpoints</h2>
            <div class="endpoint"><strong>POST</strong> /api/classification/start - Start classification</div>
            <div class="endpoint"><strong>POST</strong> /api/classification/answer - Submit answers</div>
            <div class="endpoint"><strong>POST</strong> /api/classification/explanation - Get explanations</div>
            <div class="endpoint"><strong>GET</strong> /api/departments - List all departments</div>
        </div>
        
        <div class="section">
            <h2>📖 Documentation</h2>
            <p><a href="/docs" target="_blank">Interactive API Documentation</a></p>
            <p><a href="/api/health">Health Check</a></p>
        </div>
    </body>
    </html>
    """)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )