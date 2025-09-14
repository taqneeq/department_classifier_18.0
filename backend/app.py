import os
import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# FastAPI and related imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Environment and logging setup
from dotenv import load_dotenv
import uvicorn

# Internal imports
from api.middleware import setup_middleware
from core.data_loader import load_departments, load_questions, validate_data_integrity
from utils.validation import ValidationError

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging():
    """Configure application logging"""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("taqneeq_classifier.log") if os.getenv("LOG_FILE") else logging.NullHandler()
        ]
    )
    
    # Set specific loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured at {log_level} level")
    return logger

logger = setup_logging()

# Application configuration
class Config:
    """Application configuration from environment variables"""
    
    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    RELOAD: bool = os.getenv("RELOAD", "false").lower() == "true"
    
    # Classification settings
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.85))
    SECONDARY_THRESHOLD: float = float(os.getenv("SECONDARY_CONFIDENCE_THRESHOLD", 0.70))
    MAX_QUESTIONS: int = int(os.getenv("MAX_QUESTIONS", 15))
    MIN_QUESTIONS: int = int(os.getenv("MIN_QUESTIONS", 6))
    SMOOTHING_ALPHA: float = float(os.getenv("SMOOTHING_ALPHA", 0.1))
    
    # RAG settings
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))
    MAX_CHUNKS_PER_DEPT: int = int(os.getenv("MAX_CHUNKS_PER_DEPT", 5))
    
    # Data paths
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    DEPARTMENTS_FILE: str = os.getenv("DEPARTMENTS_FILE", "data/departments.json")
    QUESTIONS_FILE: str = os.getenv("QUESTIONS_FILE", "data/question_bank.json")
    PDF_FILE: str = os.getenv("PDF_FILE", "data/departments.pdf")
    
    # Security
    ALLOWED_ORIGINS: list = os.getenv("ALLOWED_ORIGINS", "*").split(",")
    API_KEY: Optional[str] = os.getenv("API_KEY")  # Optional API key for admin endpoints

# Global configuration instance
config = Config()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown tasks"""
    
    # Startup
    logger.info("Starting Taqneeq Department Classifier...")
    logger.info(f"Configuration: DEBUG={config.DEBUG}, PORT={config.PORT}")
    
    try:
        # Validate data files exist
        data_files = [config.DEPARTMENTS_FILE, config.QUESTIONS_FILE]
        missing_files = [f for f in data_files if not os.path.exists(f)]
        
        if missing_files:
            logger.error(f"Missing required data files: {missing_files}")
            raise FileNotFoundError(f"Required data files not found: {missing_files}")
        
        # Load and validate data
        logger.info("Loading and validating data...")
        departments = load_departments(config.DEPARTMENTS_FILE)
        questions, seed_questions = load_questions(config.QUESTIONS_FILE)
        
        is_valid, errors = validate_data_integrity(departments, questions)
        if not is_valid:
            logger.error(f"Data validation failed: {errors}")
            raise ValidationError("Data validation failed", errors)
        
        logger.info(f"Loaded {len(departments)} departments and {len(questions)} questions")
        
        
        
        # Check RAG system
        if config.OPENAI_API_KEY:
            logger.info("OpenAI API key found - RAG explanations enabled")
        else:
            logger.warning("No OpenAI API key - RAG explanations will use fallback")
        
        logger.info("Taqneeq Department Classifier started successfully!")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down Taqneeq Department Classifier...")

# Create FastAPI application
def create_taqneeq_app() -> FastAPI:
    """Create and configure the main FastAPI application"""
    
    app = FastAPI(
        title="Taqneeq Department Classifier",
        description="""
        **Intelligent Department Matching for Taqneeq Techfest**
        
        This system uses adaptive Bayesian inference to match students with their ideal 
        Taqneeq department based on their interests, skills, and preferences.
        
        ## Features
        - **Adaptive Questioning**: Information gain driven question selection
        - **Trait-Based Matching**: 20 multi-departmental traits for precise classification  
        - **Intelligent Explanations**: RAG-powered explanations using actual department data
        - **Real-time Progress**: Session state tracking and progress monitoring
        
        ## Quick Start
        1. **Start Session**: `POST /api/v1/classification/start`
        2. **Answer Questions**: `POST /api/v1/classification/answer` (8-12 times typically)
        3. **Get Results**: Classification complete when `is_complete: true`
        4. **Get Explanation**: `POST /api/v1/classification/explanation` for detailed insights
        
        ## Support
        For questions about Taqneeq departments or technical issues, contact the Taqneeq organizing team.
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if config.DEBUG else None,
        redoc_url="/redoc" if config.DEBUG else None
    )
    
    # Setup middleware
    setup_middleware(app)
    
    # Import and include routers
    from api.routes import classification, departments, admin
    
    app.include_router(
        classification.router, 
        prefix="/api/v1", 
        tags=["Classification"],
        responses={
            404: {"description": "Session not found"},
            400: {"description": "Invalid request data"},
            500: {"description": "Internal server error"}
        }
    )
    
    app.include_router(
        departments.router, 
        prefix="/api/v1", 
        tags=["Departments"]
    )
    
    app.include_router(
        admin.router, 
        prefix="/api/v1", 
        tags=["Administration"]
    )
    
    return app

# Create the app instance
app = create_taqneeq_app()

# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with basic information"""
    html_content = """
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
            <p>Intelligent department matching system for Taqneeq techfest participants</p>
        </div>
        
        <div class="section">
            <h2>📋 Quick Start</h2>
            <p>Use the classification API to find your perfect Taqneeq department match:</p>
            <div class="endpoint"><strong>POST</strong> /api/v1/classification/start - Start classification</div>
            <div class="endpoint"><strong>POST</strong> /api/v1/classification/answer - Submit answers</div>
            <div class="endpoint"><strong>POST</strong> /api/v1/classification/explanation - Get detailed results</div>
        </div>
        
        <div class="section">
            <h2>🏢 Departments Available</h2>
            <p>14 specialized departments including Events, Digital Creatives, Tech & Collab, Marketing, and more.</p>
            <div class="endpoint"><strong>GET</strong> /api/v1/departments - List all departments</div>
            <div class="endpoint"><strong>GET</strong> /api/v1/departments/search?q=tech - Search departments</div>
        </div>
        
        <div class="section">
            <h2>📚 API Documentation</h2>
            <p>
                <a href="/docs" target="_blank">📖 Interactive API Documentation (Swagger UI)</a><br>
                <a href="/redoc" target="_blank">📄 Alternative Documentation (ReDoc)</a>
            </p>
        </div>
        
        <div class="section">
            <h2>⚡ System Status</h2>
            <p>
                <a href="/api/v1/health">🔍 Health Check</a><br>
                <a href="/api/v1/stats">📊 Usage Statistics</a>
            </p>
        </div>
        
        <div class="section">
            <h2>🔬 How It Works</h2>
            <p>This system uses:</p>
            <ul>
                <li><strong>Bayesian Inference</strong> - Adaptive belief updating based on responses</li>
                <li><strong>Information Theory</strong> - Questions selected for maximum information gain</li>
                <li><strong>Trait Vectors</strong> - 20 multi-departmental traits for precise matching</li>
                <li><strong>RAG System</strong> - LangChain-powered explanations using department documentation</li>
            </ul>
        </div>
        
        <footer style="text-align: center; margin-top: 40px; color: #666;">
            <p>Taqneeq Department Classifier v1.0.0 | Built for Taqneeq Techfest</p>
        </footer>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Global exception handler
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors globally"""
    logger.warning(f"Validation error on {request.url}: {exc}")
    return JSONResponse(
        status_code=400,
        content={
            "error_type": "validation_error",
            "message": str(exc),
            "details": exc.errors if hasattr(exc, 'errors') else None
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors globally"""
    logger.error(f"Unexpected error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error_type": "internal_error",
            "message": "An unexpected error occurred",
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )

# Custom startup event for additional initialization
@app.on_event("startup")
async def startup_event():
    """Additional startup tasks"""
    logger.info("Running additional startup tasks...")
    
    # Create data directory if it doesn't exist
    os.makedirs(config.DATA_DIR, exist_ok=True)
    
    # Log configuration
    logger.info(f"Server will run on {config.HOST}:{config.PORT}")
    logger.info(f"Debug mode: {config.DEBUG}")
    logger.info(f"Data directory: {config.DATA_DIR}")

def main():
    """Main entry point for running the application"""
    
    print("🎯 Starting Taqneeq Department Classifier...")
    print(f"📊 Classification System: Bayesian inference with information gain")
    print(f"🧠 Features: 20 traits, adaptive questioning, RAG explanations")
    print(f"🌐 Server: http://{config.HOST}:{config.PORT}")
    print(f"📖 API Docs: http://{config.HOST}:{config.PORT}/docs")
    print("=" * 60)
    
    try:
        # Run the server
        uvicorn.run(
            "app:app",
            host=config.HOST,
            port=config.PORT,
            reload=config.RELOAD and config.DEBUG,
            log_level="info" if config.DEBUG else "warning",
            access_log=config.DEBUG,
            workers=1  # Single worker for session management
        )
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()