import sys
import os
from pathlib import Path

# Ensure we can import the app
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    print("🚀 Starting Taqneeq Department Classifier...")
    print("🌐 Server: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    print("=" * 50)
    
    try:
        import uvicorn
        from app.main import app
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=False,  # Disable reload to avoid multiprocessing issues
            log_level="info"
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've installed: pip install fastapi uvicorn pydantic pydantic-settings")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()