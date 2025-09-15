import os
import uvicorn
from app import app  # Import your FastAPI app instance

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render injects PORT at runtime
    uvicorn.run(app, host="0.0.0.0", port=port)
