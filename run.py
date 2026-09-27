"""
SmartRAG entrypoint.

Starts the FastAPI backend. The Streamlit dashboard is a separate process
(Streamlit manages its own server loop) — start it with:

    streamlit run dashboard/app.py

Usage:
    python run.py
"""
import uvicorn

from backend.core.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "backend.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )
