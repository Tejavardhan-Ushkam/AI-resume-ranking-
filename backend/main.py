"""
Intelligent Candidate Discovery — FastAPI Backend
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config import settings
from models.database import init_db, get_chroma
from api.routes.candidates import router as candidates_router
from api.routes.search import router as search_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB, warm up embedding model, connect ChromaDB."""
    logger.info("🚀 Starting Intelligent Candidate Discovery")
    
    # Init SQLite
    init_db()
    
    # Init ChromaDB (creates dir if needed)
    get_chroma()
    
    # Warm up embedding model (first call loads model weights)
    logger.info("Warming up embedding model...")
    from core.signal_engine import embed_text
    embed_text("warmup text for embedding model initialization")
    logger.info("✅ Embedding model ready")
    
    logger.info(f"✅ Server ready at http://{settings.HOST}:{settings.PORT}")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Intelligent Candidate Discovery",
    description="""
## AI-Powered Candidate Ranking System

A novel multi-signal approach to candidate discovery featuring:
- **Deep JD Understanding**: Semantic decomposition of job descriptions
- **Career DNA Fingerprinting**: 6-axis candidate characterization
- **Ensemble Ranking**: Semantic + Skills + DNA + Behavioral signals
- **Explainable Results**: AI-generated per-candidate reasoning

### Quick Start
1. Upload candidates: `POST /candidates/bulk`
2. Search: `POST /search/`
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(candidates_router, prefix="/api")
app.include_router(search_router, prefix="/api")


@app.get("/health")
async def health():
    from models.database import count_candidates
    return {
        "status": "healthy",
        "candidates_indexed": count_candidates(),
        "embedding_model": settings.EMBEDDING_MODEL,
        "ai_enabled": bool(settings.ANTHROPIC_API_KEY),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )