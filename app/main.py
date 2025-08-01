from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.ai import embeddings
from app.api.v1 import chat  # Chat für bestehende Listen

app = FastAPI(title="ShoppiQ Backend", version="1.0.0")

# CORS für Flutter-App
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routers registrieren
app.include_router(embeddings.router, prefix="/api/ai", tags=["AI"])
app.include_router(chat.router, prefix="/api/v1", tags=["Shopping Chat"])  # Angepasst

@app.get("/")
async def root():
    return {"message": "ShoppiQ Backend is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}