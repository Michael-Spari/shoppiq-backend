from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.ai import embeddings

app = FastAPI(title="ShoppiQ Backend", version="1.0.0")

# CORS für Flutter-App
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Produktion spezifischer setzen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routers registrieren
app.include_router(embeddings.router, prefix="/api/ai", tags=["AI"])

@app.get("/")
async def root():
    return {"message": "ShoppiQ Backend is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}