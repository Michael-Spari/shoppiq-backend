from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import firebase_admin
from firebase_admin import credentials
import json

# API Routes
from app.api.ai import embeddings
from app.api.v1 import chat, generate_shopping_list

# Config
from app.core.config import settings

# FastAPI App
app = FastAPI(
    title="ShoppiQ Backend",
    description="AI-Powered Shopping Assistant Backend",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In Production: spezifische Domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Firebase Initialization
def initialize_firebase():
    try:
        if not firebase_admin._apps:
            # Firebase Credentials aus Environment Variables
            firebase_config = {
                "type": "service_account",
                "project_id": settings.FIREBASE_PROJECT_ID,
                "private_key_id": settings.FIREBASE_PRIVATE_KEY_ID,
                "private_key": settings.FIREBASE_PRIVATE_KEY,
                "client_email": settings.FIREBASE_CLIENT_EMAIL,
                "client_id": settings.FIREBASE_CLIENT_ID,
                "auth_uri": settings.FIREBASE_AUTH_URI,
                "token_uri": settings.FIREBASE_TOKEN_URI,
                "client_x509_cert_url": settings.FIREBASE_CLIENT_CERT_URL
            }
            
            cred = credentials.Certificate(firebase_config)
            firebase_admin.initialize_app(cred)
            print("✅ Firebase initialized successfully")
    except Exception as e:
        print(f"❌ Firebase initialization error: {e}")

# Initialize Firebase on startup
@app.on_event("startup")
async def startup_event():
    initialize_firebase()

# Include Routers
app.include_router(embeddings.router, prefix="/api/ai", tags=["AI"])
app.include_router(chat.router, prefix="/api/v1", tags=["Shopping Chat"])
app.include_router(generate_shopping_list.router, prefix="/api/v1", tags=["Shopping List"])

# Root Endpoints
@app.get("/")
async def root():
    return {
        "message": "ShoppiQ Backend is running!",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "api": "running",
            "firebase": "connected",
            "ai": "available"
        }
    }

# Debug Info (nur für Development)
@app.get("/debug/config")
async def debug_config():
    if not settings.DEBUG:
        return {"message": "Debug mode disabled"}
    
    return {
        "pinecone_configured": bool(settings.PINECONE_API_KEY),
        "openai_configured": bool(settings.OPENAI_API_KEY),
        "firebase_configured": bool(settings.FIREBASE_PROJECT_ID),
        "pinecone_environment": settings.PINECONE_ENVIRONMENT,
        "pinecone_index": settings.PINECONE_INDEX_NAME
    }