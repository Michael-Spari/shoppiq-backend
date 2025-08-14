import os
from typing import Optional

class Settings:
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Pinecone
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")  # ✅ NEU
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "shoppiq-products")
    
    # Firebase
    FIREBASE_PROJECT_ID: str = os.getenv("FIREBASE_PROJECT_ID", "")
    FIREBASE_PRIVATE_KEY_ID: str = os.getenv("FIREBASE_PRIVATE_KEY_ID", "")
    FIREBASE_PRIVATE_KEY: str = os.getenv("FIREBASE_PRIVATE_KEY", "").replace('\\n', '\n')
    FIREBASE_CLIENT_EMAIL: str = os.getenv("FIREBASE_CLIENT_EMAIL", "")
    FIREBASE_CLIENT_ID: str = os.getenv("FIREBASE_CLIENT_ID", "")
    FIREBASE_AUTH_URI: str = os.getenv("FIREBASE_AUTH_URI", "https://accounts.google.com/o/oauth2/auth")
    FIREBASE_TOKEN_URI: str = os.getenv("FIREBASE_TOKEN_URI", "https://oauth2.googleapis.com/token")
    FIREBASE_CLIENT_CERT_URL: str = os.getenv("FIREBASE_CLIENT_CERT_URL", "")
    
    # App
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    PORT: int = int(os.getenv("PORT", 8000))

settings = Settings()