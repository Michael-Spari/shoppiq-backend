from fastapi import APIRouter, HTTPException
from typing import List
import openai
from app.config import settings

router = APIRouter()

@router.post("/embeddings")
async def get_embeddings(text: str):
    """
    Generiert Embeddings für den gegebenen Text mit OpenAI.
    Ersetzt die getEmbeddings Funktion aus dem Flutter Frontend.
    """
    try:
        # OpenAI Client konfigurieren
        openai.api_key = settings.OPENAI_API_KEY
        
        # Embedding erstellen (OpenAI v0.28.0 Syntax)
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        
        embedding = response['data'][0]['embedding']
        token_count = response['usage']['total_tokens']
        
        return {
            "embedding": embedding,
            "token_count": token_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")