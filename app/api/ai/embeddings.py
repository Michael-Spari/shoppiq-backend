from fastapi import APIRouter, HTTPException
from typing import List
from openai import OpenAI
from app.config import settings

router = APIRouter()

# OpenAI Client initialisieren
client = OpenAI(api_key=settings.OPENAI_API_KEY)

@router.post("/embeddings")
async def get_embeddings(text: str):
    """
    Generiert Embeddings für den gegebenen Text mit OpenAI.
    Ersetzt die getEmbeddings Funktion aus dem Flutter Frontend.
    """
    try:
        # Embedding erstellen (OpenAI v1.x Syntax)
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        
        embedding = response.data[0].embedding
        token_count = response.usage.total_tokens
        
        return {
            "embedding": embedding,
            "token_count": token_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

@router.post("/embeddings/batch")
async def get_embeddings_batch(texts: List[str]):
    """
    Generiert Embeddings für mehrere Texte gleichzeitig.
    Effizienter für größere Datenmengen.
    """
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        
        embeddings = [item.embedding for item in response.data]
        token_count = response.usage.total_tokens
        
        return {
            "embeddings": embeddings,
            "token_count": token_count,
            "count": len(embeddings)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")