from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import openai
from app.config import settings

router = APIRouter()

# Pydantic v1 Models
class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    token_count: int

@router.post("/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
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
            input=request.text
        )
        
        embedding = response['data'][0]['embedding']
        token_count = response['usage']['total_tokens']
        
        return EmbeddingResponse(
            embedding=embedding,
            token_count=token_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")