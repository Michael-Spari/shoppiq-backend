from openai import OpenAI
from typing import List
from app.config import settings

class OpenAIService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    async def get_embeddings(self, text: str) -> List[float]:
        """Text zu Embeddings konvertieren"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            raise Exception(f"OpenAI Embedding Error: {str(e)}")

# Service Instanz
openai_service = OpenAIService()