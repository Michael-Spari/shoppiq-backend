import httpx
import json
from typing import List, Dict, Any, Optional
from app.config import settings
from app.models.shopping import ShoppingItem, Supermarket, ShoppingList, Recipe, CookingPlan

class PineconeService:
    def __init__(self):
        self.api_key = settings.PINECONE_API_KEY
        self.api_url = settings.PINECONE_API_URL
        
    async def _make_request(self, method: str, endpoint: str, data: Dict = None, namespace: str = None) -> Dict:
        """Helper für Pinecone API Requests"""
        url = f"{self.api_url}{endpoint}"
        if namespace:
            url += f"?namespace={namespace}"
            
        headers = {
            'Api-Key': self.api_key,
            'Content-Type': 'application/json',
        }
        
        async with httpx.AsyncClient() as client:
            if method == "POST":
                response = await client.post(url, headers=headers, json=data)
            elif method == "DELETE":
                response = await client.delete(url, headers=headers, json=data)
            else:
                response = await client.get(url, headers=headers)
                
            response.raise_for_status()
            return response.json()

    # --- Vector CRUD Operations ---
    async def upsert_vector(self, vector_id: str, embedding: List[float], metadata: Dict, namespace: str) -> bool:
        """Vektor in Pinecone einfügen oder aktualisieren"""
        try:
            data = {
                "vectors": [{
                    "id": vector_id,
                    "values": embedding,
                    "metadata": metadata
                }],
                "namespace": namespace
            }
            await self._make_request("POST", "/vectors/upsert", data)
            return True
        except Exception as e:
            print(f"Pinecone upsert error: {e}")
            return False

    async def delete_vector(self, vector_id: str, namespace: str) -> bool:
        """Vektor aus Pinecone löschen"""
        try:
            data = {"ids": [vector_id], "namespace": namespace}
            await self._make_request("POST", "/vectors/delete", data)
            return True
        except Exception as e:
            print(f"Pinecone delete error: {e}")
            return False

    async def query_vectors(self, query_vector: List[float], top_k: int, namespace: str, filter_dict: Dict = None) -> List[Dict]:
        """Ähnliche Vektoren suchen"""
        try:
            data = {
                "vector": query_vector,
                "topK": top_k,
                "includeValues": False,
                "includeMetadata": True,
                "namespace": namespace
            }
            if filter_dict:
                data["filter"] = filter_dict
                
            result = await self._make_request("POST", "/query", data)
            return result.get("matches", [])
        except Exception as e:
            print(f"Pinecone query error: {e}")
            return []

    # --- Shopping Item Operations ---
    async def add_shopping_vector(self, embedding: List[float], item: ShoppingItem, user_email: str) -> bool:
        """Shopping Item zu Pinecone hinzufügen"""
        metadata = {
            "user": user_email,
            "type": "shopping_item",
            "name": item.name,
            "uuid": item.uuid,
            "category": item.category,
            "quantity": item.quantity
        }
        return await self.upsert_vector(item.uuid, embedding, metadata, user_email)

    async def delete_shopping_vector(self, item_id: str, user_email: str) -> bool:
        """Shopping Item aus Pinecone löschen"""
        return await self.delete_vector(item_id, user_email)

    # --- Supermarket Operations ---
    async def add_supermarket_vector(self, embedding: List[float], market: Supermarket, user_email: str) -> bool:
        """Supermarkt zu Pinecone hinzufügen"""
        metadata = {
            "user": user_email,
            "type": "supermarket",
            "name": market.name,
            "uuid": market.uuid,
            "placeId": market.placeId,
            "address": market.address
        }
        return await self.upsert_vector(market.uuid, embedding, metadata, user_email)

    async def delete_supermarket_vector(self, market_id: str, user_email: str) -> bool:
        """Supermarkt aus Pinecone löschen"""
        return await self.delete_vector(market_id, user_email)

    # --- Recipe Operations ---
    async def add_recipe_vector(self, embedding: List[float], recipe: Recipe, user_email: str) -> bool:
        """Rezept zu Pinecone hinzufügen"""
        metadata = {
            "user": user_email,
            "type": "recipe",
            "name": recipe.name,
            "uuid": recipe.uuid,
            "category": recipe.category,
            "ingredients": json.dumps(recipe.ingredients)
        }
        return await self.upsert_vector(recipe.uuid, embedding, metadata, user_email)

    async def delete_recipe_vector(self, recipe_id: str, user_email: str) -> bool:
        """Rezept aus Pinecone löschen"""
        return await self.delete_vector(recipe_id, user_email)

    # --- Query Operations ---
    async def get_similar_items(self, query: str, user_email: str, item_type: str = None) -> List[Dict]:
        """Ähnliche Items basierend auf Text-Query finden"""
        from app.services.openai_service import OpenAIService
        
        openai_service = OpenAIService()
        query_embedding = await openai_service.get_embeddings(query)
        
        filter_dict = {"user": user_email}
        if item_type:
            filter_dict["type"] = item_type
            
        return await self.query_vectors(query_embedding, 10, user_email, filter_dict)

    async def get_all_items_for_user(self, user_email: str, item_type: str = None) -> List[Dict]:
        """Alle Items eines Users abrufen"""
        dummy_vector = [0.0] * 1536  # OpenAI Embedding Dimension
        
        filter_dict = {"user": user_email}
        if item_type:
            filter_dict["type"] = item_type
            
        return await self.query_vectors(dummy_vector, 100, user_email, filter_dict)

# Service Instanz
pinecone_service = PineconeService()