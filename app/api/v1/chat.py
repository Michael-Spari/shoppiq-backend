from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from openai import OpenAI
from app.config import settings
import json

router = APIRouter()
client = OpenAI(api_key=settings.OPENAI_API_KEY)

class ChatRequest(BaseModel):
    message: str
    settings: Dict[str, Any]  # Produkte und Supermärkte

class ShoppingListItem(BaseModel):
    name: str
    quantity: int
    note: str = ""
    supermarkt: str

@router.post("/shopping-list", response_model=List[ShoppingListItem])
async def generate_shopping_list(request: ChatRequest):
    """
    Generiert eine Einkaufsliste basierend auf der Benutzeranfrage.
    Ersetzt ChatShoppingListService aus Flutter.
    """
    try:
        prompt = f'''
Du bist ein intelligenter Einkaufslisten-Planer. Hier sind die verfügbaren Produkte und Supermärkte als JSON:
{json.dumps(request.settings)}
Erstelle eine Einkaufsliste für die Anfrage: "{request.message}".
Wähle ausschließlich Produkte aus der Produktliste und ordne sie passenden Supermärkten zu. Beachte die Präferenzen (z.B. vegan, glutenfrei).
Antworte ausschließlich mit einer JSON-Liste im Format:
[{{"name":"Produktname","quantity":1,"note":"optional","supermarkt":"Supermarktname"}}]
'''

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Du bist ein Einkaufslisten-Generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )
        
        # JSON Response parsen
        content = response.choices[0].message.content
        shopping_list = json.loads(content)
        
        return [ShoppingListItem(**item) for item in shopping_list]
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="KI hat ungültiges JSON zurückgegeben")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler beim Generieren der Einkaufsliste: {str(e)}")