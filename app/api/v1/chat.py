from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from openai import OpenAI
from app.config import settings
from app.models.shopping import ShoppingItem
import json

router = APIRouter()
client = OpenAI(api_key=settings.OPENAI_API_KEY)

class ChatMessage(BaseModel):
    role: str  # "user" oder "assistant"
    content: str

class ShoppingListChatRequest(BaseModel):
    message: str
    shopping_list: List[Dict[str, Any]]  # Bestehende Einkaufsliste
    chat_history: List[ChatMessage] = []  # Chat-Verlauf
    user_email: str

class ShoppingListChatResponse(BaseModel):
    response: str
    updated_list: Optional[List[Dict[str, Any]]] = None  # Falls Liste geändert wurde
    action_performed: Optional[str] = None  # "added", "removed", "modified", "none"

@router.post("/shopping-list-chat", response_model=ShoppingListChatResponse)
async def chat_about_shopping_list(request: ShoppingListChatRequest):
    """
    Chat-Service für Fragen zu bestehenden Einkaufslisten.
    Kann Items hinzufügen, entfernen, modifizieren oder Fragen beantworten.
    """
    try:
        # Aktuelle Einkaufsliste als String formatieren
        list_text = "\n".join([
            f"- {item['name']} (Menge: {item.get('quantity', 1)}, Supermarkt: {item.get('supermarkt', 'unbekannt')}, Status: {'✓ gekauft' if item.get('isChecked', False) else '○ offen'})"
            for item in request.shopping_list
        ])
        
        # System Prompt für Shopping-List Chat
        system_prompt = f"""
Du bist ein intelligenter Assistent für Einkaufslisten-Management. 

AKTUELLE EINKAUFSLISTE:
{list_text}

Du kannst:
1. Fragen zur Liste beantworten
2. Items hinzufügen/entfernen/ändern
3. Einkaufstipps geben
4. Kosten schätzen
5. Rezeptvorschläge basierend auf den Produkten machen

Wenn der User die Liste ändern möchte:
- Antworte mit der aktualisierten Liste im JSON-Format
- Erkläre was du geändert hast

Wenn der User nur eine Frage stellt:
- Beantworte sie basierend auf der aktuellen Liste
- Gib hilfreiche Tipps

Sei freundlich und hilfsbereit!
"""

        # Chat-Verlauf aufbauen
        messages = [{"role": "system", "content": system_prompt}]
        
        # Bisherige Chat-Historie hinzufügen
        for msg in request.chat_history[-5:]:  # Nur letzte 5 Nachrichten
            messages.append({"role": msg.role, "content": msg.content})
        
        # Aktuelle User-Nachricht hinzufügen
        messages.append({"role": "user", "content": request.message})

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        ai_response = response.choices[0].message.content
        
        # Prüfen ob Liste geändert wurde
        updated_list = None
        action_performed = "none"
        
        # Einfache Erkennung von Änderungsabsichten
        lower_message = request.message.lower()
        if any(keyword in lower_message for keyword in ["hinzufügen", "add", "brauche noch", "vergessen"]):
            action_performed = "added"
        elif any(keyword in lower_message for keyword in ["entfernen", "remove", "löschen", "streichen"]):
            action_performed = "removed"
        elif any(keyword in lower_message for keyword in ["ändern", "modify", "anpassen", "korrigieren"]):
            action_performed = "modified"
        
        # Versuche JSON aus der Antwort zu extrahieren (falls Liste geändert wurde)
        if "[" in ai_response and "]" in ai_response:
            try:
                json_start = ai_response.find("[")
                json_end = ai_response.rfind("]") + 1
                json_str = ai_response[json_start:json_end]
                updated_list = json.loads(json_str)
            except:
                pass  # Falls kein valides JSON gefunden wird
        
        return ShoppingListChatResponse(
            response=ai_response,
            updated_list=updated_list,
            action_performed=action_performed
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat-Fehler: {str(e)}")

@router.post("/shopping-list-suggestions")
async def get_shopping_suggestions(request: Dict[str, Any]):
    """
    Gibt Vorschläge basierend auf der aktuellen Einkaufsliste.
    """
    try:
        shopping_list = request.get("shopping_list", [])
        
        list_text = "\n".join([f"- {item['name']}" for item in shopping_list])
        
        prompt = f"""
Basierend auf dieser Einkaufsliste:
{list_text}

Gib 3-5 hilfreiche Vorschläge:
1. Fehlende Grundzutaten
2. Passende Rezeptideen
3. Kostenspartipps
4. Optimale Einkaufsreihenfolge

Antworte in kurzen, praktischen Stichpunkten.
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Du bist ein Einkaufsberater."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=500
        )
        
        return {"suggestions": response.choices[0].message.content}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Suggestions-Fehler: {str(e)}")