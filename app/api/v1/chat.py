from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from openai import OpenAI
from app.config import settings
import json
import uuid

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

class ShoppingItemResponse(BaseModel):
    """Strukturiertes Shopping Item für Response"""
    uuid: str
    name: str
    quantity: int = 1
    note: str = ""
    category: Optional[str] = None
    isChecked: bool = False
    supermarkt: Optional[str] = None

class ShoppingListChatResponse(BaseModel):
    response: str
    updated_list: Optional[List[ShoppingItemResponse]] = None  # Strukturierte Items
    action_performed: Optional[str] = None  # "added", "removed", "modified", "none"

@router.post("/shopping-list-chat", response_model=ShoppingListChatResponse)
async def chat_about_shopping_list(request: ShoppingListChatRequest):
    """
    Chat-Service für Fragen zu bestehenden Einkaufslisten.
    Kann Items hinzufügen, entfernen, modifizieren oder Fragen beantworten.
    """
    try:
        # Aktuelle Einkaufsliste als String formatieren
        if request.shopping_list:
            list_text = "\n".join([
                f"- {item['name']} (Menge: {item.get('quantity', 1)}, Supermarkt: {item.get('supermarkt', 'unbekannt')}, Status: {'✓ gekauft' if item.get('isChecked', False) else '○ offen'})"
                for item in request.shopping_list
            ])
        else:
            list_text = "Die Einkaufsliste ist aktuell leer."
        
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
- Erstelle eine aktualisierte Liste im JSON-Format
- Verwende dieses Format: [{{"name": "Produktname", "quantity": 1, "note": "", "supermarkt": "Marktname", "uuid": "unique-id"}}]
- Erkläre was du geändert hast

Wenn der User nur eine Frage stellt:
- Beantworte sie basierend auf der aktuellen Liste
- Gib hilfreiche Tipps

Sei freundlich und hilfsbereit!
"""

        # Chat-Verlauf aufbauen
        messages = [{"role": "system", "content": system_prompt}]
        
        # Bisherige Chat-Historie hinzufügen (ChatMessage zu dict konvertieren)
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
        if any(keyword in lower_message for keyword in ["hinzufügen", "add", "brauche noch", "vergessen", "füge hinzu"]):
            action_performed = "added"
        elif any(keyword in lower_message for keyword in ["entfernen", "remove", "löschen", "streichen", "weg"]):
            action_performed = "removed"
        elif any(keyword in lower_message for keyword in ["ändern", "modify", "anpassen", "korrigieren", "update"]):
            action_performed = "modified"
        
        # Versuche JSON aus der Antwort zu extrahieren (falls Liste geändert wurde)
        if "[" in ai_response and "]" in ai_response:
            try:
                json_start = ai_response.find("[")
                json_end = ai_response.rfind("]") + 1
                json_str = ai_response[json_start:json_end]
                raw_list = json.loads(json_str)
                
                # Konvertiere zu strukturierten ShoppingItemResponse Objekten
                updated_list = []
                for item in raw_list:
                    if isinstance(item, dict):
                        updated_list.append(ShoppingItemResponse(
                            uuid=item.get('uuid', str(uuid.uuid4())),
                            name=item.get('name', 'Unbekannt'),
                            quantity=item.get('quantity', 1),
                            note=item.get('note', ''),
                            category=item.get('category'),
                            isChecked=item.get('isChecked', False),
                            supermarkt=item.get('supermarkt')
                        ))
                        
            except json.JSONDecodeError as e:
                print(f"JSON parsing error: {e}")
                pass  # Falls kein valides JSON gefunden wird
            except Exception as e:
                print(f"List parsing error: {e}")
                pass
        
        # Falls Änderungsabsicht erkannt, aber kein JSON gefunden
        if action_performed != "none" and updated_list is None and request.shopping_list:
            # Kopiere bestehende Liste als Fallback
            updated_list = []
            for item in request.shopping_list:
                updated_list.append(ShoppingItemResponse(
                    uuid=item.get('uuid', str(uuid.uuid4())),
                    name=item.get('name', 'Unbekannt'),
                    quantity=item.get('quantity', 1),
                    note=item.get('note', ''),
                    category=item.get('category'),
                    isChecked=item.get('isChecked', False),
                    supermarkt=item.get('supermarkt')
                ))
        
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
        
        if shopping_list:
            list_text = "\n".join([f"- {item['name']}" for item in shopping_list])
        else:
            list_text = "Die Liste ist leer."
        
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