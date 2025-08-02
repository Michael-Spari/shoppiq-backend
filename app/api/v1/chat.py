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
    similar_lists: List[Dict[str, Any]] = []  # NEUE: Ähnliche Listen aus Pinecone

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
    Nutzt ähnliche Listen aus Pinecone als Kontext.
    """
    try:
        # DEBUG: Eingehende Daten loggen
        print(f"🔍 Received shopping list with {len(request.shopping_list)} items")
        if request.shopping_list:
            for i, item in enumerate(request.shopping_list[:3]):  # Zeige nur erste 3
                print(f"  {i+1}. {item.get('name', 'Unnamed')} - {item.get('brand', 'No brand')} (qty: {item.get('quantity', 0)})")
            if len(request.shopping_list) > 3:
                print(f"  ... und {len(request.shopping_list) - 3} weitere Items")
        
        print(f"🔍 Received {len(request.similar_lists)} similar lists")
        for similar in request.similar_lists[:2]:
            print(f"  - {similar.get('name', 'Unnamed list')}")
        
        # Aktuelle Einkaufsliste als String formatieren
        if request.shopping_list:
            list_text = "\n".join([
                f"- {item['name']} (Menge: {item.get('quantity', 1)}, Supermarkt: {item.get('supermarkt', 'unbekannt')}, Status: {'✓ gekauft' if item.get('isChecked', False) else '○ offen'})"
                for item in request.shopping_list
            ])
            print(f"📝 Formatted list text: {list_text[:200]}...")  # Erste 200 Zeichen
        else:
            list_text = "Die Einkaufsliste ist aktuell leer."
            print("❌ No items in shopping list")
        
        # ERWEITERT: Ähnliche Listen als Kontext hinzufügen
        similar_context = ""
        if request.similar_lists:
            similar_context = "\n\nÄHNLICHE FRÜHERE EINKAUFSLISTEN (als Inspiration):\n"
            for i, similar_list in enumerate(request.similar_lists[:3], 1):  # Max 3 Listen
                similar_name = similar_list.get('name', f'Liste {i}')
                similar_context += f"\n{i}. {similar_name}:\n"
                
                # Parse items wenn vorhanden
                if 'items' in similar_list:
                    try:
                        # Items können als JSON-String oder bereits als Liste vorliegen
                        items = similar_list['items']
                        if isinstance(items, str):
                            items = json.loads(items)
                        elif not isinstance(items, list):
                            items = []
                        
                        # Zeige erste 5 Items der ähnlichen Liste
                        for item in items[:5]:
                            if isinstance(item, dict):
                                item_name = item.get('name', 'Unbekannt')
                                item_qty = item.get('quantity', 1)
                                similar_context += f"   - {item_name} ({item_qty}x)\n"
                            elif isinstance(item, str):
                                similar_context += f"   - {item}\n"
                    except Exception as e:
                        print(f"Error parsing similar list items: {e}")
                        pass
                
                # Weitere Metadaten hinzufügen falls verfügbar
                if 'supermarkets' in similar_list:
                    markets = similar_list['supermarkets']
                    if isinstance(markets, str) and markets:
                        similar_context += f"   Märkte: {markets}\n"
                
                if 'note' in similar_list and similar_list['note']:
                    similar_context += f"   Notiz: {similar_list['note']}\n"
        
        # Erweiterten System Prompt mit ähnlichen Listen
        system_prompt = f"""
Du bist ein intelligenter Assistent für Einkaufslisten-Management.

AKTUELLE EINKAUFSLISTE:
{list_text}

{similar_context}

Du kannst:
1. Fragen zur Liste beantworten
2. Items hinzufügen/entfernen/ändern basierend auf aktueller und früheren Listen
3. Vorschläge aus ähnlichen Listen machen
4. Einkaufstipps geben
5. Kosten schätzen
6. Rezeptvorschläge basierend auf den Produkten machen

WICHTIGE REGELN:

Wenn der User die Liste ändern möchte:
- Erstelle eine VOLLSTÄNDIGE aktualisierte Liste im JSON-Format
- Verwende dieses exakte Format: [{{"name": "Produktname", "quantity": 1, "note": "", "supermarkt": "Marktname", "uuid": "unique-id"}}]
- Inkludiere ALLE Items (bestehende + neue + geänderte)
- Entferne nur Items die explizit gelöscht werden sollen
- Erkläre was du geändert hast

Wenn der User nur eine Frage stellt:
- Beantworte sie basierend auf der aktuellen Liste
- Nutze ähnliche Listen für bessere Vorschläge
- Gib hilfreiche Tipps

Wenn du Vorschläge machst:
- Berücksichtige Produkte aus ähnlichen Listen
- Schätze realistische Mengen
- Schlage passende Supermärkte vor

Sei freundlich, hilfsbereit und nutze die Historie intelligent!
"""

        # Chat-Verlauf aufbauen
        messages = [{"role": "system", "content": system_prompt}]
        
        # Bisherige Chat-Historie hinzufügen (ChatMessage zu dict konvertieren)
        for msg in request.chat_history[-5:]:  # Nur letzte 5 Nachrichten
            messages.append({"role": msg.role, "content": msg.content})
        
        # Aktuelle User-Nachricht hinzufügen
        messages.append({"role": "user", "content": request.message})

        print(f"🤖 Sending {len(messages)} messages to OpenAI (system + history + current)")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
            max_tokens=1500  # Erhöht für längere Listen
        )
        
        ai_response = response.choices[0].message.content
        print(f"🤖 AI Response length: {len(ai_response)} chars")
        print(f"🤖 AI Response preview: {ai_response[:150]}...")
        
        # Prüfen ob Liste geändert wurde
        updated_list = None
        action_performed = "none"
        
        # Erweiterte Erkennung von Änderungsabsichten
        lower_message = request.message.lower()
        lower_response = ai_response.lower()
        
        # Keywords definieren
        add_keywords = ["hinzufügen", "add", "brauche noch", "vergessen", "füge hinzu", "brauch noch", "setze dazu", "ergänze"]
        remove_keywords = ["entfernen", "remove", "löschen", "streichen", "weg", "raus", "delete", "entferne"]
        modify_keywords = ["ändern", "modify", "anpassen", "korrigieren", "update", "ändere", "bearbeite"]
        
        # KORRIGIERT: Richtige if-elif-else Struktur
        if any(keyword in lower_message for keyword in add_keywords) or any(keyword in lower_response for keyword in add_keywords):
            action_performed = "added"
        elif any(keyword in lower_message for keyword in remove_keywords) or any(keyword in lower_response for keyword in remove_keywords):
            action_performed = "removed"
        elif any(keyword in lower_message for keyword in modify_keywords) or any(keyword in lower_response for keyword in modify_keywords):
            action_performed = "modified"
        
        print(f"🔍 Detected action: {action_performed}")
        
        # Versuche JSON aus der Antwort zu extrahieren (falls Liste geändert wurde)
        if "[" in ai_response and "]" in ai_response:
            try:
                # Finde JSON-Array in der Antwort
                json_start = ai_response.find("[")
                json_end = ai_response.rfind("]") + 1
                json_str = ai_response[json_start:json_end]
                
                print(f"🔧 Attempting to parse JSON: {json_str[:100]}...")
                
                # Parse JSON
                raw_list = json.loads(json_str)
                
                # Konvertiere zu strukturierten ShoppingItemResponse Objekten
                updated_list = []
                for item in raw_list:
                    if isinstance(item, dict) and 'name' in item:
                        # Generiere UUID falls nicht vorhanden
                        item_uuid = item.get('uuid')
                        if not item_uuid or item_uuid == "unique-id":
                            item_uuid = str(uuid.uuid4())
                        
                        updated_list.append(ShoppingItemResponse(
                            uuid=item_uuid,
                            name=item.get('name', 'Unbekannt'),
                            quantity=max(1, item.get('quantity', 1)),  # Mindestens 1
                            note=item.get('note', ''),
                            category=item.get('category'),
                            isChecked=item.get('isChecked', False),
                            supermarkt=item.get('supermarkt')
                        ))
                
                print(f"✅ Parsed {len(updated_list)} items from AI response")
                        
            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing error: {e}")
                print(f"Problematic JSON: {json_str if 'json_str' in locals() else 'Not found'}")
                pass  # Falls kein valides JSON gefunden wird
            except Exception as e:
                print(f"❌ List parsing error: {e}")
                pass
        
        # Fallback: Falls Änderungsabsicht erkannt, aber kein JSON gefunden
        if action_performed != "none" and updated_list is None:
            if request.shopping_list:
                # Kopiere bestehende Liste als Fallback
                updated_list = []
                for item in request.shopping_list:
                    updated_list.append(ShoppingItemResponse(
                        uuid=item.get('uuid', str(uuid.uuid4())),
                        name=item.get('name', 'Unbekannt'),
                        quantity=max(1, item.get('quantity', 1)),
                        note=item.get('note', ''),
                        category=item.get('category'),
                        isChecked=item.get('isChecked', False),
                        supermarkt=item.get('supermarkt')
                    ))
                print(f"⚠️ Used fallback: copied {len(updated_list)} existing items")
            else:
                # Neue leere Liste falls keine existiert
                updated_list = []
                print("⚠️ Used fallback: created empty list")
        
        # Final Debug-Output
        print(f"🔍 Final action performed: {action_performed}")
        print(f"📋 Final updated list items: {len(updated_list) if updated_list else 0}")
        if updated_list:
            for i, item in enumerate(updated_list[:3]):
                print(f"  {i+1}. {item.name} (qty: {item.quantity})")
        
        return ShoppingListChatResponse(
            response=ai_response,
            updated_list=updated_list,
            action_performed=action_performed
        )
        
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Chat-Fehler: {str(e)}")

@router.post("/shopping-list-suggestions")
async def get_shopping_suggestions(request: Dict[str, Any]):
    """
    Gibt Vorschläge basierend auf der aktuellen Einkaufsliste.
    Erweitert um Pinecone-basierte ähnliche Listen.
    """
    try:
        shopping_list = request.get("shopping_list", [])
        similar_lists = request.get("similar_lists", [])
        
        print(f"📋 Suggestions request: {len(shopping_list)} items, {len(similar_lists)} similar lists")
        
        # Aktuelle Liste formatieren
        if shopping_list:
            list_text = "\n".join([f"- {item['name']}" for item in shopping_list])
        else:
            list_text = "Die Liste ist leer."
        
        # Ähnliche Listen hinzufügen
        similar_text = ""
        if similar_lists:
            similar_text = "\n\nÄhnliche frühere Listen:\n"
            for similar_list in similar_lists[:2]:  # Top 2
                similar_name = similar_list.get('name', 'Unbekannte Liste')
                similar_text += f"- {similar_name}\n"
        
        prompt = f"""
Basierend auf dieser Einkaufsliste:
{list_text}

{similar_text}

Gib 5-7 hilfreiche Vorschläge:
1. Fehlende Grundzutaten basierend auf vorhandenen Produkten
2. Passende Rezeptideen mit den verfügbaren Zutaten
3. Kostenspartipps und Angebote
4. Optimale Einkaufsreihenfolge nach Supermarkt-Bereichen
5. Ergänzende Produkte aus ähnlichen Listen
6. Saisonale Empfehlungen
7. Gesunde Alternativen

Antworte in kurzen, praktischen Stichpunkten mit Emojis.
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Du bist ein intelligenter Einkaufsberater mit Zugang zu Einkaufshistorie."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=800
        )
        
        suggestions = response.choices[0].message.content
        print(f"✅ Generated suggestions: {len(suggestions)} chars")
        
        return {"suggestions": suggestions}
        
    except Exception as e:
        print(f"❌ Suggestions error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Suggestions-Fehler: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check für den Chat-Service"""
    return {
        "status": "healthy",
        "service": "shopping-list-chat",
        "openai_configured": bool(settings.OPENAI_API_KEY)
    }