from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import uuid
from datetime import datetime

# OpenAI & Pinecone
from openai import OpenAI
import pinecone

# Config
from app.config import settings

router = APIRouter()

# Initialize services
client = OpenAI(api_key=settings.OPENAI_API_KEY)

# ✅ PINECONE INITIALISIERUNG
pinecone.init(
    api_key=settings.PINECONE_API_KEY,
    environment=settings.PINECONE_ENVIRONMENT
)
index = pinecone.Index(settings.PINECONE_INDEX_NAME)

# ✅ FIREBASE KORREKT INITIALISIEREN
FIREBASE_ENABLED = False
db = None

def initialize_firebase():
    global FIREBASE_ENABLED, db
    
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
        
        # Prüfen ob bereits initialisiert
        if firebase_admin._apps:
            db = firestore.client()
            FIREBASE_ENABLED = True
            print("✅ Firebase already initialized")
            return
        
        # Firebase Credentials prüfen
        if not all([
            settings.FIREBASE_PROJECT_ID,
            settings.FIREBASE_PRIVATE_KEY,
            settings.FIREBASE_CLIENT_EMAIL
        ]):
            print("⚠️ Firebase credentials incomplete - using mock mode")
            return
        
        # Firebase Config
        firebase_config = {
            "type": "service_account",
            "project_id": settings.FIREBASE_PROJECT_ID,
            "private_key_id": settings.FIREBASE_PRIVATE_KEY_ID,
            "private_key": settings.FIREBASE_PRIVATE_KEY,
            "client_email": settings.FIREBASE_CLIENT_EMAIL,
            "client_id": settings.FIREBASE_CLIENT_ID,
            "auth_uri": settings.FIREBASE_AUTH_URI,
            "token_uri": settings.FIREBASE_TOKEN_URI,
            "client_x509_cert_url": settings.FIREBASE_CLIENT_CERT_URL
        }
        
        # Firebase initialisieren
        cred = credentials.Certificate(firebase_config)
        firebase_admin.initialize_app(cred)
        
        db = firestore.client()
        FIREBASE_ENABLED = True
        print("✅ Firebase initialized successfully")
        
    except ImportError:
        print("⚠️ Firebase Admin SDK not available")
    except Exception as e:
        print(f"❌ Firebase initialization error: {e}")

# Firebase beim Start initialisieren
initialize_firebase()

# ... ALLE Models bleiben EXAKT GLEICH ...
class GenerateShoppingListRequest(BaseModel):
    settings: Dict[str, Any]
    user_email: str
    list_name: Optional[str] = "KI-Einkaufsliste"
    context: Optional[str] = None

class ShoppingItemResponse(BaseModel):
    uuid: str
    name: str
    quantity: int
    unit: Optional[str] = "Stück"
    category: Optional[str] = None
    estimated_price: Optional[float] = None
    supermarket: Optional[str] = None
    note: Optional[str] = None

class ShoppingListResponse(BaseModel):
    uuid: str
    name: str
    created_at: str
    items: List[ShoppingItemResponse]
    total_estimated_price: Optional[float] = None
    supermarkets: List[str] = []
    created_by: str

class GenerateShoppingListResponse(BaseModel):
    shopping_list: ShoppingListResponse
    success: bool
    message: Optional[str] = None

# ... ALLE Helper Functions bleiben EXAKT GLEICH bis auf save_shopping_list_to_firebase ...

async def save_shopping_list_to_firebase(shopping_list_data: Dict, user_email: str) -> str:
    """Speichert ShoppingList in Firebase Firestore (mit Fallback)"""
    
    if not FIREBASE_ENABLED or not db:
        mock_id = f"mock_firebase_{uuid.uuid4()}"
        print(f"⚠️ Firebase not available - using mock ID: {mock_id}")
        return mock_id
    
    try:
        # Firestore Collection Reference
        shopping_lists_ref = db.collection('shopping_lists')
        
        # Document erstellen
        doc_ref = shopping_lists_ref.add(shopping_list_data)
        doc_id = doc_ref[1].id
        
        print(f"✅ Shopping list saved to Firebase with ID: {doc_id}")
        return doc_id
        
    except Exception as e:
        print(f"❌ Firebase save error: {e}")
        # Fallback zu Mock ID statt Exception
        mock_id = f"mock_firebase_{uuid.uuid4()}"
        print(f"📝 Using fallback mock ID: {mock_id}")
        return mock_id

async def generate_ai_shopping_list(settings: Dict[str, Any], user_email: str, context: Optional[str] = None, user_products: List[Dict] = []) -> Dict[str, Any]:
    """Generiert Shopping List mit OpenAI basierend auf Settings und User-History"""
    
    # User Context aus Pinecone
    user_context = ""
    if user_products:
        product_names = [p.get('name', '') for p in user_products[:20]]
        user_context = f"\n\nDeine bisherigen Produkte: {', '.join(product_names)}"
    
    # System Prompt
    system_prompt = f"""Du bist ein intelligenter Einkaufslistenassistent. 
Erstelle eine detaillierte Einkaufsliste basierend auf den Benutzereinstellungen.

WICHTIG: Antworte NUR mit einem gültigen JSON-Array im folgenden Format:
[
  {{
    "name": "Produktname",
    "quantity": 1,
    "unit": "Stück",
    "category": "Kategorie",
    "estimated_price": 2.50,
    "supermarket": "REWE",
    "note": "Optional: Hinweise"
  }}
]

Kategorien: Obst & Gemüse, Fleisch & Fisch, Milchprodukte, Getränke, Brot & Backwaren, Tiefkühlkost, Konserven, Süßwaren, Haushaltsartikel, Sonstiges

Supermärkte: REWE, EDEKA, ALDI, LIDL, Kaufland, Netto

Berücksichtige realistische deutsche Preise.{user_context}"""

    # User Message
    user_message = f"""Erstelle eine Einkaufsliste für folgende Einstellungen:
{json.dumps(settings, indent=2, ensure_ascii=False)}

Zusätzlicher Kontext: {context or 'Keine spezifischen Anforderungen'}

Erstelle eine sinnvolle Einkaufsliste mit 15-25 Produkten."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        ai_response = response.choices[0].message.content
        print(f"🤖 AI Response length: {len(ai_response)} chars")
        
        # JSON extrahieren
        json_start = ai_response.find('[')
        json_end = ai_response.rfind(']') + 1
        
        if json_start == -1 or json_end == 0:
            raise Exception("No valid JSON found in AI response")
            
        json_str = ai_response[json_start:json_end]
        items_data = json.loads(json_str)
        
        print(f"✅ Parsed {len(items_data)} items from AI")
        return {"items": items_data, "raw_response": ai_response}
        
    except Exception as e:
        print(f"❌ AI Generation Error: {e}")
        raise HTTPException(status_code=500, detail=f"AI generation failed: {e}")

async def save_items_to_pinecone(items: List[Dict], user_email: str, list_uuid: str):
    """Speichert Items in Pinecone Vector DB für zukünftige Empfehlungen"""
    
    vectors_to_upsert = []
    
    for item in items:
        try:
            # Text für Embedding
            text_for_embedding = f"{item['name']} {item.get('category', '')} {item.get('note', '')}"
            
            # OpenAI Embedding erstellen
            embedding_response = client.embeddings.create(
                model="text-embedding-3-small",
                input=text_for_embedding
            )
            
            vector_id = f"item_{uuid.uuid4()}"
            
            # Metadata
            metadata = {
                "user_email": user_email,
                "list_uuid": list_uuid,
                "item_type": "shopping_item",
                "name": item["name"],
                "category": item.get("category"),
                "quantity": item.get("quantity", 1),
                "estimated_price": item.get("estimated_price"),
                "supermarket": item.get("supermarket"),
                "created_at": datetime.now().isoformat(),
            }
            
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embedding_response.data[0].embedding,
                "metadata": metadata
            })
            
        except Exception as e:
            print(f"⚠️ Error creating embedding for {item.get('name', 'unknown')}: {e}")
            continue
    
    # Batch upsert zu Pinecone
    if vectors_to_upsert:
        try:
            index.upsert(vectors=vectors_to_upsert)
            print(f"✅ Upserted {len(vectors_to_upsert)} vectors to Pinecone")
        except Exception as e:
            print(f"❌ Pinecone upsert error: {e}")

async def save_shopping_list_to_firebase(shopping_list_data: Dict, user_email: str) -> str:
    """Speichert ShoppingList in Firebase Firestore"""
    
    try:
        # Firestore Collection Reference
        shopping_lists_ref = db.collection('shopping_lists')
        
        # Document erstellen
        doc_ref = shopping_lists_ref.add(shopping_list_data)
        doc_id = doc_ref[1].id
        
        print(f"✅ Shopping list saved to Firebase with ID: {doc_id}")
        return doc_id
        
    except Exception as e:
        print(f"❌ Firebase save error: {e}")
        raise HTTPException(status_code=500, detail=f"Firebase save failed: {e}")

@router.post("/generate-shopping-list", response_model=GenerateShoppingListResponse)
async def generate_shopping_list(request: GenerateShoppingListRequest):
    """
    Generiert eine komplette Einkaufsliste mit AI und speichert sie in Firebase + Pinecone
    """
    try:
        print(f"🛒 Generating shopping list for user: {request.user_email}")
        print(f"📋 Settings: {request.settings}")
        
        # 1. User's bisherige Produkte aus Pinecone laden
        user_products = await get_user_product_context(request.user_email)
        
        # 2. AI Shopping List generieren
        ai_result = await generate_ai_shopping_list(
            request.settings, 
            request.user_email,
            request.context,
            user_products
        )
        
        # 3. ShoppingList Model erstellen
        list_uuid = str(uuid.uuid4())
        created_at = datetime.now()
        
        # Items zu Response Format konvertieren
        shopping_items = []
        supermarkets = set()
        total_price = 0.0
        
        for item_data in ai_result["items"]:
            item_uuid = str(uuid.uuid4())
            
            item = ShoppingItemResponse(
                uuid=item_uuid,
                name=item_data["name"],
                quantity=item_data.get("quantity", 1),
                unit=item_data.get("unit", "Stück"),
                category=item_data.get("category"),
                estimated_price=item_data.get("estimated_price"),
                supermarket=item_data.get("supermarket"),
                note=item_data.get("note")
            )
            
            shopping_items.append(item)
            
            # Supermarkt sammeln
            if item.supermarket:
                supermarkets.add(item.supermarket)
            
            # Preis summieren
            if item.estimated_price:
                total_price += item.estimated_price * item.quantity
        
        # ShoppingList Response
        shopping_list = ShoppingListResponse(
            uuid=list_uuid,
            name=request.list_name,
            created_at=created_at.isoformat(),
            items=shopping_items,
            total_estimated_price=round(total_price, 2) if total_price > 0 else None,
            supermarkets=list(supermarkets),
            created_by=request.user_email
        )
        
        # 4. Firebase speichern
        firebase_data = {
            "uuid": list_uuid,
            "name": request.list_name,
            "created_at": created_at,
            "items": [item.dict() for item in shopping_items],
            "total_estimated_price": shopping_list.total_estimated_price,
            "supermarkets": shopping_list.supermarkets,
            "created_by": request.user_email,
            "settings": request.settings,
            "context": request.context
        }
        
        firebase_doc_id = await save_shopping_list_to_firebase(firebase_data, request.user_email)
        
        # 5. Items zu Pinecone speichern (für zukünftige Empfehlungen)
        await save_items_to_pinecone(ai_result["items"], request.user_email, list_uuid)
        
        return GenerateShoppingListResponse(
            shopping_list=shopping_list,
            success=True,
            message=f"Einkaufsliste erfolgreich generiert mit {len(shopping_items)} Produkten"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Generate shopping list error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

# Health Check
@router.get("/generate-shopping-list/health")
async def health_check():
    return {"status": "healthy", "service": "generate_shopping_list"}