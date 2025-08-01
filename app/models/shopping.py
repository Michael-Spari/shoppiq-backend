from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class ShoppingItem(BaseModel):
    uuid: str
    name: str
    quantity: int = 1
    note: Optional[str] = ""
    category: Optional[str] = None
    isChecked: bool = False
    supermarkt: Optional[str] = None

class Supermarket(BaseModel):
    uuid: str
    name: str
    address: str
    placeId: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class ShoppingList(BaseModel):
    uuid: str
    name: str
    items: List[ShoppingItem]
    created_at: datetime
    updated_at: datetime
    user_email: str

class Recipe(BaseModel):
    uuid: str
    name: str
    ingredients: List[str]
    instructions: List[str]
    category: Optional[str] = None
    cooking_time: Optional[int] = None
    difficulty: Optional[str] = None
    user_email: str

class CookingPlan(BaseModel):
    uuid: str
    name: str
    recipes: List[Recipe]
    date: datetime
    user_email: str