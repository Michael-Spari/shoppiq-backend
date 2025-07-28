import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_API_URL = os.getenv("PINECONE_API_URL")
    
    # Optional: Validation
    def __post_init__(self):
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        if not self.PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
    
settings = Settings()