"""Configuración central del proyecto."""
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Configuración de la aplicación."""
    
    GEMINI_API_KEY = "AIzaSyBMt3o2djQMpbKd_p6Uk1n3znFxefp1rGw"
    
    POSTGRES_CONNECTION_STRING = "postgresql://jorge:laberinto@localhost:5432/labtech_bot"
    
    QDRANT_URL = "http://localhost:6333"
    QDRANT_API_KEY = "e5362baf-c777-4d57-a609-6eaf1f9e87f6"
    QDRANT_COLLECTION_1 = "documentos_pdf"  
    QDRANT_COLLECTION_2 = "documentos_tarifas"  
    
    SERVER_HOST = "0.0.0.0"
    SERVER_PORT = 5678
    
    CORS_ORIGINS = [
        "https://agent-tt.netlify.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "*"
    ]


settings = Settings()
