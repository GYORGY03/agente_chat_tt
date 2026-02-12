from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    STATUS=os.getenv("STATUS", "development")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    OPENAI_URL=os.getenv("OPENAI_URL")
    OPENAI_API_KEY=os.getenv("OPENAI_API_KEY", "not-needed")
    POSTGRES_CONNECTION_STRING = os.getenv("POSTGRES_CONNECTION_STRING")
    
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_1 = os.getenv("QDRANT_COLLECTION_1")  
    QDRANT_COLLECTION_2 = os.getenv("QDRANT_COLLECTION_2")  
    
    SERVER_HOST = os.getenv("SERVER_HOST")
    SERVER_PORT = int(os.getenv("SERVER_PORT"))
    
    CORS_ORIGINS = os.getenv("CORS_ORIGINS").split(",") 


settings = Settings()
