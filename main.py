from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import settings
from models import ChatRequest, ChatResponse
from memory import PostgresChatMemory
from utils import GeminiClient
from tools import init_qdrant_client, create_retrieval_tool_from_collection
from agents import SimpleAgent
from utils.openai_client import OpenAIClient


app = FastAPI(
    title="TRANSTUR Chat Agent API",
    description="API de chat para TRANSTUR con integración de Qdrant y Gemini",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_agent: Optional[SimpleAgent] = None
_memory: Optional[PostgresChatMemory] = None


async def bootstrap() -> None:

    global _agent, _memory

    print("[BOOTSTRAP] Inicializando conexión a PostgreSQL...")
    if not settings.POSTGRES_CONNECTION_STRING:
        print("[BOOTSTRAP] ⚠️ POSTGRES_CONNECTION_STRING no configurada")
        _memory = None
    else:
        try:
            _memory = PostgresChatMemory(settings.POSTGRES_CONNECTION_STRING)
            await _memory.init()
            print("[BOOTSTRAP]  Memoria PostgreSQL inicializada correctamente")
        except Exception as e:
            print(f"[BOOTSTRAP]  Error inicializando PostgreSQL: {type(e).__name__}: {str(e)}")
            _memory = None

    llm = None
    if settings.GEMINI_API_KEY and settings.STATUS == "production":
        try:
            gemini_client = GeminiClient(settings.GEMINI_API_KEY)
            llm = gemini_client
            print("[BOOTSTRAP]  Cliente Gemini inicializado correctamente")
        except Exception as e:
            print(f"[BOOTSTRAP]  No se pudo inicializar Gemini client: {e}")
    
    else:
        try:
            openai_client = OpenAIClient(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_URL)
            llm = openai_client
            print("[BOOTSTRAP]  Cliente OpenAI inicializado correctamente")
        except Exception as e:
            print(f"[BOOTSTRAP]  No se pudo inicializar OpenAI client: {e}")

    q_client = init_qdrant_client(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)
    embeddings = None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=settings.GEMINI_API_KEY
        )
        print("[BOOTSTRAP]  Embeddings de Gemini inicializados correctamente")
    except Exception as e:
        print(f"[BOOTSTRAP]  No se pudieron inicializar embeddings de Gemini: {e}")
        embeddings = None

    tool1 = None
    tool2 = None
    tool1_desc = "Contiene: Términos y Condiciones de Renta, Requisitos del Conductor, Políticas de Cancelación, Garantía de Vehículo, Contrato y Vigencia, Penalidades por Drop-Off, Canales de Contacto Oficiales."
    tool2_desc = "Contiene: Tarifas de Alquiler, Precios por Categoría, Disponibilidad de Modelos, Ubicaciones de Oficinas, Datos Operacionales de Flota, Información Logística."
    
    if q_client and embeddings:
        tool1 = create_retrieval_tool_from_collection(
            settings.QDRANT_COLLECTION_1, 
            q_client, 
            embeddings
        )
        tool2 = create_retrieval_tool_from_collection(
            settings.QDRANT_COLLECTION_2, 
            q_client, 
            embeddings
        )

    if llm and _memory:
        _agent = SimpleAgent(
            llm=llm, 
            memory=_memory, 
            tool1=tool1, 
            tool2=tool2, 
            tool1_desc=tool1_desc, 
            tool2_desc=tool2_desc
        )
        print("[BOOTSTRAP]  Agente SimpleAgent inicializado correctamente")
    else:
        print("[BOOTSTRAP]  AVISO: Agente no inicializado completamente. Revisa GEMINI_API_KEY y POSTGRES_CONNECTION_STRING.")

@app.on_event("startup")
async def on_startup():

    await bootstrap()



@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):

    print(f"[CHAT] Mensaje recibido de {request.chat_id}: {request.message}")
    
    try:
        if _agent is None:
            raise HTTPException(
                status_code=503, 
                detail="El servicio no está disponible. El agente no está inicializado."
            )
        
        reply = await _agent.run(request.chat_id, request.message)
        
        print(f"[CHAT] Respuesta generada para {request.chat_id}: {reply[:100]}...")
        
        return ChatResponse(chat_id=request.chat_id, response=reply)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[CHAT] ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error procesando mensaje: {str(e)}"
        )


@app.get("/chat/{chat_id}/history")
async def get_chat_history(chat_id: str, limit: int = 10):

    print(f"[HISTORY] Solicitando historial de {chat_id} (limit={limit})")
    
    try:
        if _memory is None:
            raise HTTPException(
                status_code=503, 
                detail="El servicio de memoria no está disponible."
            )
        
        history = await _memory.get_recent(chat_id, limit=limit)
        
        print(f"[HISTORY] Recuperados {len(history)} mensajes para {chat_id}")
        
        return {"chat_id": chat_id, "history": history}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[HISTORY] ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"Error obteniendo historial: {str(e)}"
        )


@app.get("/health")
async def health_check():

    return {
        "status": "ok",
        "agent_ready": _agent is not None,
        "memory_ready": _memory is not None,
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app", 
        host=settings.SERVER_HOST, 
        port=settings.SERVER_PORT, 
        reload=True
    )
