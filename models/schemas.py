"""Modelos Pydantic para requests y responses."""
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Request para enviar un mensaje al chat."""
    chat_id: str
    message: str


class ChatResponse(BaseModel):
    """Response con la respuesta del agente."""
    chat_id: str
    response: str
