"""Modelos Pydantic para requests y responses."""
from pydantic import BaseModel


class ChatRequest(BaseModel):
    chat_id: str
    message: str


class ChatResponse(BaseModel):
    chat_id: str
    response: str
    chat_id: str
    response: str
