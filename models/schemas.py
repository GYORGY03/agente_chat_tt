from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    chat_id: str = Field(..., min_length=1, max_length=100)
    message: str = Field(..., min_length=1, max_length=5000)


class ChatResponse(BaseModel):
    chat_id: str
    response: str
    