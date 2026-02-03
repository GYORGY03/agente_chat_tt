import asyncio
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except Exception:
    ChatGoogleGenerativeAI = None


class GeminiClient:

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError("Instala langchain-google-genai: pip install langchain-google-genai")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY no configurada en entorno")
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.0,
        )

    async def generate(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()

        def sync_call():
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)

        return await loop.run_in_executor(None, sync_call)
