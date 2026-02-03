import asyncio
try:
    from openai import OpenAI
except Exception:
        print("Instala openai: pip install openai")
        OpenAI = None

class OpenAIClient:
    def __init__(self, api_key: str, base_url: str):
        if OpenAI is None:
            raise RuntimeError("Instala openai: pip install openai")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY no configurada en entorno")
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    async def generate(self, prompt: str) -> str:
        loop = asyncio.get_event_loop()

        def sync_call():
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return response.choices[0].message.content

        return await loop.run_in_executor(None, sync_call)