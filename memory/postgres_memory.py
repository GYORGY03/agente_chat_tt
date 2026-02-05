from typing import Dict, List, Optional

try:
    import asyncpg
except Exception:  
    asyncpg = None


class PostgresChatMemory:

    def __init__(self, dsn: str):
        if asyncpg is None:
            raise RuntimeError("`asyncpg` no está instalado. Instala asyncpg para usar PostgresChatMemory.")
        if not dsn:
            raise RuntimeError("POSTGRES_CONNECTION_STRING no configurada.")
        self._dsn = dsn
        self._pool: Optional[asyncpg.Pool] = None

    async def init(self) -> None:
        if self._pool:
            print("[POSTGRES] Pool de conexiones ya inicializado")
            return
        
        print(f"[POSTGRES] Conectando a PostgreSQL...")
        print(f"[POSTGRES] DSN: {self._dsn[:50]}...")
        
        try:
            self._pool = await asyncpg.create_pool(self._dsn)
            print("[POSTGRES] Pool de conexiones creado exitosamente")
            
            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chat_messages_web (
                        id SERIAL PRIMARY KEY,
                        chat_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
                )
                print("[POSTGRES]  Tabla 'chat_messages_web' verificada/creada")
                
                result = await conn.fetchval("SELECT COUNT(*) FROM chat_messages_web")
                print(f"[POSTGRES]  Mensajes en base de datos: {result}")
        except Exception as e:
            print(f"[POSTGRES] Error al conectar: {type(e).__name__}: {str(e)}")
            raise

    async def add_message(self, chat_id: str, role: str, content: str) -> None:
        if not self._pool:
            await self.init()
        
        print(f"[POSTGRES] Guardando mensaje - Chat: {chat_id}, Role: {role}")
        
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    "INSERT INTO chat_messages_web(chat_id, role, content) VALUES($1, $2, $3)",
                    chat_id,
                    role,
                    content,
                )
                print(f"[POSTGRES]  Mensaje guardado exitosamente")
        except Exception as e:
            print(f"[POSTGRES]  Error al guardar mensaje: {type(e).__name__}: {str(e)}")
            raise

    async def get_recent(self, chat_id: str, limit: int = 10) -> List[Dict[str, str]]:
        if not self._pool:
            await self.init()
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT role, content, created_at FROM chat_messages_web WHERE chat_id=$1 ORDER BY created_at DESC LIMIT $2",
                chat_id,
                limit,
            )
        return [dict(row) for row in reversed(rows)]
