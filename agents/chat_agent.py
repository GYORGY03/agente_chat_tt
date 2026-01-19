"""
Agente de Chat para TRANSTUR
"""
import asyncio
from typing import Optional, List, Dict, Any, Callable


class SimpleAgent:
    """
    Agente de chat con capacidades de clasificación de preguntas,
    expansión de queries y búsqueda en bases de conocimiento
    """
    
    def __init__(
        self,
        llm,
        memory,
        tool1: Optional[Callable] = None,
        tool2: Optional[Callable] = None,
        tool1_desc: str = "",
        tool2_desc: str = ""
    ):
        """
        Args:
            llm: Cliente LLM (GeminiClient)
            memory: Sistema de memoria (PostgresChatMemory)
            tool1: Herramienta de búsqueda en KB-1 (Políticas y Legal)
            tool2: Herramienta de búsqueda en KB-2 (Operaciones y Tarifas)
            tool1_desc: Descripción de KB-1
            tool2_desc: Descripción de KB-2
        """
        self.llm = llm
        self.memory = memory
        self.tool1 = tool1
        self.tool2 = tool2
        self.tool1_desc = tool1_desc
        self.tool2_desc = tool2_desc
    
    async def expand_query(self, query: str) -> List[str]:
        """
        Expande la query con sinónimos para mejorar resultados de búsqueda
        
        Args:
            query: Query original del usuario
            
        Returns:
            Lista de queries [original, expandida]
        """
        query_lower = query.lower()
        
        synonyms = {
            'tarifa': ['precio', 'costo', 'valor'],
            'auto': ['vehículo', 'carro', 'automóvil'],
            'alquiler': ['renta', 'arrendamiento'],
            'cancelar': ['anular', 'cancelación'],
            'requisito': ['condición', 'requerimiento'],
            'conductor': ['chofer', 'operador'],
            'seguro': ['cobertura', 'protección'],
            'oficina': ['sucursal', 'agencia', 'punto de servicio'],
            'disponible': ['disponibilidad', 'stock'],
            'categoría': ['tipo', 'clase', 'grupo'],
            'documento': ['documentación', 'papeles'],
            'contrato': ['acuerdo', 'convenio']
        }
        
        expanded_terms = []
        for key, values in synonyms.items():
            if key in query_lower:
                for syn in values:
                    if syn not in query_lower:
                        expanded_terms.append(syn)
        
        if expanded_terms:
            expanded_query = query + " " + " ".join(expanded_terms[:3])
            return [query, expanded_query]
        else:
            return [query]
    
    async def classify_question(self, query: str) -> Dict[str, Any]:
        """
        Clasifica la pregunta para priorizar búsqueda en KB correcta
        
        Args:
            query: Query del usuario
            
        Returns:
            Dict con:
                - prioritize: 'kb1', 'kb2' o None
                - kb1_filter: Filtros de metadata para KB-1
                - kb2_filter: Filtros de metadata para KB-2
        """
        query_lower = query.lower()
        
        legal_keywords = [
            'términos', 'condiciones', 'política', 'politica', 'requisito', 
            'conductor', 'licencia', 'edad', 'cancelación', 'cancelacion',
            'modificación', 'modificacion', 'garantía', 'garantia',
            'modelo', 'marca', 'drop-off', 'entrega', 'devolver',
            'contrato', 'vigencia', 'duración', 'duracion', 'penalidad',
            'contacto', 'correo', 'email', 'teléfono', 'telefono',
            'privacidad', 'datos personales', 'responsabilidad',
            'daño', 'dano', 'accidente', 'multa', 'infracción',
            'infraccion', 'seguridad', 'protección', 'proteccion', 'medidas'
        ]
        
        tarifa_keywords = [
            'precio', 'tarifa', 'costo', 'valor', 'cuánto', 'cuanto',
            'disponibilidad', 'disponible', 'oficina', 'sucursal',
            'ubicación', 'ubicacion', 'dónde', 'donde', 'localización',
            'localizacion', 'categoría', 'categoria', 'flota', 'modelos',
            'temporada', 'descuento', 'promoción', 'promocion'
        ]
        
        result = {
            'prioritize': None,
            'kb1_filter': None,
            'kb2_filter': None
        }
        
        kb1_matches = sum(1 for kw in legal_keywords if kw in query_lower)
        kb2_matches = sum(1 for kw in tarifa_keywords if kw in query_lower)
        
        if kb1_matches > kb2_matches:
            result['prioritize'] = 'kb1'
            print(f"[CLASIFICACIÓN] Pregunta clasificada como: POLÍTICAS/LEGAL")
        elif kb2_matches > kb1_matches:
            result['prioritize'] = 'kb2'
            print(f"[CLASIFICACIÓN] Pregunta clasificada como: TARIFAS/OPERACIONES")
        else:
            print(f"[CLASIFICACIÓN] Pregunta clasificada como: GENERAL (buscar en ambas KB)")
        
        return result

    async def run(self, chat_id: str, user_message: str) -> str:
        """
        Ejecuta el agente para procesar un mensaje del usuario
        
        Args:
            chat_id: ID del chat
            user_message: Mensaje del usuario
            
        Returns:
            Respuesta del agente
        """
        await self.memory.add_message(chat_id, "user", user_message)

        recent = await self.memory.get_recent(chat_id, limit=8)

        classification = await self.classify_question(user_message)

        docs1 = []
        docs2 = []
        
        expanded_queries = await self.expand_query(user_message)
        
        k1 = 8 if classification['prioritize'] == 'kb1' else 5
        k2 = 5 if classification['prioritize'] == 'kb1' else 8
        
        score_threshold_kb1 = 0.0  
        score_threshold_kb2 = 0.0  
        print(f"\n[QDRANT]  Iniciando búsquedas paralelas - KB-1 (k={k1}) y KB-2 (k={k2})")
        
        main_query = expanded_queries[0]
        
        async def search_kb1():
            """Búsqueda en KB-1 con manejo de errores"""
            if not self.tool1:
                print(f"[QDRANT]  KB-1: Herramienta no disponible")
                return []
            try:
                results = await self.tool1(
                    main_query, 
                    k=k1, 
                    metadata_filter=classification['kb1_filter'], 
                    score_threshold=score_threshold_kb1
                )
                
                if len(results) < 2 and len(expanded_queries) > 1:
                    print(f"[QDRANT] KB-1: Pocos resultados, intentando con query expandida...")
                    results_expanded = await self.tool1(
                        expanded_queries[1], 
                        k=k1, 
                        metadata_filter=classification['kb1_filter'], 
                        score_threshold=score_threshold_kb1
                    )
                    existing = {getattr(d, 'page_content', '') for d in results}
                    for doc in results_expanded:
                        if getattr(doc, 'page_content', '') not in existing:
                            results.append(doc)
                
                return results
            except Exception as e:
                print(f"[QDRANT]  Error en KB-1: {type(e).__name__}: {str(e)}")
                return []
        
        async def search_kb2():
            """Búsqueda en KB-2 con manejo de errores"""
            if not self.tool2:
                print(f"[QDRANT]  KB-2: Herramienta no disponible")
                return []
            try:
                results = await self.tool2(
                    main_query, 
                    k=k2, 
                    metadata_filter=classification['kb2_filter'], 
                    score_threshold=score_threshold_kb2
                )
                
                if len(results) < 2 and len(expanded_queries) > 1:
                    print(f"[QDRANT] KB-2: Pocos resultados, intentando con query expandida...")
                    results_expanded = await self.tool2(
                        expanded_queries[1], 
                        k=k2, 
                        metadata_filter=classification['kb2_filter'], 
                        score_threshold=score_threshold_kb2
                    )
                    existing = {getattr(d, 'page_content', '') for d in results}
                    for doc in results_expanded:
                        if getattr(doc, 'page_content', '') not in existing:
                            results.append(doc)
                
                return results
            except Exception as e:
                print(f"[QDRANT]  Error en KB-2: {type(e).__name__}: {str(e)}")
                return []
        
        docs1, docs2 = await asyncio.gather(search_kb1(), search_kb2(), return_exceptions=False)
        
        if isinstance(docs1, Exception):
            print(f"[QDRANT]  Excepción en KB-1: {docs1}")
            docs1 = []
        if isinstance(docs2, Exception):
            print(f"[QDRANT]  Excepción en KB-2: {docs2}")
            docs2 = []
        
        print(f"[QDRANT]  KB-1: Se encontraron {len(docs1)} documentos (ordenados por relevancia)")
        for i, doc in enumerate(docs1, 1):
            content = getattr(doc, 'page_content', '') or str(doc)
            metadata = getattr(doc, 'metadata', {})
            combined_score = metadata.get('score', 'N/A')
            vector_score = metadata.get('vector_score', 'N/A')
            term_score = metadata.get('term_score', 'N/A')
            
            if content:
                preview = content[:300] + "..." if len(content) > 300 else content
                print(f"[QDRANT] KB-1 Doc #{i} [Combined:{combined_score} Vector:{vector_score} Terms:{term_score}]")
                print(f"         {preview}\n")
            else:
                print(f"[QDRANT] KB-1 Doc #{i}:  CONTENIDO VACÍO\n")
        
        print(f"[QDRANT]  KB-2: Se encontraron {len(docs2)} documentos (ordenados por relevancia)")
        for i, doc in enumerate(docs2, 1):
            content = getattr(doc, 'page_content', '') or str(doc)
            metadata = getattr(doc, 'metadata', {})
            combined_score = metadata.get('score', 'N/A')
            vector_score = metadata.get('vector_score', 'N/A')
            term_score = metadata.get('term_score', 'N/A')
            
            if content:
                preview = content[:300] + "..." if len(content) > 300 else content
                print(f"[QDRANT] KB-2 Doc #{i} [Combined:{combined_score} Vector:{vector_score} Terms:{term_score}]")
                print(f"         {preview}\n")
            else:
                print(f"[QDRANT] KB-2 Doc #{i}:  CONTENIDO VACÍO\n")
        
        print(f"\n[QDRANT] Resumen: KB-1={len(docs1)} docs, KB-2={len(docs2)} docs\n")

        prompt_parts = [
            "1. ROL, IDENTIDAD Y OBJETIVO",
            "Identidad: Eres un Agente Virtual de Atención al Cliente altamente profesional para TRANSTUR, S.A. (marcas Cubacar, Havanautos y REX).",
            "",
            "Tono: Profesional y Absolutamente Preciso.",
            "",
            "Misión Principal: Responder todas las consultas de los clientes basándote EXCLUSIVAMENTE en la información de las bases de conocimiento.",
            "",
            "2. RESTRICCIONES DE CONOCIMIENTO (PROTOCOLO CRÍTICO)",
            "2.1. FUENTES DE CONOCIMIENTO",
            "Tu información proviene de dos bases de datos (Qdrant) que serán consultadas simultáneamente. Debes identificar la fuente de cada fragmento:",
            "",
            "KB-1: POLÍTICAS Y LEGAL (Términos, Condiciones de Renta, Políticas de Privacidad).",
            "",
            "KB-2: OPERACIONES Y TARIFAS (Datos Operacionales, Precios, Ubicaciones, Logística).",
            "",
            "2.2. REGLAS DE ORO",
            "PROHIBICIÓN ABSOLUTA: NUNCA debes inventar, adivinar, especular o utilizar conocimiento previo o externo a los fragmentos de texto recuperados de KB-1 y KB-2 PARA CONSULTAS SOBRE SERVICIOS, POLÍTICAS Y OPERACIONES.",
            "",
            "EXCEPCIÓN - USO DEL HISTORIAL: Para preguntas personales o de contexto conversacional (como '¿Sabes mi nombre?', '¿De qué estábamos hablando?', saludos, etc.), SÍ PUEDES y DEBES usar la información del HISTORIAL DE CONVERSACIÓN. El historial es tu memoria de la conversación actual con este cliente específico.",
            "",
            "FORMATO: Sintetiza la información clara y concisamente. Utiliza viñetas para respuestas que cubran múltiples puntos.",
            "",
            "2.3. MANEJO DE INFORMACIÓN PARCIAL",
            "REGLA DE ORO: Siempre intenta ayudar al cliente con la información disponible, aunque sea parcial o indirecta.",
            "",
            "- Si encuentras información RELACIONADA pero no exactamente lo que busca: Proporciona la información relacionada que tengas y explica cómo se relaciona con su consulta",
            "- Si la información es parcial: Comparte lo que sabes y ofrece contactar canales oficiales para más detalles",
            "- Solo si NO hay NADA relacionado (documentos completamente irrelevantes): Indica que no tienes esa información específica y proporciona contactos oficiales",
            "",
            "IMPORTANTE: Prioriza ser ÚTIL con información parcial o relacionada antes que decir que no tienes información. Si hay algo que pueda ayudar al cliente, compártelo.",
            "",
            "3. INSTRUCCIONES DE ACCIÓN Y PRIORIZACIÓN",
            "Al formular una respuesta, utiliza la siguiente jerarquía de acción y fuente:",
            "",
            "INFERENCIA INTELIGENTE: Si la pregunta es sobre un tema específico y no encuentras información directa, pero encuentras información relacionada (ej: pregunta sobre medidas de seguridad y encuentras info sobre responsabilidades, daños, seguros), usa esa información para dar una respuesta útil.",
            "",
            "CONTRATO Y VIGENCIA: Para preguntas sobre la duración del alquiler o el contrato, busca en KB-1."
            "",
            "MODIFICACIONES Y CANCELACIONES: Para cambios de reserva, busca en KB-1 e instruye al cliente a contactar a cubacar@transtur.cu (desde el correo de registro).",
            "",
            "TARIFAS Y DISPONIBILIDAD (Datos Variables): Si la consulta es sobre precios, tarifas, disponibilidad de modelos, o ubicaciones específicas de oficinas, prioriza la información de KB-2: OPERACIONES.",
            "",
            "GARANTÍA DEL VEHÍCULO: Si el cliente pregunta sobre modelos o marcas, busca en KB-1 y aclara que solo se garantiza la categoría del auto, no el modelo específico.",
            "",
            "PENALIDADES: Si se menciona la entrega en otra oficina (Drop-Off), busca en KB-1 e informa el cargo de drop-off más una penalidad del 50%.",
            "",
            "CANALES DE CONTACTO: Para cualquier pregunta sobre cómo contactar a la empresa, prioriza KB-1 y proporciona:",
            "INSTRUCCIÓN CLAVE DE FORMATO:",
            "FORMATO DE RESPUESTA: Responde SIEMPRE en texto plano, claro y profesional. NO uses formato Markdown.",
            "- NO uses asteriscos (*) para negritas o cursivas",
            "- NO uses almohadillas (#) para títulos",
            "- NO uses guiones bajos (_) para formato",
            "- Usa texto simple con viñetas (guiones -) cuando sea necesario",
            "- Separa secciones con saltos de línea simples",
            "- Mantén un tono profesional y amigable sin formatos especiales",
            "Cuando respondas con información extraída de la base de datos (ej. tarifas, detalles de productos), debes REESTRUCTURAR y REFORMATAR el texto en formato simple y claro, eliminando cualquier carácter de marcado interno que pueda confundir al usuario.",
            "",
            "---",
            "",
            f"CONTEXTO DE KB-1 (POLÍTICAS Y LEGAL): {self.tool1_desc}",
        ]
        
        if docs1:
            for d in docs1:
                content = getattr(d, 'page_content', '') or str(d)
                if content:
                    prompt_parts.append(f"- {content}")
        else:
            prompt_parts.append("(Sin información relevante en KB-1)")
        
        prompt_parts.append("")
        prompt_parts.append(f"CONTEXTO DE KB-2 (OPERACIONES Y TARIFAS): {self.tool2_desc}")
        
        if docs2:
            for d in docs2:
                content = getattr(d, 'page_content', '') or str(d)
                if content:
                    prompt_parts.append(f"- {content}")
        else:
            prompt_parts.append("(Sin información relevante en KB-2)")
        
        prompt_parts.append("")
        prompt_parts.append("HISTORIAL DE CONVERSACIÓN:")
        
        if recent:
            for m in recent:
                role = "Cliente" if m['role'] == "user" else "Agente"
                prompt_parts.append(f"{role}: {m['content']}")
        else:
            prompt_parts.append("(Primera interacción)")
        
        prompt_parts.append("")
        prompt_parts.append(f"CONSULTA ACTUAL DEL CLIENTE:")
        prompt_parts.append(user_message)
        prompt_parts.append("")
        prompt_parts.append("TU RESPUESTA (siguiendo todas las instrucciones anteriores):")

        prompt = "\n".join(prompt_parts)

        reply = await self.llm.generate(prompt)

        await self.memory.add_message(chat_id, "agent", reply)

        return reply
