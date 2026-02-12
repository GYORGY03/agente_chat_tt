import asyncio
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path


class SimpleAgent:

    
    def __init__(
        self,
        llm,
        memory,
        tool1: Optional[Callable] = None,
        tool2: Optional[Callable] = None,
        tool1_desc: str = "",
        tool2_desc: str = ""
    ):

        self.llm = llm
        self.memory = memory
        self.tool1 = tool1
        self.tool2 = tool2
        self.tool1_desc = tool1_desc
        self.tool2_desc = tool2_desc

        prompt_path = Path(__file__).parent.parent / "prompts" / "system_prompt.txt"
        
        if prompt_path.exists():
            self.system_prompt_template = prompt_path.read_text(encoding="utf-8")
            print(f"[AGENT] Prompt cargado desde {prompt_path}")
        else:
            print(f"[AGENT] Archivo de prompt no encontrado: {prompt_path}")
                
    async def expand_query(self, query: str) -> List[str]:

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
            'kb2_filter': None,
            'threshold_kb1': 0.60,  # Default moderado
            'threshold_kb2': 0.60   # Default moderado
        }
        
        kb1_matches = sum(1 for kw in legal_keywords if kw in query_lower)
        kb2_matches = sum(1 for kw in tarifa_keywords if kw in query_lower)
        
        match_diff = abs(kb1_matches - kb2_matches)
        
        if kb1_matches > kb2_matches:
            result['prioritize'] = 'kb1'
            if match_diff >= 3:  
                result['threshold_kb1'] = 0.50
                result['threshold_kb2'] = 0.70
                print(f"[CLASIFICACIÓN] POLÍTICAS/LEGAL (Alta confianza: {kb1_matches} vs {kb2_matches}) - Thresholds: KB1=0.50, KB2=0.70")
            else:  # Confianza moderada
                result['threshold_kb1'] = 0.55
                result['threshold_kb2'] = 0.65
                print(f"[CLASIFICACIÓN] POLÍTICAS/LEGAL (Confianza moderada: {kb1_matches} vs {kb2_matches}) - Thresholds: KB1=0.55, KB2=0.65")
        elif kb2_matches > kb1_matches:
            result['prioritize'] = 'kb2'
            # KB2 prioritaria: threshold más permisivo en KB2, más estricto en KB1
            if match_diff >= 3:  # Alta confianza
                result['threshold_kb1'] = 0.70
                result['threshold_kb2'] = 0.50
                print(f"[CLASIFICACIÓN] TARIFAS/OPERACIONES (Alta confianza: {kb2_matches} vs {kb1_matches}) - Thresholds: KB1=0.70, KB2=0.50")
            else:  # Confianza moderada
                result['threshold_kb1'] = 0.65
                result['threshold_kb2'] = 0.55
                print(f"[CLASIFICACIÓN] TARIFAS/OPERACIONES (Confianza moderada: {kb2_matches} vs {kb1_matches}) - Thresholds: KB1=0.65, KB2=0.55")
        else:
            # Sin clasificación clara: thresholds moderados y balanceados
            result['threshold_kb1'] = 0.55
            result['threshold_kb2'] = 0.55
            print(f"[CLASIFICACIÓN] GENERAL (sin preferencia clara: {kb1_matches} vs {kb2_matches}) - Thresholds: KB1=0.55, KB2=0.55")
        
        return result

    async def run(self, chat_id: str, user_message: str) -> str:

        await self.memory.add_message(chat_id, "user", user_message)

        recent = await self.memory.get_recent(chat_id, limit=8)

        classification = await self.classify_question(user_message)

        docs1 = []
        docs2 = []
        
        expanded_queries = await self.expand_query(user_message)
        
        k1 = 16 if classification['prioritize'] == 'kb1' else 10
        k2 = 10 if classification['prioritize'] == 'kb1' else 16
        
        # Usar thresholds dinámicos de la clasificación
        score_threshold_kb1 = classification.get('threshold_kb1', 0.60)
        score_threshold_kb2 = classification.get('threshold_kb2', 0.60)
        
        main_query = expanded_queries[0]
        
        async def search_kb1():
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
            else:
                print(f"[QDRANT] KB-2 Doc #{i}:  CONTENIDO VACÍO\n")
        
        print(f"\n[QDRANT] Resumen: KB-1={len(docs1)} docs, KB-2={len(docs2)} docs\n")

        kb1_context = self._format_docs(docs1)
        kb2_context = self._format_docs(docs2)
        history = self._format_history(recent)
        
        prompt = self.system_prompt_template.format(
            kb1_desc=self.tool1_desc,
            kb1_context=kb1_context,
            kb2_desc=self.tool2_desc,
            kb2_context=kb2_context,
            history=history,
            query=user_message
        )

        reply = await self.llm.generate(prompt)

        await self.memory.add_message(chat_id, "agent", reply)

        return reply
    
    def _format_docs(self, docs: List[Any]) -> str:
        if not docs:
            return "(Sin información relevante)"
        
        lines = []
        for doc in docs:
            content = getattr(doc, 'page_content', '') or str(doc)
            if content:
                lines.append(f"- {content}")
        
        return "\n".join(lines) if lines else "(Sin información relevante)"

    def _format_history(self, messages: List[Dict[str, str]]) -> str:
        if not messages:
            return "(Primera interacción)"
        
        lines = []
        for msg in messages:
            role = "Cliente" if msg['role'] == "user" else "Agente"
            lines.append(f"{role}: {msg['content']}")
        
        return "\n".join(lines)

