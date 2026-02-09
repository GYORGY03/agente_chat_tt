import asyncio
from typing import Any, Dict, List, Optional

try:
    from qdrant_client import QdrantClient
    from langchain_qdrant import QdrantVectorStore
    from langchain_core.documents import Document
except Exception:
    QdrantClient = None
    QdrantVectorStore = None
    Document = None


def init_qdrant_client(url: str, api_key: Optional[str] = None):
    if QdrantClient is None:
        print("WARNING: qdrant-client o langchain no instalados; las herramientas RAG no estarán disponibles.")
        return None
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    client = QdrantClient(url=url, **kwargs)
    return client


def create_retrieval_tool_from_collection(
    collection_name: str, 
    qdrant_client, 
    embeddings
) -> Any:

    if QdrantVectorStore is None or Document is None:
        async def missing_tool(query: str, metadata_filter: Optional[Dict] = None):
            return [{"page_content": "Qdrant no disponible: instala qdrant-client/langchain-qdrant"}]

        return missing_tool


    vectorstore = QdrantVectorStore(
        client=qdrant_client,
        collection_name=collection_name,
        embedding=embeddings,
        content_payload_key="text", 
    )

    async def tool_async(
        query: str, 
        k: int = 18, 
        metadata_filter: Optional[Dict] = None, 
        score_threshold: float = 0.35
    ) -> List[Any]:
        print(f"[QDRANT TOOL] Ejecutando búsqueda asíncrona: query='{query[:50]}...', k={k}")
        
        try:
            search_k = k * 4  
            
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                lambda: vectorstore.similarity_search_with_score(query, k=search_k)
            )
            print(f"[QDRANT TOOL] Resultados brutos obtenidos: {len(results)}")
            
            query_lower = query.lower()
            query_terms = set(query_lower.split())
            
            scored_docs = []
            for idx, (doc, vector_score) in enumerate(results):
                content = getattr(doc, 'page_content', '')
                
                if not content or len(content.strip()) < 20:
                    continue
                
                content_lower = content.lower()
                
                term_matches = sum(1 for term in query_terms if len(term) > 3 and term in content_lower)
                term_score = term_matches / max(len(query_terms), 1)
                
                has_substantive_text = len([w for w in content_lower.split() if len(w) > 5]) > 10
                text_quality_bonus = 0.1 if has_substantive_text else 0.0
                
                combined_score = (-vector_score if vector_score < 0 else vector_score) + (term_score * 0.3) + text_quality_bonus
                
                scored_docs.append({
                    'doc': doc,
                    'vector_score': vector_score,
                    'term_score': term_score,
                    'combined_score': combined_score,
                    'content': content
                })
            
            scored_docs.sort(key=lambda x: x['combined_score'], reverse=True)
            
            filtered_docs = []
            docs_above_threshold = [item for item in scored_docs if item['combined_score'] >= score_threshold]
            
            print(f"[QDRANT TOOL] Filtrado por threshold={score_threshold}: {len(docs_above_threshold)} de {len(scored_docs)} documentos pasan")
            
            for idx, item in enumerate(docs_above_threshold[:k]):
                doc = item['doc']
                print(f"[QDRANT TOOL] Doc {idx+1}: vector={item['vector_score']:.4f}, terms={item['term_score']:.2f}, combined={item['combined_score']:.4f}, len={len(item['content'])}, preview={item['content'][:80]}...")
                
                if not hasattr(doc, 'metadata'):
                    doc.metadata = {}
                doc.metadata['score'] = f"{item['combined_score']:.4f}"
                doc.metadata['vector_score'] = f"{item['vector_score']:.4f}"
                doc.metadata['term_score'] = f"{item['term_score']:.2f}"
                
                filtered_docs.append(doc)
            
            print(f"[QDRANT TOOL] Retornando {len(filtered_docs)} documentos (de {len(results)} candidatos)")
            return filtered_docs
            
        except Exception as e:
            print(f"[QDRANT TOOL]  ERROR: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    return tool_async
