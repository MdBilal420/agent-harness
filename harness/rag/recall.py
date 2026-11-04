import os
from pathlib import Path

import chromadb
from anthropic import Anthropic
import voyageai

from harness.tools.registry import tool
from harness.config import (
    RAG_INDEX_PATH,
    RAG_EMBEDDING_MODEL,
    RAG_DEFAULT_RESULTS
)
from harness.rag.indexer import COLLECTION_NAME


def _project_root() -> Path:
    return Path(__file__).parent.parent.parent


@tool 
def recall(query:str, max_results:int=None) -> str:
    "Search the RAG index for the most relevant documents"

    if max_results is None:
        max_results = RAG_DEFAULT_RESULTS

    # 1. ChromaDB collection 

    index_dir = _project_root() / RAG_INDEX_PATH
    if not index_dir.exists():
        return "RAG index not found"
    
    try:
        chroma_client = chromadb.PersistentClient(path=str(index_dir))
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        return f"Error getting collection: {e}"

    # 2. Embed the query
    try:
        vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
        embed_response = vo.embed(
            model=RAG_EMBEDDING_MODEL,
            texts=query,
            input_type="document",
        )
        query_embedding = embed_response.embeddings[0]
    except Exception as e:
        print(f"Error embedding query: {e}")
        return f"Error embedding query: {e}"
    
    # 3. Query the collection
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=max_results,
        )
    except Exception as e:
        print(f"Error querying collection: {e}")
        return f"Error querying collection: {e}"
    
    # 4. Format the results
    documents = results["documents"][0] if results.get("documents") else []
    metadatas = results["metadatas"][0] if results.get("metadatas") else []
    sources = [m.get("source","(Unknown Source)") for m in metadatas]
    
    return _format_results(documents, sources, query)
    
def _format_results(chunks: list[str], sources: list[str], query: str) -> str:
    "Format the results into a string"

    lines = [f"### Recall results for query: {query}"]
    if not chunks:
        lines.append("No matching chunks found")
        return "\n".join(lines)
    
    for i, (chunk, source) in enumerate(zip(chunks, sources), start=1):
        lines.append(f"**Result {i}** - from `{source}`")
        lines.append(f"```\n{chunk.strip()}\n```")
        lines.append("")
    
    return "\n".join(lines)
    
