
import os
from pathlib import Path

import chromadb
from anthropic import Anthropic
import voyageai

from harness.config import (
    RAG_CORPUS_PATH, 
    RAG_INDEX_PATH, 
    RAG_EMBEDDING_MODEL, 
    RAG_CHUNK_SIZE, 
    RAG_CHUNK_OVERLAP, 
)

COLLECTION_NAME: str = "knowledge"

CHARS_PER_TOKEN: int = 4

def _project_root() -> Path:
    return Path(__file__).parent.parent.parent

def _find_corpus_files(corpus_dir: Path) -> list[Path]:
    files = list(corpus_dir.rglob("*.md")) + list(corpus_dir.rglob("*.txt"))
    return sorted(files)

def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Chunk text into smaller chunks of a given size, with a given overlap."""

    # 1. token counts to character counts
    chunk_chars = chunk_size * CHARS_PER_TOKEN
    overlap_chars = chunk_overlap * CHARS_PER_TOKEN

    # 2. split paragraphs
    paragraphs = text.split("\n\n")

    # 3. chunk paragraphs
    chunks: list[str] = []
    current_chunk: str = ""
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > chunk_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = current_chunk[-overlap_chars:] + "\n\n" + paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += paragraph
    # 4. Flush remaining chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # 5. hard cut
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= chunk_chars:
            final_chunks.append(chunk)
        else:
            # hard character based split with overlap
            for i in range(0, len(chunk), chunk_chars - overlap_chars):
                final_chunks.append(chunk[i:i+chunk_chars])
    return final_chunks

def _embed_chunks(texts: list[str]) -> list[list[float]]:
    """Embed chunks using the Voyage API."""
    vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))
    response = vo.embed(
        model=RAG_EMBEDDING_MODEL,
        texts=texts,
        input_type="document",
    )
    return response.embeddings

def build_index() -> None:
    " Read the corpus files, chunk the text, embed the chunks, and store in ChromaDB."

    project_root = _project_root()
    corpus_dir = project_root / RAG_CORPUS_PATH
    index_dir = project_root / RAG_INDEX_PATH


    # 1. Validate corpus
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    
    if not os.environ.get("VOYAGE_API_KEY"):
        raise RuntimeError("VOYAGE_API_KEY is not set")

    # 2. Find corpus files
    files = _find_corpus_files(corpus_dir)
    print(f"Found {len(files)} corpus files")

    # 3. Chunk each file along their source path
    all_chunks: list[str] = []
    all_metadata: list[dict] = []
    for file_path in files:
        text = file_path.read_text()
        chunks = _chunk_text(text, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({
                "source": str(file_path.relative_to(project_root)),
                "chunk_index": i,
            })
    print(f"Found {len(all_chunks)} chunks")

    # 4. Embed chunks
    print(f"Embedding chunks via {RAG_EMBEDDING_MODEL}...")
    embeddings = _embed_chunks(all_chunks)

    # 4. reset the chromaDB collection and write chunks + embeddings
    chroma_client = chromadb.PersistentClient(path=str(index_dir))
    try:
        chroma_client.delete(collection_name=COLLECTION_NAME)
    except Exception as e:
        print(f"Error deleting collection: {e}")
        pass
    
    collection = chroma_client.create_collection(name=COLLECTION_NAME)
    ids = [f"{m['source']}#{m['chunk_index']}" for m in all_metadata]
    collection.add(
        ids=ids,
        documents=all_chunks,
        metadatas=all_metadata,
        embeddings=embeddings,
    )
    print(f"Indexed {len(all_chunks)} chunks into ChromaDB")