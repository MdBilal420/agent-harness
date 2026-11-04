from dotenv import load_dotenv
from harness.rag.indexer import build_index


def main():
    print("Indexing RAG...")
    load_dotenv()
    build_index()
    print("RAG indexed successfully")


if __name__ == "__main__":
    main()