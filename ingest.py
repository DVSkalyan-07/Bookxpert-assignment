import os
from pypdf import PdfReader
import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAIEmbeddingFunction
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

def extract_pdf_pages(file_path: str) -> list[dict]:
    extracted_data = []
    file_name = os.path.basename(file_path)
    try:
        reader = PdfReader(file_path)
        for index, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                clean_text = " ".join(text.split())
                extracted_data.append({
                    "text": clean_text,
                    "metadata": {
                        "source": file_name,
                        "page": index + 1
                    }
                })
    except Exception as e:
        print(f"Error reading PDF {file_name}: {e}")
    return extracted_data

def chunk_extracted_pages(pages: list[dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[dict]:
    chunks = []
    for page in pages:
        text = page["text"]
        metadata = page["metadata"]
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + chunk_size, text_length)
            chunk_text = text[start:end]
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "chunk_range": f"{start}-{end}"
                }
            })
            start += (chunk_size - chunk_overlap)
    return chunks

def save_to_vector_db(chunks: list[dict], db_path: str = "./db"):
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable missing in .env file")
    client = chromadb.PersistentClient(path=db_path)
    embedding_fn = GoogleGenerativeAIEmbeddingFunction(
        api_key=api_key,
        model_name="models/text-embedding-004"
    )
    collection = client.get_or_create_collection(
        name="document_knowledge_base",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    ids = [f"id_{i}" for i in range(len(chunks))]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"\nSuccessfully indexed {len(chunks)} chunks into the vector database.")

def run_ingestion_pipeline():
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created '{data_dir}' folder. Place your PDFs there and run again.")
        return
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in '{data_dir}'.")
        return
    all_chunks = []
    for file_name in tqdm(pdf_files, desc="Ingesting PDFs"):
        file_path = os.path.join(data_dir, file_name)
        raw_pages = extract_pdf_pages(file_path)
        chunks = chunk_extracted_pages(raw_pages)
        all_chunks.extend(chunks)
    if all_chunks:
        save_to_vector_db(all_chunks)

if __name__ == "__main__":
    run_ingestion_pipeline()
