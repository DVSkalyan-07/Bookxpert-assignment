import os
import google.generativeai as genai
import chromadb
from chromadb.utils.embedding_functions import GoogleGenerativeAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

def query_rag_pipeline(user_query: str, db_path: str = "./db", k: int = 3) -> dict:
    if not os.path.exists(db_path):
        return {"answer": "Error: Run ingestion script first.", "citations": []}
    
    client = chromadb.PersistentClient(path=db_path)
    embedding_fn = GoogleGenerativeAIEmbeddingFunction(
        api_key=api_key,
        model_name="models/text-embedding-004"
    )
    collection = client.get_collection(
        name="document_knowledge_base",
        embedding_function=embedding_fn
    )
    results = collection.query(query_texts=[user_query], n_results=k)
    
    context_blocks = []
    citations = []
    documents = results.get('documents', [[]])[0]
    metadatas = results.get('metadatas', [[]])[0]
    
    for doc, meta in zip(documents, metadatas):
        source_name = meta.get('source', 'Unknown File')
        page_num = meta.get('page', 'Unknown Page')
        citation_str = f"Source: {source_name}, Page: {page_num}"
        context_blocks.append(f"[{citation_str}]\nContext: {doc}")
        citations.append(citation_str)
        
    context_payload = "\n\n---\n\n".join(context_blocks)
    
    system_prompt = (
        "You are a professional, accurate document Q&A assistant. "
        "Answer the user's question using ONLY the provided document context below. "
        "Cite the sources (filenames and pages) inline next to facts you cite. "
        "If the answer cannot be found in the context, clearly state: "
        "'I am sorry, but the provided documents do not contain the answer to your question.' "
        "Do not make up facts or use external knowledge sources."
    )
    
    prompt = (
        f"{system_prompt}\n\n"
        f"CONTEXT INFORMATION:\n{context_payload}\n\n"
        f"USER QUESTION: {user_query}\n\n"
        f"GROUNDED ANSWER:"
    )
    
    model = genai.GenerativeModel('gemini-2.5-flash-preview-09-2025')
    response = model.generate_content(prompt)
    return {"answer": response.text, "citations": list(set(citations))}
