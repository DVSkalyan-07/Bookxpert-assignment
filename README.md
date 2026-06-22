# AI Engineering Internship: Document Q&A Bot with RAG

A robust, localized, production-ready Retrieval-Augmented Generation (RAG) system built with Python. This software reads, chunks, and indexes custom textual content from binary documents (PDFs) into a localized database system (`ChromaDB`), dynamically executing vector similarity searches to provide contextualized responses via the Google Gemini API with strict grounding rules.

## Project Architecture & Pipeline Flow
1. **Document Extraction**: Page-by-page text compilation keeping index boundaries intact.
2. **Text Chunking**: Segment isolation using fixed window configurations ($1000$ character size, $200$ character semantic overlap).
3. **Database Population**: Local low-latency vector index persistence matching mathematical data topologies via Cosine distance maps using `text-embedding-004`.
4. **Context Grounding**: System configurations forcing zero-hallucination processing parameters utilizing `gemini-2.5-flash-preview-09-2025`.

---

## Technical Stack Setup
* **Language Runtime**: Python 3.11+
* **Vector Store Core**: ChromaDB
* **LLM & Embeddings Engine**: Google Gemini API
* **Document Extraction Engine**: PyPDF

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/document-qa-bot.git](https://github.com/YOUR_USERNAME/document-qa-bot.git)
   cd document-qa-bot
