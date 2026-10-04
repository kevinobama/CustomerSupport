Here's the modified script with conditional logic: it computes a hash of the PDF file, checks whether that hash is already recorded in a small manifest file, and only runs the loader → splitter → embedding → `FAISS.from_documents` pipeline when the document is **new or has changed**. Otherwise, it skips straight to loading the existing FAISS index from disk for querying.

```python
import os
import json
import hashlib
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings
from langchain_groq import ChatGroq

load_dotenv()
# groq needs proxy
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:1080"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:1080"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

##################################
# 0. Config / Paths
##################################
pdf_path = "/home/kevin/kevindata/www/python/CustomerSupport/Docs/RAG-agent-and-SQL-agent.pdf"
VECTORSTORE_DIR = "cloudflare_vectorstore_index"
MANIFEST_PATH = os.path.join(VECTORSTORE_DIR, "manifest.json")


##################################
# Helpers: detect new/changed document
##################################
def file_hash(path: str) -> str:
    """Compute a SHA-256 hash of the file contents to detect changes."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {}


def save_manifest(manifest: dict) -> None:
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


##################################
# 1. Embedding model (needed regardless, for query-time embedding)
##################################
embedding = CloudflareWorkersAIEmbeddings(
    model_name="@cf/baai/bge-base-en-v1.5"
)

##################################
# 2. Decide: new document → re-embed, else → just load vectorstore
##################################
current_hash = file_hash(pdf_path)
manifest = load_manifest()
index_file = os.path.join(VECTORSTORE_DIR, "index.faiss")

is_new_or_changed = manifest.get(pdf_path) != current_hash
vectorstore_missing = not os.path.exists(index_file)

if is_new_or_changed or vectorstore_missing:
    print("New or updated document detected — running embedding pipeline...")

    # ---- Load PDF ----
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print("Pages:", len(docs))

    # ---- Split Documents ----
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    print("Chunks:", len(splits))

    # ---- Embed + build vectorstore ----
    vectorstore = FAISS.from_documents(
        documents=splits,
        embedding=embedding
    )
    vectorstore.save_local(VECTORSTORE_DIR)

    # ---- Update manifest so we skip re-embedding next time ----
    manifest[pdf_path] = current_hash
    save_manifest(manifest)

else:
    print("No new document detected — loading existing vectorstore (no embedding step).")
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embedding,
        allow_dangerous_deserialization=True
    )

##################################
# 3. Retrieve
##################################
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 3
    }
)

##################################
# 4. LLM
##################################
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=1
)

##################################
# 5. Prompt
##################################
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using only the context.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

##################################
# 6. Format docs
##################################
def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

##################################
# 7. RAG Chain
##################################
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

##################################
# 8. Ask
##################################
response = rag_chain.invoke(
    "What are my technical stacks in this project?"
)

print(response)
```

### Key changes & reasoning

| Aspect | Explanation |
|---|---|
| **Change detection** | `file_hash()` computes a SHA-256 hash of the PDF. A `manifest.json` stores the last known hash per file path. If the hash differs (or doesn't exist yet) → treated as a "new document." |
| **Embedding model instantiation** | `CloudflareWorkersAIEmbeddings` is still instantiated unconditionally — it's required at **query time** too (to embed the user's question), not just for document indexing. |
| **Heavy pipeline guarded** | `PyPDFLoader` → `RecursiveCharacterTextSplitter` → `FAISS.from_documents` (the actual embedding-generation step) only runs inside the `if is_new_or_changed or vectorstore_missing:` branch. |
| **Fast path** | If the document is unchanged and the index already exists on disk, it goes straight to `FAISS.load_local(...)` — no PDF parsing, splitting, or document embedding calls. |
| **Manifest persistence** | After a successful rebuild, the manifest is updated so subsequent runs with the same file skip re-embedding. |

### Optional enhancements
- **Multiple documents**: wrap this logic in a loop over a folder of PDFs, hashing each one and only embedding the changed/new files, then merging into the FAISS index with `vectorstore.add_documents(...)`.
- **Mtime instead of hash**: faster but less reliable than content hashing (won't detect same-content-different-timestamp edge cases).
- **Incremental updates**: instead of fully rebuilding, you could load the existing index and call `vectorstore.add_documents(new_splits)` only for the new file, then re-save — avoids reprocessing documents already indexed.

Let me know if you'd like the folder-scanning / incremental-update version.