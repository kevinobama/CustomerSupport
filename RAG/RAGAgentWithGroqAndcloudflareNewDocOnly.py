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
pdf_path = "../Docs/RAG-agent-and-SQL-agent.pdf"
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
