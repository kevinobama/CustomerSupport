import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_cloudflare.embeddings import CloudflareWorkersAIEmbeddings


load_dotenv()
#groq needs proxy
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:1080"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:1080"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
os.environ["no_proxy"] = "localhost,127.0.0.1"

##################################
# 1. Load PDF
##################################
pdf_path = "/home/kevin/kevindata/www/python/CustomerSupport/Docs/RAG-agent-and-SQL-agent.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()
print("Pages:", len(docs))
##################################
# 2. Split Documents
##################################
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

splits = text_splitter.split_documents(docs)
# print("splits:", splits)
print("Chunks:", len(splits))

##################################
# 3. Embedding
##################################
embedding = CloudflareWorkersAIEmbeddings(
    model_name="@cf/baai/bge-base-en-v1.5"
)
# =============save_local==================
vectorstore = FAISS.from_documents(
    documents=splits,
    embedding=embedding
)
# Save FAISS index, docstore, and index_to_docstore_id to disk.
vectorstore.save_local("cloudflare_vectorstore_index")

vectorstore = FAISS.load_local(
    "cloudflare_vectorstore_index",
    embedding,
    allow_dangerous_deserialization=True
)

# question = "What programming languages does Kevin know?"
##################################
# 4. Retrieve
##################################

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k":3
    }
)

##################################
# 5. LLM
##################################
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=1
)
##################################
# 6. Prompt
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
# 7. Format docs
##################################
def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

##################################
# 8. RAG Chain
##################################
rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        |
        prompt
        |
        llm
        |
        StrOutputParser()
)

##################################
# 9. Ask
##################################
response = rag_chain.invoke(
    "What  are my technical stacks in this project?"
)

print(response)