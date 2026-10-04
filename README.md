installation:
# 1. Create a virtual environment
python3 -m venv venv

# 2. Activate it
source venv/bin/activate

# 3. Now install the packages
pip install fastapi uvicorn

uvicorn main:app --host 0.0.0.0 --port 8000
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Yes. Based on the imports in your source, this is the `requirements.txt` I would use with **uv**.

```text
python-dotenv
langchain-text-splitters
langchain-community
langchain-core
langchain-cloudflare
langchain-groq
faiss-cpu
pypdf
requests
```

### Install with `uv`

If you're starting a new project:

```bash
uv init
uv add python-dotenv langchain-text-splitters langchain-community langchain-core langchain-cloudflare langchain-groq faiss-cpu pypdf requests
```

Or, if you have the `requirements.txt` above:

```bash
uv venv
source .venv/bin/activate

uv pip install -r requirements.txt
```

### One important detail

You **don't need to explicitly install** these:

```text
os
json
hashlib
```

They're Python standard-library modules.

You also don't need to separately install `typing` for this code.

### For your 1 GB Lightsail server

I'd actually prefer:

```bash
uv add python-dotenv langchain-text-splitters langchain-community \
  langchain-core langchain-cloudflare langchain-groq \
  faiss-cpu pypdf requests
```

because `uv` will resolve compatible versions and generate your `uv.lock`.

Your resulting project will roughly be:

```text
RAG/
├── .venv/
├── Docs/
│   └── RAG-agent-and-SQL-agent.pdf
├── cloudflare_vectorstore_index/
│   ├── index.faiss
│   ├── index.pkl
│   └── manifest.json
├── .env
├── main.py
├── pyproject.toml
└── uv.lock
```

One correction from your pasted source: the lines

```python
os.environ["HTTPS_PROXY"] = "[http://127.0.0.1:1080](http://127.0.0.1:1080)"
```

appear to have been transformed by Markdown/HTML. In the actual Python file they should be:

```python
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:1080"
os.environ["HTTP_PROXY"] = "http://127.0.0.1:1080"
```