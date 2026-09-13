# RAG with FAISS + Cross-Encoder

Retrieval-Augmented Generation over a PDF: chunk → embed (MiniLM) → FAISS retrieve → cross-encoder rerank → OpenAI answer.

Corpus: NSW _Environmental Planning and Assessment Act 1979_.

## Layout

```
rag-faiss-crossencoder/
├── Environmental Planning and Assessment Act 1979 No 203.pdf
├── rag-faiss-crossencoder.ipynb
├── pyproject.toml
├── uv.lock
└── README.md
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
cd rag-faiss-crossencoder
uv sync
```

Set `OPENAI_API_KEY` in a `.env` at the repo root (see parent README), or export it in your shell.

## Run

```bash
uv run jupyter notebook rag-faiss-crossencoder.ipynb
```

Or register the env as a Jupyter kernel and open the notebook in your IDE:

```bash
uv run python -m ipykernel install --user --name=rag-faiss-crossencoder --display-name="Python (rag-faiss-crossencoder)"
```

## Run the notebook

Open `jupyter notebook rag-faiss-crossencoder.ipynb`, click **Select Kernel** → **Jupyter Kernel…**, and choose **Python (rag-faiss-crossencoder)**. Reload the IDE if you do not see it.

## Pipeline

1. Load PDF with LangChain `PyPDFLoader`
2. Chunk with `RecursiveCharacterTextSplitter`
3. Embed with `SentenceTransformer("all-MiniLM-L6-v2")`
4. Index / search with FAISS (`IndexFlatL2`)
5. Rerank with `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`
6. Answer with OpenAI using the top reranked chunks as context
