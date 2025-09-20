# Agentic RAG using LangGraph

Agentic Retrieval-Augmented Generation (RAG) graph built in **LangGraph** with:
- **Conditional edges for relevance grading** that gate the flow between retrieval → answer → rewrite/retry.
- **Semantic retrieval** via chunking + vector index to ground answers in source docs.

> Notebook: `AgRAGLG.ipynb`

---

## ✨ Highlights

- **Agentic control flow**: a graph that first decides whether to retrieve at all, then **grades** retrieved context and **loops** through query **rewrite/retry** when relevance is poor.
- **Semantic RAG pipeline**: documents are chunked and embedded, exposed as a retriever tool used by the agent to ground answers.
- **Concise answering**: the answer node is prompted to keep responses short and to admit uncertainty (“don’t know”) when evidence is weak.
- **Swap-in embeddings**: default uses OpenAI embeddings; easily switchable to SentenceTransformers (free) with one line change (snippet below).

---

## 🧱 Tech Stack

- **LangGraph** for agentic control flow (graph, nodes, conditional edges).
- **LangChain** (loaders, text splitters, retriever tool).
- **Vector store**: in-memory index (sufficient for the demo corpus).
- **Models**: chat model (grader + answerer) and embeddings (configurable).

---

## 🗂️ Data Ingestion & Indexing

Example corpus uses three Lilian Weng blog posts (reward hacking, hallucination, video diffusion).  
Documents are loaded from the web, split, embedded, and indexed.

```python
# Load and split
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]
docs = [WebBaseLoader(u).load() for u in urls]
docs = [d for batch in docs for d in batch]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
splits = text_splitter.split_documents(docs)
````

### Vector Index & Retriever

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

vectorstore = InMemoryVectorStore.from_documents(
    documents=splits, embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()
```

> **Use SentenceTransformers instead (free):**

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = InMemoryVectorStore.from_documents(splits, embedding=embedding)
retriever = vectorstore.as_retriever()
```

Expose the retriever to the agent as a **tool**:

```python
from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts."
)
```

---

## 🧠 Graph: Nodes & Control Flow

**Nodes**

* `generate_query_or_respond`: decide whether to answer directly or call tools.
* `retrieve`: tool node that queries the vector index.
* `grade_documents`: relevance grader → route to `generate_answer` or `rewrite_question`.
* `rewrite_question`: improves the query; loops back to `retrieve`.
* `generate_answer`: composes a short, grounded answer (admits “don’t know” if needed).

**Prompts (summaries)**

* **Grader**: “Is the retrieved context relevant to the question? yes/no.”
* **Rewrite**: “Rewrite the question to improve retrieval.”
* **Answer**: “Use retrieved context; 3 sentences max; say you don’t know if unsure.”

**Wiring the graph**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import MessagesState

workflow = StateGraph(MessagesState)

workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

# Decide whether to retrieve at all
workflow.add_edge(START, "generate_query_or_respond")
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {"tools": "retrieve", END: END}
)

# Grade retrieved docs → answer or rewrite
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("rewrite_question", "retrieve")
workflow.add_edge("generate_answer", END)

app = workflow.compile()
```

**Flow (ASCII)**

```
START → generate_query_or_respond ──(tools?)──▶ retrieve ──(relevant?)──▶ generate_answer ─▶ END
            │                                    │
            └────────────── no tools ────────────┘
                                                 └── no ─▶ rewrite_question ─┐
                                                                             └─▶ retrieve (loop)
```

---

## 🔎 Example Interaction (conceptual)

* **Q**: “What are the types of reward hacking?”
* Agent calls `retrieve` → grader says “relevant” → `generate_answer` summarizes the taxonomy from the posts.
* If retrieval is poor, `rewrite_question` improves it (e.g., “List reward hacking categories per Lilian Weng”) and tries again.

---

## 📦 Dependencies

* `langgraph`
* `langchain`, `langchain-community`, `langchain-text-splitters`
* One chat model (grader + answerer)
* One embedding model (OpenAI or SentenceTransformers)

---

## 📝 Notes & Limitations

* In-memory vector index is used for simplicity; swap to a production store (e.g., FAISS/Chroma/PGVector) for larger corpora.
* The grader is conservative by design; it may trigger rewrites even when partial context is relevant.
* The notebook demonstrates the agentic pattern on a small, well-structured corpus; performance on noisy, heterogeneous data may vary.

---

## 📚 Acknowledgments

* Built by following the **LangGraph** RAG agent pattern ([here](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/)).
* Example corpus: public posts by **Lilian Weng** (used for educational purposes).

```
