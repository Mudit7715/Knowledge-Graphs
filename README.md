# 🧠 Knowledge Graph Construction with LLMs and Neo4j

This guide demonstrates how to build knowledge graphs using LLMs and Neo4j. The implementation extracts entities/relationships from text and visualizes connections in a graph database.

---

## 📦 Installation

### Core Requirements

```bash
# Base packages
pip install --upgrade --quiet langchain langchain-community langchain-ollama neo4j

# Experimental graph features
pip install --upgrade --quiet langchain_experimental
```

---

### LLM Setup (Choose One)

#### ✅ Option 1: Local Ollama (Recommended)

```bash
# Windows
curl -OL https://ollama.com/download/OllamaSetup.exe
./OllamaSetup.exe

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

📄 [Full Ollama installation guide](https://github.com/ollama/ollama/blob/main/docs/linux.md)

---

#### ☁️ Option 2: Cloud Models (OpenAI/HuggingFace)

```python
from langchain.llms import OpenAI  # Requires OpenAI API key
llm = OpenAI(model_name="gpt-4")
```

📄 [LangChain LLM integrations](https://python.langchain.com/docs/integrations/llms)

---

## ⚙️ Configuration

### Neo4j Connection

```python
import os
from langchain_community.graphs import Neo4jGraph

os.environ["NEO4J_URI"] = "neo4j+s://your-cluster.databases.neo4j.io"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "your-password"

graph = Neo4jGraph(
    url=os.environ["NEO4J_URI"],
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"]
)
```

---

### LLM Initialization

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1",  # Try "mistral" or "phi3" alternatives
    temperature=0,
    base_url="http://localhost:11434/"  # Local Ollama endpoint
)
```

---

## 🚀 Usage

### 1. Text to Knowledge Graph

```python
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.schema import Document

llm_transformer = LLMGraphTransformer(llm=llm)

text = "Elon Musk founded SpaceX and Tesla..."
documents = [Document(page_content=text)]

# Generate graph structure
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# Store in Neo4j
graph.add_graph_documents(graph_documents)
```

---

### 2. Querying the Graph

```python
from langchain.chains import GraphCypherQAChain

qa_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True
)

response = qa_chain.invoke({"query": "Who founded SpaceX?"})
print(response["result"])  # Elon Musk founded SpaceX
```

---

## 🔄 Alternative Implementations

### Using OpenAI Models

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key="sk-..."
)
```

📄 [LangChain OpenAI documentation](https://python.langchain.com/docs/integrations/chat/openai)

---

## 🛠 Troubleshooting

### Common Issues

- **"Unknown relationship type" warnings:** Verify with `print(graph.schema)`
- **Ollama connection errors:** Ensure service is running (`ollama serve`)
- **Missing properties:** Add explicit extraction instructions

### Debugging Tools

```python
# Check stored nodes/relationships
print(graph.query("MATCH (n) RETURN n LIMIT 10"))

# View full schema
graph.refresh_schema()
print(graph.schema)
```

---

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/docs)
- [Neo4j Python Driver Guide](https://neo4j.com/docs/python-manual/current/)
- [Ollama Model Library](https://ollama.com/library)
