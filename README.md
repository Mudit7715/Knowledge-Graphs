```markdown
# Knowledge Graph Construction with LLMs and Neo4j

This guide demonstrates how to build knowledge graphs using LLMs and Neo4j. The implementation extracts entities/relationships from text and visualizes connections in a graph database.

## 📦 Installation

### Core Requirements
```
# Base packages
!pip install --upgrade --quiet langchain langchain-community langchain-ollama neo4j

# Experimental graph features
!pip install --upgrade --quiet langchain_experimental
```

### LLM Setup (Choose One)

#### Option 1: Local Ollama (Recommended)
```
# Windows
curl -OL https://ollama.com/download/OllamaSetup.exe
./OllamaSetup.exe

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```
[Full Ollama installation guide](https://github.com/ollama/ollama/blob/main/docs/linux.md)

#### Option 2: Cloud Models (OpenAI/HuggingFace)
```
from langchain.llms import OpenAI  # Requires OpenAI API key
llm = OpenAI(model_name="gpt-4") 
```
[LangChain LLM integrations](https://python.langchain.com/docs/integrations/llms)

## ⚙️ Configuration

### Neo4j Connection
```
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

### LLM Initialization
```
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1",  # Try "mistral" or "phi3" alternatives
    temperature=0,
    base_url="http://localhost:11434/"  # Local Ollama endpoint
)
```

## 🚀 Usage

### 1. Text to Knowledge Graph
```
from langchain_experimental.graph_transformers import LLMGraphTransformer

llm_transformer = LLMGraphTransformer(llm=llm)

text = "Elon Musk founded SpaceX and Tesla..."
documents = [Document(page_content=text)]

# Generate graph structure
graph_documents = llm_transformer.convert_to_graph_documents(documents)

# Store in Neo4j
graph.add_graph_documents(graph_documents)
```

### 2. Querying the Graph
```
from langchain.chains import GraphCypherQAChain

qa_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True
)

response = qa_chain.invoke({"query": "Who founded SpaceX?"})
print(response["result"])  # Elon Musk founded SpaceX
```

## 🔄 Alternative Implementations

### Using OpenAI Models
```
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key="sk-..."
)
```
[LangChain OpenAI documentation](https://python.langchain.com/docs/integrations/chat/openai)

## 🛠 Troubleshooting

**Common Issues:**
- "Unknown relationship type" warnings: Verify with `print(graph.schema)`
- Ollama connection errors: Ensure service is running (`ollama serve`)
- Missing properties: Add explicit extraction instructions

**Debugging Tools:**
```
# Check stored nodes/relationships
print(graph.query("MATCH (n) RETURN n LIMIT 10"))

# View full schema
graph.refresh_schema()
print(graph.schema)
```

## 📚 Resources
- [LangChain Documentation](https://python.langchain.com/docs)
- [Neo4j Python Driver Guide](https://neo4j.com/docs/python-manual/current/)
- [Ollama Model Library](https://ollama.com/library)
```

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/53975326/4558eeae-cf9b-4b67-befb-9638754626b5/Untitled103.ipynb
[2] https://python.langchain.com/api_reference/core/language_models.html
[3] https://dev.to/evolvedev/how-to-install-ollama-on-windows-1ei5
[4] https://www.datacamp.com/tutorial/how-to-build-llm-applications-with-langchain
[5] https://python.langchain.com/docs/introduction/
[6] https://www.reddit.com/r/ollama/comments/1ibhxvm/guide_to_installing_and_locally_running_ollama/
[7] https://github.com/ollama/ollama/blob/main/docs/linux.md
[8] https://python.langchain.com/v0.1/docs/modules/model_io/llms/
[9] https://python.langchain.com/docs/integrations/llms/
[10] https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.llms.LLM.html
[11] https://python.langchain.com/v0.1/docs/integrations/llms/
[12] https://python.langchain.com/v0.1/docs/integrations/llms/openllm/
[13] https://python.langchain.com/api_reference/langchain/chains/langchain.chains.llm.LLMChain.html
[14] https://www.kdnuggets.com/ollama-tutorial-running-llms-locally-made-super-simple
[15] https://js.langchain.com/docs/integrations/llms/
[16] https://python.langchain.com/docs/tutorials/llm_chain/
[17] https://ollama.com/download
[18] https://www.langchain.com/langchain
[19] https://github.com/langchain-ai/langchain
[20] https://www.langchain.com
[21] https://python.langchain.com/v0.1/docs/modules/model_io/

---
Answer from Perplexity: pplx.ai/share
