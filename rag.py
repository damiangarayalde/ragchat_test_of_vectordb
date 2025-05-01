import shutil
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import os


# Cleanup previous data
chroma_dir = "./chroma-data"
chroma_client = chromadb.PersistentClient(path=chroma_dir)


# Delete the ChromaDB collection and storage directory.
try:
    chroma_client.delete_collection("documents")
    if os.path.exists("./chroma-data"):
        shutil.rmtree("./chroma-data")
    os.makedirs("./chroma-data", exist_ok=True)
except Exception as e:
    print(f"Error clearing ChromaDB: {e}")


# documents, collection and vector store can be run in any order
documents = SimpleDirectoryReader("./data").load_data()  # Load documents
collection = chroma_client.get_or_create_collection(
    name="documents_collection",
    metadata={"hnsw:space": "cosine"}
)
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(  # Create index
    documents,
    storage_context=storage_context,
    show_progress=True
)

index.storage_context.persist(persist_dir=chroma_dir)


# Create query engine
query_engine = index.as_query_engine(
    similarity_top_k=3,  # Number of chunks to retrieve
    response_mode="tree_summarize"  # Options: "compact", "refine", "tree_summarize"
)

# Option 1: Simple query
question = "Cuales son las dos tematicas principales del documento?, listalo como dos bullet points"
response = query_engine.query(question)
print(f"\nQuestion: {question}")
print(f"Answer: {response}\n")

# # Option 2: Chat interface with memory
# chat_engine = index.as_chat_engine(
#     chat_mode="context",  # Options: "simple", "context", "condense_question"
#     system_prompt="You are a helpful assistant that answers questions based on the provided documents.",
#     verbose=True
# )

# # Custom prompt template
# custom_prompt = PromptTemplate("""\
# Context information is below.
# ---------------------
# {context_str}
# ---------------------
# Given the context information and not prior knowledge, answer the query.
# Query: {query_str}
# Answer: \
# """)

# # Ask follow-up questions
# questions = [
#     "Summarize the key points?",
#     "What are the most surprising findings?"
# ]

# for q in questions:
#     response = chat_engine.chat(custom_prompt.format(query_str=q))
#     print(f"Question: {q}")
#     print(f"Answer: {response}\n")
#     print("="*50)
