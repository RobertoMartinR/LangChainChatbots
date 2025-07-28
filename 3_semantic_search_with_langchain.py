# Official documentation: https://python.langchain.com/docs/tutorials/retrievers/

# Import necessary libraries
import getpass
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# Ensure Google API key is set
if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
        prompt="Enter your LangSmith API key (optional): "
    )
if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass(
        prompt='Enter your LangSmith Project Name (default = "default"): '
    )
    if not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "default"

from langchain_community.document_loaders import PyPDFLoader

# Load PDF file
file_path = "nke-10k-2023.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

print(len(docs))


"""
PyPDFLoader loads one Document object per PDF page.

Lets access to:
    -The string content of the page;
    -Metadata containing the file name and page number.
"""
print(f"{docs[0].page_content[:200]}\n")
print(docs[0].metadata)

# Now let's split the information using RecursiveCharacterTextSplitter into chunks of 1000 characters and 200 characters overlap
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)  
all_splits = text_splitter.split_documents(docs)
print(len(all_splits))

# Now we can use embeddings to convert the text chunks into vectors and given a query, find the most relevant chunks to identify the most relevant information.
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Define embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_1 = embeddings.embed_query(all_splits[0].page_content)
vector_2 = embeddings.embed_query(all_splits[1].page_content)

assert len(vector_1) == len(vector_2)
print(f"Generated vectors of length {len(vector_1)}\n")
print(vector_1[:10])

# Now we can use these vectors to perform semantic search after adding them to a vector store.
from langchain_chroma import Chroma

vector_store = Chroma(
    collection_name="training_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_db_langchain_test" # Where to save data locally
)

ids = vector_store.add_documents(all_splits)

# There are different methods to permorm semantic search


# Method 1: Return documents based on similarity to a string query:
results = vector_store.similarity_search(
    "How many distribution centers does Nike have in the US?"
)
print(results[0])


# Method 2 Async query:
results = await vector_store.asimilarity_search("When was Nike incorporated?")
print(results[0])


# Method 3: Return scores:
results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
doc, score = results[0]
print(f"Score: {score}\n")
print(doc)


# Method 4: Return documents based on similarity to an embedded query:
embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")
results = vector_store.similarity_search_by_vector(embedding)
print(results[0])


# RETRIEVER
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain


@chain
def retriever(query: str) -> List[Document]:
    return vector_store.similarity_search(query, k=1)


retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)


# Also VectorStores implement an as_retriever method:
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

retriever.batch(
    [
        "How many distribution centers does Nike have in the US?",
        "When was Nike incorporated?",
    ],
)
