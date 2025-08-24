import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import file_read

# Drop rows with missing chunk_text
df_filtered = file_read.readFile().dropna(subset=["chunk_text"])

# Create list of texts
texts = df_filtered["chunk_text"].tolist()

# Create metadata for each chunk
metadatas = df_filtered[[
    "source_url",
     "tags"
]].to_dict(orient="records")


# Load a pre-trained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert text to embeddings
embeddings = model.encode(texts, show_progress_bar=True)


# Create ChromaDB client
client = chromadb.Client(Settings())

# Create a collection
collection = client.create_collection(name="rabindra_info")

# Add data to the collection
collection.add(
    documents=texts,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=[f"chunk_{i}" for i in range(len(texts))]
)
