from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Load documents
loader = TextLoader("data/sample.txt")
docs = loader.load()

# Embed and store
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

# QA Chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=db.as_retriever()
)

# Ask a question
query = "What is this hackathon project about?"
print("Answer:", qa.run(query))