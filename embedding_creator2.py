# embedding_creator.py

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import OllamaEmbeddings
from tqdm import tqdm

load_dotenv()

def create_embeddings():
    # Load the PDF
    loader = PyPDFLoader('doc_samp1_merged.pdf')
    documents = loader.load_and_split()
    
    # Split the documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    print(f"Total chunks created: {len(texts)}")
    
    # Initialize Pinecone and embeddings
    index_name = "lawllm2"
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    # Create embeddings and store in Pinecone in batches
    batch_size = 100
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        PineconeVectorStore.from_documents(batch, embeddings, index_name=index_name)
    
    print("Embeddings created and stored successfully!")

if __name__ == "__main__":
    create_embeddings()