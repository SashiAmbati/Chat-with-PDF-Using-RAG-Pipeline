import fitz  # PyMuPDF for text extraction
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.chat_models import ChatOpenAI

# Constants for embedding and PDF processing
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DB_DIM = 384  # Dimension of the embeddings
CHUNK_SIZE = 500  # Max characters per chunk for embedding
PDF_PATH = "/mnt/data/sample_data.pdf"  # Path to your PDF file

# Step 1: Extract text from PDF using PyMuPDF (fitz)
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

# Step 2: Extract tables from specific page using pdfplumber
def extract_tables_from_pdf(pdf_path, page_number):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number <= len(pdf.pages):
                page = pdf.pages[page_number - 1]
                return page.extract_table()
            else:
                raise ValueError("Page number out of range")
    except Exception as e:
        print(f"Error extracting table: {e}")
        return None

# Step 3: Extract unemployment data from raw text
def extract_unemployment_data(text):
    lines = text.splitlines()
    unemployment_info = {}
    for line in lines:
        if "|" in line:  # Assuming "|" separates degree and unemployment rate
            parts = line.split("|")
            if len(parts) == 2:
                degree, rate = parts[0].strip(), parts[1].strip()
                unemployment_info[degree] = rate
    return unemployment_info

# Step 4: Chunk the extracted text into smaller parts for embedding
def chunk_text(text, chunk_size):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Step 5: Generate embeddings for text chunks using SentenceTransformers
def generate_embeddings(chunks, model_name):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    return embeddings

# Step 6: Build a vector database (FAISS) for similarity search
def build_vector_db(embeddings):
    index = faiss.IndexFlatL2(VECTOR_DB_DIM)  # Use L2 distance metric
    embeddings_np = np.array(embeddings).astype('float32')  # Convert embeddings to numpy array
    index.add(embeddings_np)
    return index

# Step 7: Query the vector database and retrieve top-k most relevant chunks
def query_vector_db(query, index, model_name, chunks, top_k=3):
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])  # Get the embedding of the query
    query_embedding_np = np.array(query_embedding).astype('float32')
    distances, indices = index.search(query_embedding_np, top_k)
    return [(chunks[i], distances[j]) for j, i in enumerate(indices[0])]  # Return chunk and its distance

# Step 8: Generate a response using the LLM (OpenAI GPT)
def generate_response_with_llm(query, retrieved_chunks):
    context = "\n\n".join([chunk for chunk, _ in retrieved_chunks])
    prompt = f"Use the following context to answer the query:\n\n{context}\n\nQuery: {query}"
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)  # Adjust this to your API setup
    response = llm.call_as_llm(prompt)
    return response

# Main Function to execute the steps
def main():
    # Step 1: Extract text from the PDF
    pdf_text = extract_text_from_pdf(PDF_PATH)
    print("Text extracted from PDF.")

    # Step 2: Chunk the text into smaller parts
    chunks = chunk_text(pdf_text, CHUNK_SIZE)
    print(f"Text chunked into {len(chunks)} chunks.")

    # Step 3: Generate embeddings for the chunks
    embeddings = generate_embeddings(chunks, EMBEDDING_MODEL_NAME)
    print("Embeddings generated.")

    # Step 4: Build the vector database
    vector_db = build_vector_db(embeddings)
    print("Vector database built.")

    # Step 5: Process a user query (example: "What is the unemployment rate for Bachelor's degree?")
    query = "What is the unemployment rate for Bachelor's degree?"
    retrieved_chunks = query_vector_db(query, vector_db, EMBEDDING_MODEL_NAME, chunks)
    print("Retrieved relevant chunks from the database.")

    # Step 6: Generate a response from the LLM using the retrieved context
    response = generate_response_with_llm(query, retrieved_chunks)
    print("Generated Response:", response)

    # Step 7: Extract specific unemployment data (e.g., rates by degree type)
    unemployment_data = extract_unemployment_data(pdf_text)
    print("Unemployment Data:", unemployment_data)

    # Step 8: Extract table data from a specific page (e.g., Page 2)
    table_data = extract_tables_from_pdf(PDF_PATH, page_number=2)
    print("Table Data:", table_data)

if __name__ == "__main__":
    main()
