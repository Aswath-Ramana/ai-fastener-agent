import os
import sys
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- Configuration ---
CATALOG_PATH = "data/catalog.pdf"
INDEX_PATH = "faiss_catalog_index"
# We will control the batch size for embedding creation manually
MANUAL_BATCH_SIZE = 500

def main():
    """
    Main function to load a PDF catalog, split it into chunks,
    generate embeddings, and save it as a FAISS vector store.
    This version uses manual batching to avoid API token limits.
    """
    # 1. Load Environment Variables and Check for API Key
    load_dotenv() 

    if "OPENAI_API_KEY" not in os.environ:
        print("🚨 Error: OPENAI_API_KEY not found.")
        print("Please create a '.env' file and add your OPENAI_API_KEY.")
        sys.exit(1)

    print(f"✅ OpenAI API Key found.")

    # 2. Load the PDF Document
    if not os.path.exists(CATALOG_PATH):
        print(f"🚨 Error: Catalog file not found at '{CATALOG_PATH}'")
        sys.exit(1)

    print(f"📚 Loading catalog from: {CATALOG_PATH}")
    loader = PyPDFLoader(CATALOG_PATH)
    documents = loader.load()
    print(f"👍 Catalog loaded successfully. Total pages: {len(documents)}")

    # 3. Split the Document into Smaller Chunks
    print(f"🔪 Splitting document into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents)
    total_chunks = len(chunks)
    print(f"   Created {total_chunks} text chunks from the catalog.")

    # 4. Initialize the Embeddings Model
    # We don't need the chunk_size parameter here anymore as we'll control it ourselves.
    print("🧠 Initializing OpenAI embeddings model...")
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # 5. Create the FAISS Vector Store using MANUAL BATCHING
    print("\n⏳ Creating embeddings and building the FAISS index in batches...")
    print(f"   Processing {total_chunks} chunks in batches of {MANUAL_BATCH_SIZE}...")
    
    try:
        # Step 5.1: Initialize the index with the very first batch of documents
        print(f"   Processing batch 1...")
        start_time = time.time()
        vector_store = FAISS.from_documents(
            documents=chunks[:MANUAL_BATCH_SIZE],
            embedding=embeddings_model
        )
        
        # Step 5.2: Loop through the remaining documents in batches and add them
        for i in range(MANUAL_BATCH_SIZE, total_chunks, MANUAL_BATCH_SIZE):
            batch_num = (i // MANUAL_BATCH_SIZE) + 1
            print(f"   Processing batch {batch_num}...")
            # Select the next batch of chunks
            batch = chunks[i : i + MANUAL_BATCH_SIZE]
            # Embed and add them to the existing index
            vector_store.add_documents(documents=batch)
            time.sleep(0.5) # A small delay can help prevent rate-limiting issues

        end_time = time.time()
        print(f"\n✅ All batches processed successfully in {end_time - start_time:.2f} seconds.")

        # Step 5.3: Save the completed index
        print(f"\n💾 Saving the final vector store to disk at '{INDEX_PATH}'...")
        vector_store.save_local(INDEX_PATH)
        
        print("\n🎉 --- Success! --- 🎉")
        print("Catalog preparation is complete.")
        print(f"The vector store is saved and ready for use in your application.")

    except Exception as e:
        print(f"\n🚨 An error occurred during FAISS index creation: {e}")
        print("   Please check your OpenAI API key, account status, and internet connection.")
        sys.exit(1)


if __name__ == "__main__":
    main()