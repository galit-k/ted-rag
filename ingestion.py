import os
import time
import pandas as pd
from tqdm import tqdm  # For a progress bar
from dotenv import load_dotenv

# LangChain & Pinecone imports
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# 1. Load Environment Variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
LLMOD_API_KEY = os.getenv("LLMOD_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

# --- CONFIGURATION ---
CSV_PATH = "ted_talks_en.csv"  # Make sure this file is in your folder
CHUNK_SIZE = 1000              # Approx tokens/chars
CHUNK_OVERLAP = 200            # Overlap to keep context
BATCH_SIZE = 100               # How many vectors to upload at once

# !!! BUDGET SAFETY: Set to 20 for testing. Set to None for full run. !!!
LIMIT = None  
# ---------------------

def ingest_data():
    print("🚀 Starting ingestion pipeline...")

    # 1. Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    if PINECONE_INDEX_NAME not in [i.name for i in pc.list_indexes()]:
        print(f"❌ Index '{PINECONE_INDEX_NAME}' not found in Pinecone!")
        print("Please create it in the Pinecone console with Dimension=1536, Metric=cosine.")
        return
    
    index = pc.Index(PINECONE_INDEX_NAME)
    print(f"✅ Connected to Pinecone Index: {PINECONE_INDEX_NAME}")

    # 2. Initialize Embedding Model (via LLMod.ai)
    embeddings = OpenAIEmbeddings(
        api_key=LLMOD_API_KEY,
        base_url=OPENAI_API_BASE,
        model="RPRTHPB-text-embedding-3-small" # REQUIRED by assignment
    )

    # 3. Load Data
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: {CSV_PATH} not found.")
        return

    print(f"📂 Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # Safety Limit
    if LIMIT:
        print(f"⚠️ TEST MODE: Processing only first {LIMIT} rows.")
        df = df.head(LIMIT)
    
    print(f"📊 Processing {len(df)} talks...")

    # 4. Process & Chunk Data
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    vectors_to_upsert = []
    
    # Iterate through each talk
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        talk_id = str(row['talk_id'])
        title = row['title']
        speaker = row['speaker_1']
        transcript = row['transcript']

        # Skip if transcript is missing
        if pd.isna(transcript) or transcript == "":
            continue

        # Split transcript into chunks
        chunks = text_splitter.split_text(transcript)

        for chunk_index, chunk_text in enumerate(chunks):
            # Create a unique ID for this chunk (e.g., "1234_0", "1234_1")
            vector_id = f"{talk_id}_{chunk_index}"
            
            # Prepare metadata (This is what you retrieve later)
            metadata = {
                "talk_id": talk_id,
                "title": title,
                "speaker": speaker,
                "chunk_text": chunk_text, # We store the text to display it later
                "chunk_index": chunk_index
            }
            
            # Add to list (we will embed in batches if needed, 
            # but usually we embed then add to list. 
            # To save API calls, we can embed individually or in batch.
            # Here we will collect text, embed batch, then zip.)
            # For simplicity in this script, we'll follow a standard LangChain flow 
            # or manual flow. Let's do manual for clarity and control.
            
            # NOTE: Embedding one by one is slow. Let's add to a temp list and embed in batches.
            # However, for simplicity, let's just append the raw text here and embed the whole batch later.
            vectors_to_upsert.append({
                "id": vector_id,
                "text": chunk_text,
                "metadata": metadata
            })

    print(f"🧠 Generated {len(vectors_to_upsert)} chunks. Starting Embedding & Upload...")

    # 5. Embed and Upsert in Batches
    # We process the 'vectors_to_upsert' list in batches of BATCH_SIZE
    total_vectors = len(vectors_to_upsert)
    
    for i in range(0, total_vectors, BATCH_SIZE):
        batch = vectors_to_upsert[i : i + BATCH_SIZE]
        
        # Extract just the text to embed
        texts_to_embed = [item["text"] for item in batch]
        
        try:
            # A. Generate Embeddings
            embeddings_list = embeddings.embed_documents(texts_to_embed)
            
            # B. Prepare for Pinecone
            pinecone_vectors = []
            for j, embedding in enumerate(embeddings_list):
                item = batch[j]
                pinecone_vectors.append({
                    "id": item["id"],
                    "values": embedding,
                    "metadata": item["metadata"]
                })
            
            # C. Upload to Pinecone
            index.upsert(vectors=pinecone_vectors)
            
            print(f"   Shape: {i}/{total_vectors} uploaded...", end="\r")
            
        except Exception as e:
            print(f"\n❌ Error processing batch {i}: {e}")
            # Don't stop the whole script, just try next batch
            continue

    print("\n✅ Ingestion Complete! Data is now in Pinecone.")

if __name__ == "__main__":
    ingest_data()   