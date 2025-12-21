from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone

# 1. Load Environment Variables
load_dotenv()

# 2. Initialize App
app = FastAPI()

# 3. Initialize Clients (Global Scope for efficiency)
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    
    embeddings = OpenAIEmbeddings(
        api_key=os.getenv("LLMOD_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        model="RPRTHPB-text-embedding-3-small"
    )

    llm = ChatOpenAI(
        api_key=os.getenv("LLMOD_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        model="RPRTHPB-gpt-5-mini"
    )
except Exception as e:
    print(f"Server Startup Error: {e}")

# --- CONFIGURATION (Must match ingestion) ---
CHUNK_SIZE = 1000
OVERLAP_RATIO = 0.2
TOP_K = 5
# --------------------------------------------

# 4. Request Model
class QueryRequest(BaseModel):
    question: str

# 5. The RAG Endpoint (POST /api/prompt)
@app.post("/api/prompt")
async def get_response(request: QueryRequest):
    try:
        # A. Embed the User Question
        query_vector = embeddings.embed_query(request.question)

        # B. Query Pinecone
        search_results = index.query(
            vector=query_vector,
            top_k=TOP_K,
            include_metadata=True
        )

        # C. Build Context String & Context List for JSON
        context_text = ""
        context_list = []
        
        for match in search_results['matches']:
            meta = match['metadata']
            # Add to text block for LLM
            context_text += f"\n---\nTitle: {meta.get('title')}\nSpeaker: {meta.get('speaker')}\nTranscript Snippet: {meta.get('chunk_text')}\n"
            
            # Add to JSON output list
            context_list.append({
                "talk_id": meta.get('talk_id'),
                "title": meta.get('title'),
                "chunk": meta.get('chunk_text'),
                "score": match['score']
            })

        # D. Construct the System Prompt (EXACTLY from Assignment)
        system_prompt = (
            "You are a TED Talk expert assistant. Your goal is to answer questions based "
            "ONLY on the provided context. If the answer cannot be found in the context, "
            "say 'I cannot answer this based on the provided TED talks.' "
            "Do not use outside knowledge. "
            "Keep answers concise and relevant."
        )

        user_prompt_template = f"Context:\n{context_text}\n\nQuestion: {request.question}"

        # E. Generate Answer
        # We use invoke with a list of messages for Chat models
        messages = [
            ("system", system_prompt),
            ("user", user_prompt_template)
        ]
        
        ai_response = llm.invoke(messages)

        # F. Return JSON
        return {
            "response": ai_response.content,
            "context": context_list,
            "Augmented_prompt": {
                "System": system_prompt,
                "User": user_prompt_template
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 6. Stats Endpoint (GET /api/stats)
@app.get("/api/stats")
async def get_stats():
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }