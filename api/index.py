# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import os
# from dotenv import load_dotenv

# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from pinecone import Pinecone

# # 1. Load Environment Variables
# load_dotenv()

# # 2. Initialize App
# app = FastAPI()

# # 3. Initialize Clients (Global Scope for efficiency)
# try:
#     pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#     index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    
#     embeddings = OpenAIEmbeddings(
#         api_key=os.getenv("LLMOD_API_KEY"),
#         base_url=os.getenv("OPENAI_API_BASE"),
#         model="RPRTHPB-text-embedding-3-small"
#     )

#     llm = ChatOpenAI(
#         api_key=os.getenv("LLMOD_API_KEY"),
#         base_url=os.getenv("OPENAI_API_BASE"),
#         model="RPRTHPB-gpt-5-mini"
#     )
# except Exception as e:
#     print(f"Server Startup Error: {e}")

# # --- CONFIGURATION (Must match ingestion) ---
# CHUNK_SIZE = 1000
# OVERLAP_RATIO = 0.2
# TOP_K = 5
# # --------------------------------------------

# # 4. Request Model
# class QueryRequest(BaseModel):
#     question: str

# # 5. The RAG Endpoint (POST /api/prompt)
# @app.post("/api/prompt")
# async def get_response(request: QueryRequest):
#     try:
#         # A. Embed the User Question
#         query_vector = embeddings.embed_query(request.question)

#         # B. Query Pinecone
#         search_results = index.query(
#             vector=query_vector,
#             top_k=TOP_K,
#             include_metadata=True
#         )

#         # C. Build Context String & Context List for JSON
#         context_text = ""
#         context_list = []
        
#         for match in search_results['matches']:
#             meta = match['metadata']
#             # Add to text block for LLM
#             context_text += f"\n---\nTitle: {meta.get('title')}\nSpeaker: {meta.get('speaker')}\nTranscript Snippet: {meta.get('chunk_text')}\n"
            
#             # Add to JSON output list
#             context_list.append({
#                 "talk_id": meta.get('talk_id'),
#                 "title": meta.get('title'),
#                 "chunk": meta.get('chunk_text'),
#                 "score": match['score']
#             })

#         # D. Construct the System Prompt (EXACTLY from Assignment)
#         system_prompt = (
#             "You are a TED Talk expert assistant. Your goal is to answer questions based "
#             "ONLY on the provided context. If the answer cannot be found in the context, "
#             "say 'I cannot answer this based on the provided TED talks.' "
#             "Do not use outside knowledge. "
#             "Keep answers concise and relevant."
#         )

#         user_prompt_template = f"Context:\n{context_text}\n\nQuestion: {request.question}"

#         # E. Generate Answer
#         # We use invoke with a list of messages for Chat models
#         messages = [
#             ("system", system_prompt),
#             ("user", user_prompt_template)
#         ]
        
#         ai_response = llm.invoke(messages)

#         # F. Return JSON
#         return {
#             "response": ai_response.content,
#             "context": context_list,
#             "Augmented_prompt": {
#                 "System": system_prompt,
#                 "User": user_prompt_template
#             }
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # 6. Stats Endpoint (GET /api/stats)
# @app.get("/api/stats")
# async def get_stats():
#     return {
#         "chunk_size": CHUNK_SIZE,
#         "overlap_ratio": OVERLAP_RATIO,
#         "top_k": TOP_K
#     }


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

# 3. Initialize Clients
# We do this OUTSIDE a try-except block so that if it fails,
# Vercel logs will show the specific error (missing key, etc.)
try:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    LLMOD_API_KEY = os.getenv("LLMOD_API_KEY")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

    if not PINECONE_API_KEY or not LLMOD_API_KEY:
        raise ValueError("Missing API Keys in Environment Variables!")

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    embeddings = OpenAIEmbeddings(
        api_key=LLMOD_API_KEY,
        base_url=OPENAI_API_BASE,
        model="RPRTHPB-text-embedding-3-small"
    )

    llm = ChatOpenAI(
        api_key=LLMOD_API_KEY,
        base_url=OPENAI_API_BASE,
        model="RPRTHPB-gpt-5-mini"
    )
except Exception as e:
    print(f"CRITICAL STARTUP ERROR: {e}")
    # We set these to None so the code doesn't crash with "Not Defined" later,
    # but we can raise a proper HTTP error if they are missing.
    embeddings = None
    index = None
    llm = None

# --- CONFIGURATION ---
CHUNK_SIZE = 1000
OVERLAP_RATIO = 0.2
TOP_K = 5
# ---------------------

class QueryRequest(BaseModel):
    question: str

@app.post("/api/prompt")
async def get_response(request: QueryRequest):
    # Check if clients are ready
    if embeddings is None or index is None or llm is None:
        raise HTTPException(
            status_code=500, 
            detail="Server failed to initialize AI clients. Check Vercel Logs for 'CRITICAL STARTUP ERROR'."
        )

    try:
        # A. Embed
        query_vector = embeddings.embed_query(request.question)

        # B. Query Pinecone
        search_results = index.query(
            vector=query_vector,
            top_k=TOP_K,
            include_metadata=True
        )

        # C. Build Context
        context_text = ""
        context_list = []
        
        matches = search_results.get('matches', [])
        if not matches:
            # Handle case where no results found
            pass 

        for match in matches:
            meta = match.get('metadata', {})
            context_text += f"\n---\nTitle: {meta.get('title', 'Unknown')}\nSpeaker: {meta.get('speaker', 'Unknown')}\nSnippet: {meta.get('chunk_text', '')}\n"
            
            context_list.append({
                "talk_id": meta.get('talk_id'),
                "title": meta.get('title'),
                "chunk": meta.get('chunk_text'),
                "score": match.get('score', 0)
            })

        # D. System Prompt
        system_prompt = (
            "You are a TED Talk expert assistant. Your goal is to answer questions based "
            "ONLY on the provided context. If the answer cannot be found in the context, "
            "say 'I cannot answer this based on the provided TED talks.' "
            "Do not use outside knowledge. Keep answers concise."
        )

        user_prompt_template = f"Context:\n{context_text}\n\nQuestion: {request.question}"

        # E. Generate
        messages = [
            ("system", system_prompt),
            ("user", user_prompt_template)
        ]
        
        ai_response = llm.invoke(messages)

        return {
            "response": ai_response.content,
            "context": context_list,
            "Augmented_prompt": {
                "System": system_prompt,
                "User": user_prompt_template
            }
        }

    except Exception as e:
        # Print actual error to Vercel logs so you can see it
        print(f"RUNTIME ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats")
async def get_stats():
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }