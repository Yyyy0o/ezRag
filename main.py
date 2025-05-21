import os
import faiss
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from utils import get_embeddings
from openai import OpenAI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()

# OpenAI 初始化
openai = OpenAI(api_key=os.getenv("api_key"),base_url=os.getenv("base_url"))
# 加载向量库
index = faiss.read_index("vector_store/index.faiss")
with open("vector_store/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

class QARequest(BaseModel):
    query: str

@app.post("/qa")
def answer_query(req: QARequest):
    query_embedding = get_embeddings([req.query])[0]
    D, I = index.search(np.array([query_embedding]).astype("float32"), k=3)
    retrieved = "\n\n".join([chunks[i] for i in I[0]])

    prompt = f"""
你是一个智能问答助手，请根据以下知识回答问题：

{retrieved}

问题：{req.query}
如果找不到答案，请回复“我不知道”。
"""

    response = openai.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return {"answer": response.choices[0].message.content}
