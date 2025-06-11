import os
import tiktoken
from typing import List
from openai import OpenAI
from PyPDF2 import PdfReader
from dotenv import load_dotenv

load_dotenv()

# OpenAI 初始化
openai = OpenAI(api_key=os.getenv("api_key"),base_url=os.getenv("base_url"))

def load_pdf_text(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

def chunk_text(text: str, chunk_size=500, overlap=100) -> List[str]:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = encoding.decode(tokens[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def get_embeddings(texts: List[str]) -> List[List[float]]:
    # 将texts分成每组最多6个元素的批次
    batch_size = 6
    batched_texts = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    embeddings = []
    for batch in batched_texts:
        response = openai.embeddings.create(
            model="text-embedding-v3",
            input=batch
        )
        embeddings.extend([e.embedding for e in response.data])
    return embeddings
