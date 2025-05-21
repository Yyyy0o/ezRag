import os
import faiss
import pickle
from utils import load_pdf_text, chunk_text, get_embeddings
import numpy as np

doc_path = "docs/2025Q1.pdf"
text = load_pdf_text(doc_path)
chunks = chunk_text(text)

embeddings = get_embeddings(chunks)
print(f"类型: {type(embeddings)}, 元素类型: {[type(v) for v in embeddings]}")
print(f"每个向量维度: {[len(v) for v in embeddings]}")

dim = len(embeddings[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings).astype("float32"))

# 保存向量库和原文索引
faiss.write_index(index, "vector_store/index.faiss")
with open("vector_store/chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)
