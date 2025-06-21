import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import requests

# 加载数据和模型
df = pd.read_csv("data/douban_cleaned.csv")
model = SentenceTransformer("shibing624/text2vec-base-multilingual")
index = faiss.read_index("faiss_index.index")

def search_similar(question, top_k=3):
    question_vec = model.encode([question])
    D, I = index.search(np.array(question_vec), top_k)
    return [df.iloc[i]["text"] for i in I[0]]

def call_llama(context, question):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": "Bearer your_llama_api_key",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/llama-3-8b-instruct",
        "messages": [
            {"role": "system", "content": "你是一个中文评论分析专家。"},
            {"role": "user", "content": f"请根据以下评论回答问题：\n{context}\n问题：{question}"}
        ]
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

def rag_answer(question):
    docs = search_similar(question)
    context = "\n".join(docs)
    return call_llama(context, question)