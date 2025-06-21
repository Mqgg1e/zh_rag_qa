# rag_qa.py

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# 设置路径
INDEX_PATH = "vector_index/faiss.index"
CSV_PATH = "vector_index/gzxb_cleaned.csv"

# 加载评论文本
df = pd.read_csv(CSV_PATH)
corpus = df["comment"].tolist()

# 加载向量索引
index = faiss.read_index(INDEX_PATH)

# 加载 embedding 模型（和 preprocess.py 中一致）
embed_model = SentenceTransformer("shibing624/text2vec-base-chinese")

# 加载本地或 API 的 LLM —— 这里假设你使用 HuggingFace 本地模型（可替换为 OpenAI/LLaMA）
# 你可以替换为 openai.ChatCompletion API
tokenizer = AutoTokenizer.from_pretrained("IDEA-CCNL/Randeng-Pegasus-238M-Chinese", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("IDEA-CCNL/Randeng-Pegasus-238M-Chinese", trust_remote_code=True)

# 获取用户输入
query = input("请输入你的问题：")

# 编码问题为向量
query_vec = embed_model.encode([query])
query_vec = query_vec.astype("float32")

# 检索相似评论
top_k = 3
scores, indices = index.search(query_vec, top_k)

# 取出最相关的文本
retrieved = [corpus[i] for i in indices[0]]
context = "\n".join(retrieved)

# 拼接提示词
prompt = f"以下是一些与问题相关的评论：\n{context}\n\n请根据这些内容回答问题：{query}"

# 构造输入
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
outputs = model.generate(**inputs, max_new_tokens=150)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n🧠 回答：")
print(answer)
