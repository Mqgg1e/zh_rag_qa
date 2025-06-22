# rag_qa.py

import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# 设置路径
INDEX_PATH = "vector_index/faiss_index.index"
CSV_PATH = "vector_index/faiss_docs.csv"

# 加载评论文本
df = pd.read_csv(CSV_PATH)
corpus = df["comment"].tolist()

# 加载向量索引
index = faiss.read_index(INDEX_PATH)

# 加载 embedding 模型（和 preprocess.py 中一致）
embed_model = SentenceTransformer("shibing624/text2vec-base-multilingual")

# 加载本地或 API 的 LLM —— 这里假设你使用 HuggingFace 本地模型（可替换为 OpenAI/LLaMA）
# 你可以替换为 openai.ChatCompletion API
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)

# 获取用户输入
query = input("请输入你的问题：")

query_vec = embed_model.encode([query])
query_vec = query_vec.astype("float32")

top_k = 3
scores, indices = index.search(query_vec, top_k)

retrieved = [corpus[i] for i in indices[0]]
context = "\n".join(retrieved)

prompt = f"根据以下评论内容，简洁回答问题：{query}\n评论内容：\n{context}"


inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

outputs = model.generate(
    **inputs,
    max_new_tokens=150,
    no_repeat_ngram_size=3,
    early_stopping=False,  # 关闭，或者设置 num_beams>1 才用
    do_sample=True,        # 采样模式开启 temperature 才生效
    temperature=0.7,
    top_p=0.9,             # 也可以用 nucleus sampling 限制采样范围，防止无意义词
    top_k=50               # 限制采样时只从概率最高的50个token采样
)

# decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("模型原始输出:", decoded)
tokens = outputs[0].tolist()
text = "".join([tokenizer.decode([t]) for t in tokens if t not in tokenizer.all_special_ids])
print("模型输出拼接后:", text)

print("\n 回答：")

answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print([tokenizer.decode([token]) for token in outputs[0][:20]])
print("\n 回答：")
print(answer)
