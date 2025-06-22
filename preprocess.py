import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import os
import re
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 加载清洗后的中文评论数据
df = pd.read_csv("data/gzxb_cleaned.csv")

# 2. 过滤无效文本（太短的评论、缺失）
df = df.dropna(subset=["clean_text"])
df = df[df["clean_text"].str.len() >= 5].drop_duplicates(subset=["clean_text"]).reset_index(drop=True)


def clean_text(text):
    # 去掉中英文符号标签，例如【赞】、[赞]、（评论）等
    text = re.sub(r"[【】\[\]（）()《》<>]", "", text)
    # 去掉特殊字符和多余空格、数字标记
    text = re.sub(r"[§#￥@&*~^]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# df = df[~df["clean_text"].str.contains(r"^([赞好评\d\s【】\[\]（）])+?$")]
df = df[~df["clean_text"].str.contains(r"^(?:[赞好评\d\s【】\[\]（）])+?$")]


df["clean_text"] = df["clean_text"].apply(clean_text)

# 3. 加载中文向量模型（CPU 也能跑，推荐模型）
print(" 正在加载中文向量模型...")
model = SentenceTransformer("BAAI/bge-large-zh-v1.5",device=device)

# 4. 批量将文本转为向量（向量维度为 768）
print(" 正在将文本向量化...")
sentences = df["clean_text"].tolist()
embeddings = model.encode(sentences, show_progress_bar=True)

# 5. 使用 FAISS 创建索引并保存
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# 创建保存目录
os.makedirs("vector_index", exist_ok=True)

# 保存索引文件和原始文本索引映射
faiss.write_index(index, "vector_index/faiss_index.index")
df.iloc[:len(sentences)].to_csv("vector_index/faiss_docs.csv", index=False)

print(f" 向量数量: {len(sentences)}，索引已保存至 vector_index/")
