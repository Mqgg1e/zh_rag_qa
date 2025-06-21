import streamlit as st
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 页面标题
st.title("🎯 中文 RAG 评论问答机器人")

# 加载数据
@st.cache_resource
def load_resources():
    df = pd.read_csv("vector_index/faiss_docs.csv")
    corpus = df["clean_text"].tolist()

    index = faiss.read_index("vector_index/faiss_index.index")
    embed_model = SentenceTransformer("shibing624/text2vec-base-multilingual")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True).eval()

    return corpus, index, embed_model, tokenizer, model

corpus, index, embed_model, tokenizer, model = load_resources()

# 用户输入
query = st.text_input("请输入你的问题：")

if query:
    with st.spinner("🔍 检索中..."):
        # 向量化
        query_vec = embed_model.encode([query]).astype("float32")
        scores, indices = index.search(query_vec, 3)
        context = "\n".join([corpus[i] for i in indices[0]])

        prompt = f"根据以下评论内容，简洁回答问题：{query}\n评论内容：\n{context}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.markdown("### 🤖 回答：")
        st.success(answer)
