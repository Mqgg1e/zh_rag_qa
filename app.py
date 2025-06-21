import streamlit as st
from rag_qa import rag_answer

st.title("📚 中文评论智能问答系统")
query = st.text_input("请输入你的问题：")

if st.button("提交"):
    with st.spinner("正在查询..."):
        answer = rag_answer(query)
    st.success("答复：")
    st.write(answer)