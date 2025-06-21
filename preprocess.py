import pandas as pd
import re

# 1. 读取原始 CSV 文件（请根据你的路径调整）
df = pd.read_csv("data/bilibili_gzxb.csv")

# 2. 清洗函数
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<.*?>", "", text)  # 去除 HTML 标签
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", "", text)  # 去除特殊字符、标点、emoji
    return text.strip()

# 3. 清洗评论字段
df_cleaned = df[["comment"]].copy()
df_cleaned["clean_text"] = df_cleaned["comment"].map(clean_text)

# 4. 可选：去除过短的文本（如少于5字）
df_cleaned = df_cleaned[df_cleaned["clean_text"].str.len() >= 5]

# 5. 保存为新 CSV 文件
df_cleaned.to_csv("data/gzxb_cleaned.csv", index=False)
df_cleaned[["comment"]].to_csv("data/gzxb_raw.csv", index=False)

print("✅ 清洗完成，已保存为 gzxb_cleaned.csv 和 gzxb_raw.csv")
