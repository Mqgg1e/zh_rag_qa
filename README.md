# Chinese RAG Comment QA Bot

This project is a Chinese Retrieval-Augmented Generation (RAG) question-answering system. It retrieves the most relevant user comments from a dataset and generates concise answers to user queries using a large language model (LLM).

## Features

- **Retrieval-Augmented Generation:** Combines semantic search with generative AI for accurate answers.
- **Chinese Language Support:** Uses state-of-the-art models for Chinese text.
- **GPU Acceleration:** Optimized for platforms with GPU support (e.g., Kaggle).
- **Streamlit Web Interface:** Simple and interactive UI for asking questions.

## How It Works

1. **Data Loading:** Loads a CSV file of cleaned comments and a FAISS vector index for fast similarity search.
2. **Embedding:** Encodes user queries using a powerful embedding model (`BAAI/bge-large-zh-v1.5`).
3. **Retrieval:** Searches the FAISS index to find the top relevant comments.
4. **Prompt Construction:** Builds a prompt with the retrieved comments and the user question.
5. **Answer Generation:** Uses a large LLM (`Qwen/Qwen1.5-7B-Chat`) to generate a concise answer.
6. **Display:** Shows the answer in the Streamlit web app.

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies.

## Usage

1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
2. Prepare your data and FAISS index in the `vector_index` directory.
3. Run the Streamlit app:
    ```
    streamlit run app.py
    ```
4. Enter your question in the web interface to get an answer.

## File Structure

- `app.py`: Main Streamlit web application.
- `vector_index/faiss_docs.csv`: CSV file with cleaned comments.
- `vector_index/faiss_index.index`: FAISS vector index for retrieval.
- `requirements.txt`: Python dependencies.

## Models Used

- **Embedding:** `BAAI/bge-large-zh-v1.5`
- **LLM:** `Qwen/Qwen1.5-7B-Chat`

## License

This project is for research and educational purposes.