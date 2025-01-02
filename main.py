from flask import Flask, request, jsonify
from flask import Flask, render_template, request, url_for
from flask_cors import CORS
import pandas as pd
import tiktoken
import openai
import numpy as np
import os
import json

openai.api_key = os.getenv('OPENAI_API_KEY')
app = Flask(__name__)
CORS(app)
# Load the CSV file
csv_path = "embs_docs/combined_embedded_documents.csv"  # Ganti dengan path file CSV Anda
df = pd.read_csv(csv_path)

EMBEDDING_MODEL = "text-embedding-3-large"
COMPLETIONS_MODEL = "gpt-4o"

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    result = openai.embeddings.create(
        model=model,
        input=text
    )
    return result.data[0].embedding

df['Embedding'] = df['Embedding'].apply(lambda x: np.array(eval(x)))

def vector_similarity(x: np.array, y: np.array) -> float:
    """
    Mengembalikan kesamaan antara dua vektor numpy array.
    """
    return np.dot(x, y)

def order_by_similarity(query: str, df) -> list[tuple[float, int]]:
    """
    Mencari embedding query untuk query yang diberikan, dan membandingkannya dengan semua embedding dokumen
    untuk menemukan bagian yang paling relevan. 
    
    Mengembalikan daftar bagian dokumen, diurutkan berdasarkan relevansi secara menurun.
    """
    query_embedding = get_embedding(query)
    
    # Jika embedding sudah diubah menjadi numpy array di awal, langsung pakai di sini
    similarities = [
        (vector_similarity(query_embedding, embedding), idx)
        for idx, embedding in enumerate(df['Embedding'])
    ]
    
    # Mengurutkan berdasarkan kesamaan vektor secara menurun
    similarities = sorted(similarities, reverse=True, key=lambda x: x[0])
    
    return similarities

# Menentukan parameter dan konstanta
MAX_SECTION_LEN = 2000
SEPARATOR = "\n* "
ENCODING = "gpt2"

encoding = tiktoken.get_encoding(ENCODING)
separator_len = len(encoding.encode(SEPARATOR))

def construct_prompt(question: str, df: pd.DataFrame, top_n: int = 3, max_section_len: int = 2000) -> str:
    """
    Menyusun prompt dengan mengambil beberapa dokumen yang paling relevan, 
    mempertimbangkan batasan jumlah token.
    """
    most_relevant_sections = order_by_similarity(question, df)
    chosen_sections = []
    chosen_sections_len = 0
    chosen_sections_indexes = []
    
    separator = "\n* "
    encoding = tiktoken.get_encoding("gpt2")
    separator_len = len(encoding.encode(separator))
    
    for _, idx in most_relevant_sections:
        section = df.loc[idx]
        content = section['Berita']
        tokens = section['Tokens']
        
        chosen_sections_len += tokens + separator_len
        if chosen_sections_len > max_section_len:
            break
        
        chosen_sections.append(separator + content.replace("\n", " "))
        chosen_sections_indexes.append(str(idx))
    
    prompt = (
        f"You are a chatbot that provides answers based on the following context:"
        f"{''.join(chosen_sections)}. "
        f"If the answer cannot be found in the articles, write "
        f"'Maaf, saya tidak memiliki informasi mengenai pertanyaan yang anda berikan. "
        f"Apakah ada hal lain yang dapat saya bantu?'"
    )
    prompt += f"\n\nQuestion: {question}"
    return prompt

def answer_with_gpt_4(prompt: str) -> str:
    """
    Mengirim prompt ke model GPT-4 untuk mendapatkan jawaban.
    """
    response = openai.chat.completions.create(
        model=COMPLETIONS_MODEL,
        messages=[
            {"role": "system", "content": "You are an informational chatbot."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    
    # Extract the question from the incoming JSON payload
    question = data.get("question", "")
    
    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    prompt = construct_prompt(question, df)
    response = answer_with_gpt_4(prompt)
    
    return jsonify({"response": response}), 200

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)