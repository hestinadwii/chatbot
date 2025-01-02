import pandas as pd
import tiktoken
import openai
import numpy as np
import os

# Set API key Anda di sini
openai.api_key = os.getenv('OPENAI_API_KEY')
# Load the CSV file
csv_path = "documents/Informasi kandidat per provinsi.csv" 
df = pd.read_csv(csv_path)


# Menginisialisasi model embedding dan completion
EMBEDDING_MODEL = "text-embedding-3-large"
COMPLETIONS_MODEL = "gpt-4o"

def get_embedding(text: str, model: str = EMBEDDING_MODEL) -> list[float]:
    result = openai.embeddings.create(
        model=model,
        input=text
    )
    return result.data[0].embedding

def compute_doc_embeddings(df: pd.DataFrame) ->pd.DataFrame:
    """
    Membuat embedding untuk setiap baris di dataframe menggunakan OpenAI Embeddings API.
    
    Mengembalikan dictionary yang memetakan antara vektor embedding dan index baris yang sesuai.
    """

    encoding = tiktoken.get_encoding("gpt2")
    embeddings = []
    token_counts = []

    for idx, row in df.iterrows():
        text = row['Berita']
        embedding = get_embedding(text)
        embeddings.append(embedding)
        token_count = len(encoding.encode(text))
        token_counts.append(token_count)
    
    df['Embedding'] = embeddings
    df['Tokens'] = token_counts
    return df

input_folder = "chatbot_docs"
output_csv_path = "embs_docs/combined_embedded_documents.csv" 

combined_df = pd.DataFrame()

for file_name in os.listdir(input_folder):
    if file_name.endswith(".csv"):
        file_path = os.path.join(input_folder, file_name)
        print(f"Memproses file: {file_path}")
        
        # Load setiap file CSV
        df = pd.read_csv(file_path)

        if 'Berita' not in df.columns:
            print(f"File {file_name} tidak memiliki kolom 'Berita'. Melewati file ini.")
            continue

        # Menghitung embeddings untuk dokumen
        df_with_embeddings = compute_doc_embeddings(df)
        
        combined_df = pd.concat([combined_df, df_with_embeddings], ignore_index=True)

combined_df.to_csv(output_csv_path, index=False)
print(f"Embedding selesai. Semua hasil disimpan ke {output_csv_path}.")