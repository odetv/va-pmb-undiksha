import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.llm import embedder


def cosine_similarity_manual(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norma_vec1 = np.linalg.norm(vec1)
    norma_vec2 = np.linalg.norm(vec2)
    return dot_product / (norma_vec1 * norma_vec2)


def get_cos_similarity_rr():
    # Load data dari test case excel
    eo = pd.read_excel('test/human/calc/calc_testcase.xlsx', sheet_name='RR', usecols='B')['Question'].tolist()
    eg1 = pd.read_excel('test/human/calc/calc_testcase.xlsx', sheet_name='RR', usecols='F')['P1'].tolist()
    eg2 = pd.read_excel('test/human/calc/calc_testcase.xlsx', sheet_name='RR', usecols='G')['P2'].tolist()
    eg3 = pd.read_excel('test/human/calc/calc_testcase.xlsx', sheet_name='RR', usecols='H')['P3'].tolist()

    # Simpan hasil cosine similarity
    results = []

    # Inisialisasi embedding model
    _, EMBEDDER = embedder()
    embeddings = EMBEDDER

    # Loop melalui semua pertanyaan asli dan pertanyaan artifisial
    for i in range(len(eo)):
        try:
            # Buat embeddings
            question_embedding = embeddings.embed_query(eo[i])
            p1_embedding = embeddings.embed_query(eg1[i])
            p2_embedding = embeddings.embed_query(eg2[i])
            p3_embedding = embeddings.embed_query(eg3[i])

            # Hitung cosine similarity
            sim1 = cosine_similarity_manual(p1_embedding, question_embedding)
            sim2 = cosine_similarity_manual(p2_embedding, question_embedding)
            sim3 = cosine_similarity_manual(p3_embedding, question_embedding)

            results.append([i + 1, sim1, sim2, sim3])
        except Exception as e:
            print(f"Error processing row {i + 1}: {e}")
            results.append([i + 1, None, None, None])

    # Buat DataFrame dari hasil
    df_results = pd.DataFrame(results, columns=['No Test Case', 'Eg1', 'Eg2', 'Eg3'])

    # Simpan ke file excel
    df_results.to_excel('test/human/result/cos_similarity_rr.xlsx', index=False)
    print("Cosine similarity saved to 'test/human/result/cos_similarity_rr.xlsx'.")


def main():
    get_cos_similarity_rr()

if __name__ == "__main__":
    main()