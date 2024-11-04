from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)


questions = [
    "Siapa rektor undiksha?", 
    "Siapa rektor undiksha?", 
    "Berapa ada fakultas?",
]

ground_truths = [
    "Prof. Dr. I Wayan Lasmawan, M.Pd.", 
    "Prof. Dr. I Wayan Lasmawan, M.Pd.",  
    "Universitas Pendidikan Ganesha memiliki 9 fakultas."
]

answers = [
    "Prof. Dr. I Nyoman Jampel, M.Pd.", 
    "Prof. Dr. I Wayan Lasmawan, M.Pd.",
    "Universitas Pendidikan Ganesha memiliki 9 fakultas."
]

contexts = [
    [
        "Salam Harmoni🙏 Rektor Universitas Pendidikan Ganesha (Undiksha) adalah Prof. Dr. I Wayan Lasmawan, M.Pd."
    ], 
    [
        "Salam Harmoni🙏 Rektor Universitas Pendidikan Ganesha (Undiksha) adalah Prof. Dr. I Wayan Lasmawan, M.Pd."
    ], 
    [
        "Salam Harmoni🙏\n\nUniversitas Pendidikan Ganesha memiliki 9 fakultas, yaitu:\n\n1. Fakultas Teknik dan Kejuruan (FTK)\n2. Fakultas Olahraga dan Kesehatan (FOK)\n3. Fakultas Matematika dan Ilmu Pengetahuan Alam (FMIPA)\n4. Fakultas Ilmu Pendidikan (FIP)\n5. Fakultas Hukum dan Ilmu Sosial (FHIS)\n6. Fakultas Ekonomi (FE)\n7. Fakultas Bahasa dan Seni (FBS)\n8. Fakultas Kedokteran (FK)\n9. Fakultas Pascasarjana\n\nHarap diperhatikan jawaban ini dihasilkan oleh AI, mungkin saja jawaban yang dihasilkan tidak sesuai."
    ]
]

data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truths,
    "reference": ground_truths
}

dataset = Dataset.from_dict(data)

result = evaluate(
    dataset = dataset, 
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

df = result.to_pandas()
df.columns = ["question", "answer", "contexts", "ground_truths", "context_precision", "context_recall", "faithfulness", "answer_relevancy"]
df.to_excel("eval/output_basic.xlsx", index=True)