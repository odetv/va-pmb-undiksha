import pandas as pd
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

ground_truth = [
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
    "ground_truth": ground_truth
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
df.columns = ["question", "answer", "contexts", "ground_truth", "context_precision", "context_recall", "faithfulness", "answer_relevancy"]
df['average'] = df[['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']].mean(axis=1)
empty_row = pd.Series([None] * len(df.columns), index=df.columns)
df = pd.concat([df, pd.DataFrame([empty_row])], ignore_index=True)
average_row = df[['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']].mean().to_frame().T
average_row['question'] = 'Average'
average_row['answer'] = ''
average_row['contexts'] = ''
average_row['ground_truth'] = ''
average_row['average'] = average_row[['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']].mean(axis=1)
df = pd.concat([df, average_row], ignore_index=True)
with pd.ExcelWriter("eval/score_basic.xlsx", engine='xlsxwriter') as writer:
    df.to_excel(writer, index=False, sheet_name='Evaluation')
    workbook  = writer.book
    worksheet = writer.sheets['Evaluation']
    last_row = len(df)
    worksheet.merge_range(f'A{last_row + 1}:D{last_row + 1}', 'Average', workbook.add_format({'align': 'center'}))