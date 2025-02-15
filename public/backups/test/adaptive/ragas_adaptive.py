import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from datetime import datetime
from configs.sample_case import questions, ground_truths
from configs.rag_adaptive import rag_adaptive
from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

answers = []
contexts = []

for question in questions:
    context, answer = rag_adaptive(question)
    contexts.append([ctx['answer'] for ctx in context])
    answers.append(answer)

data = {
    "question": questions,
    "contexts": contexts,
    "answer": answers,
    "ground_truth": ground_truths
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
df.columns = ["question", "contexts", "answer", "ground_truth", "context_precision", "context_recall", "faithfulness", "answer_relevancy"]
df['average'] = df[['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']].mean(axis=1)
empty_row = pd.Series([None] * len(df.columns), index=df.columns)
df = pd.concat([df, pd.DataFrame([empty_row])], ignore_index=True)
average_row = df[['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']].mean().to_frame().T
average_row['question'] = 'Average'
average_row['contexts'] = ''
average_row['answer'] = ''
average_row['ground_truth'] = ''
average_row['average'] = average_row[['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']].mean(axis=1)
df = pd.concat([df, average_row], ignore_index=True)

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
file_name = f"test/adaptive/results/score_ragas_adaptive_{timestamp}.xlsx"

with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
    df.to_excel(writer, index=False, sheet_name='Evaluation')
    workbook  = writer.book
    worksheet = writer.sheets['Evaluation']
    last_row = len(df)
    worksheet.merge_range(f'A{last_row + 1}:D{last_row + 1}', 'Average', workbook.add_format({'align': 'auto'}))