import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from main import rag_adaptive

def get_context_and_answer():
    questions = pd.read_excel('test/human/data/data_testcase.xlsx', sheet_name='QA', usecols='B')['Question'].tolist()
    ground_truths = pd.read_excel('test/human/data/data_testcase.xlsx', sheet_name='QA', usecols='C')['Ground Truth'].tolist()
    results = []

    for index, (question, ground_truth) in enumerate(zip(questions, ground_truths), start=1):
        contexts, answers = rag_adaptive(question)
        # context = [context['answer'] for context in contexts if 'answer' in context]

        results.append({
            'Question': question,
            'Ground Truth': ground_truth,
            'Context': contexts,
            'Answer': answers
        })

        print(f"Proses Test Case [{index}/{len(questions)}]")
    
    df_results = pd.DataFrame(results)
    
    df_results.to_excel('test/human/result/result_testcase.xlsx', index=False, sheet_name='DATA')


def main():
    get_context_and_answer()

if __name__ == "__main__":
    main()