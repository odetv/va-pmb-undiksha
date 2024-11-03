import sys
import os
from typing import List, Dict, Union
from dataclasses import dataclass
from datasets import Dataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import build_graph
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    context_precision, 
    context_recall,
    faithfulness,
    answer_relevancy
)


@dataclass
class QueryResult:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str

class RagasEvaluator:
    def __init__(self):
        self.results: List[QueryResult] = []
        
    def collect_response(self, question: str, ground_truth: str) -> None:
        """Menjalankan query dan mengumpulkan informasi untuk evaluasi"""
        try:
            response = build_graph(question)
            if isinstance(response, dict):
                answer = response.get("answer", "")
                contexts = response.get("contexts", [])
                if isinstance(contexts, str):
                    contexts = [contexts]
                elif isinstance(contexts, list):
                    contexts = [str(ctx) if not isinstance(ctx, str) else ctx for ctx in contexts]
                else:
                    contexts = []
            else:
                answer = str(response)
                contexts = [answer]
            
            self.results.append(QueryResult(
                question=question,
                answer=answer,
                contexts=contexts,
                ground_truth=ground_truth
            ))
            
        except Exception as e:
            print(f"Error saat mengumpulkan respons untuk pertanyaan '{question}': {str(e)}")
    
    def prepare_dataset(self) -> Dataset:
        """Mengkonversi hasil menjadi dataset yang kompatibel dengan Ragas"""
        data = {
            'question': [],
            'answer': [],
            'contexts': [],
            'ground_truth': []
        }
        
        for result in self.results:
            data['question'].append(result.question)
            data['answer'].append(result.answer)
            contexts = result.contexts if result.contexts else [""]
            data['contexts'].append(contexts)
            data['ground_truth'].append(result.ground_truth)
        
        dataset = Dataset.from_dict(data)
        
        print("\nFormat Dataset:")
        print(f"Number of examples: {len(dataset)}")
        print(f"Sample data point:")
        print(dataset[0] if len(dataset) > 0 else "No data")
        
        return dataset
    
    def evaluate(self, output_file: str = "eval/output_test_rag.xlsx") -> None:
        """Menjalankan evaluasi Ragas dan menyimpan hasil"""
        try:
            dataset = self.prepare_dataset()
            
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness
            ]
            
            score = evaluate(
                dataset=dataset,
                metrics=metrics
            )
            
            df = score.to_pandas()
            df.to_excel(output_file)
            print(f"\nHasil evaluasi disimpan ke {output_file}")
            print("\nMetrics:")
            print(df)
            return df
            
        except Exception as e:
            print(f"Error detail saat melakukan evaluasi:")
            import traceback
            print(traceback.format_exc())
            raise

def main():
    evaluator = RagasEvaluator()
    
    test_cases = [
        {
            "question": "Siapa rektor undiksha?",
            "ground_truth": """Salam Harmoni🙏
Rektor Universitas Pendidikan Ganesha (Undiksha) adalah Prof. Dr. I Wayan Lasmawan, M.Pd. 
Harap diperhatikan jawaban ini dihasilkan oleh AI, mungkin saja jawaban yang dihasilkan tidak sesuai."
"""
        },
        {
            "question": "Berapa jumlah fakultas di Undiksha?",
            "ground_truth": """"Salam Harmoni🙏
Universitas Pendidikan Ganesha memiliki 9 fakultas, yaitu:
1. Fakultas Teknik dan Kejuruan (FTK)
2. Fakultas Olahraga dan Kesehatan (FOK)
3. Fakultas Matematika dan Ilmu Pengetahuan Alam (FMIPA)
4. Fakultas Ilmu Pendidikan (FIP)
5. Fakultas Hukum dan Ilmu Sosial (FHIS)
6. Fakultas Ekonomi (FE)
7. Fakultas Bahasa dan Seni (FBS)
8. Fakultas Kedokteran (FK)
9. Fakultas Pascasarjana
Harap diperhatikan jawaban ini dihasilkan oleh AI, mungkin saja jawaban yang dihasilkan tidak sesuai."
"""
        }
    ]
    
    for test_case in test_cases:
        evaluator.collect_response(
            question=test_case["question"],
            ground_truth=test_case["ground_truth"]
        )
    
    try:
        results = evaluator.evaluate()
    except Exception as e:
        print(f"Terjadi error saat evaluasi: {str(e)}")

if __name__ == "__main__":
    main()