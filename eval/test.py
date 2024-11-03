from datasets import Dataset 
from ragas import evaluate
from ragas.metrics import answer_correctness, context_precision, context_recall, faithfulness, answer_relevancy

data_samples = {
    'question': [
        'When was the first super bowl?', 
        'Who won the most super bowls?'
    ],
    'answer': [
        'The first superbowl was held on Jan 15, 1967', 
        'The most super bowls have been won by The New England Patriots'
    ],
    'contexts': [
        [
            'The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'
        ], 
        [
            'The Green Bay Packers...Green Bay, Wisconsin.',
            'The Packers compete...Football Conference'
        ]
    ],
    'ground_truth': [
        'The first superbowl was held on January 15, 1967', 
        'The New England Patriots have won the Super Bowl a record six times'
    ]
}

dataset = Dataset.from_dict(data_samples)

score = evaluate(dataset, metrics=[context_precision, context_recall, faithfulness, answer_relevancy, answer_correctness])
df = score.to_pandas()
df.to_excel("output_test_rag.xlsx")