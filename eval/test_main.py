from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from main import generalAgent

retriever = generalAgent

def test_answer_relevancy():
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    test_case = LLMTestCase(
        input="Siapa rektor undiksha?",
        actual_output="Prof. Dr. I Wayan Lasmawan, M.Pd",
        retrieval_context=["{retriever}"]
    )
    assert_test(test_case, [answer_relevancy_metric])