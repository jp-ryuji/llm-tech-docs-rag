from typing import Dict, List


class RAGEvaluator:
    """
    A class for evaluating Retrieval-Augmented Generation (RAG) systems.

    This evaluator tests the quality of RAG responses by comparing them
    against expected keywords in the answers.
    """

    def __init__(self, query_engine):
        """
        Initialize the RAGEvaluator with a query engine.

        Args:
            query_engine: An object with a query method that takes a question
                         and returns a response dictionary with 'answer',
                         'confidence', and 'sources' keys.
        """
        self.query_engine = query_engine

    def create_test_questions(self):
        """
        Create ground truth Q&A pairs for evaluation.

        Returns:
            List[Dict]: A list of dictionaries, each containing:
                - 'question': A test question string
                - 'expected_keywords': A list of keywords expected in a good answer
        """
        return [
            {
                "question": "How do I create a basic FastAPI application?",
                "expected_keywords": ["FastAPI", "app", "uvicorn", "@app.get"],
            },
            {
                "question": "How do I handle path parameters?",
                "expected_keywords": ["path", "parameter", "{item_id}", "int"],
            },
            # NOTE: Add more test cases if needed
        ]

    def evaluate_answers(self, test_cases: List[Dict]):
        """
        Evaluate the query engine's responses to test questions.

        For each test case, this method:
        1. Queries the engine with the test question
        2. Calculates a keyword match score by comparing the response with expected keywords
        3. Collects metadata about the response (confidence, source count)

        Args:
            test_cases (List[Dict]): A list of test case dictionaries, each with:
                - 'question': The question to ask the query engine
                - 'expected_keywords': Keywords expected in a correct answer

        Returns:
            List[Dict]: A list of evaluation results, each containing:
                - 'question': The test question
                - 'answer': The query engine's response
                - 'confidence': The engine's confidence in its answer
                - 'keyword_score': Ratio of matched keywords to expected keywords (0.0 to 1.0)
                - 'sources_count': Number of sources used to generate the response
        """
        results = []

        for case in test_cases:
            response = self.query_engine.query(case["question"])

            # Simple keyword-based evaluation
            answer_text = response["answer"].lower()
            keyword_matches = sum(
                1 for kw in case["expected_keywords"] if kw.lower() in answer_text
            )

            results.append(
                {
                    "question": case["question"],
                    "answer": response["answer"],
                    "confidence": response["confidence"],
                    "keyword_score": keyword_matches / len(case["expected_keywords"]),
                    "sources_count": len(response["sources"]),
                }
            )

        return results
