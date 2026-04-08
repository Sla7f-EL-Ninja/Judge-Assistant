"""
test_answer_consistency.py -- Verify answer consistency across repeated runs.

Runs representative queries multiple times and checks that the answers
are semantically similar using embedding cosine similarity.

Marker: behavioral, llm_eval
"""

import pytest
_ADAPTER = None

def _get_adapter():
    global _ADAPTER
    if _ADAPTER is None:
        from Supervisor.agents.civil_law_rag_adapter import CivilLawRAGAdapter
        _ADAPTER = CivilLawRAGAdapter()
    return _ADAPTER

@pytest.mark.behavioral
@pytest.mark.llm_eval
class TestAnswerConsistency:
    """Verify that repeated queries produce semantically consistent answers."""

    REPRESENTATIVE_QUERIES = [
        "ما هي شروط صحة العقد في القانون المدني المصري؟",
        "ما أحكام المسؤولية التقصيرية؟",
        "ما هي أحكام الفسخ في العقود الملزمة للجانبين؟",
        "ما أحكام التقادم المسقط في القانون المدني؟",
        "ما هي أحكام حوالة الحق؟",
    ]
    NUM_RUNS = 3
    MIN_SIMILARITY = 0.80

    def _get_answer(self, query: str) -> str:
        """Run a query through CivilLawRAGAdapter and return the response."""
        adapter = _get_adapter()
        result = adapter.invoke(
            query=query,
            context={
                "case_id": "",
                "uploaded_files": [],
                "conversation_history": [],
            },
        )
        return result.response

    def _compute_cosine_similarity(self, texts: list) -> float:
        """Compute average pairwise cosine similarity for a list of texts."""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            model = SentenceTransformer("BAAI/bge-m3")
            embeddings = model.encode(texts, normalize_embeddings=True)

            # Compute pairwise cosine similarities
            similarities = []
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = float(np.dot(embeddings[i], embeddings[j]))
                    similarities.append(sim)

            return sum(similarities) / len(similarities) if similarities else 0.0
        except ImportError:
            pytest.skip("sentence-transformers not available")

    @pytest.mark.parametrize("query", REPRESENTATIVE_QUERIES)
    def test_answer_consistency_per_query(self, query):
        """Each query run N times should produce similar answers."""
        try:
            answers = []
            for _ in range(self.NUM_RUNS):
                answer = self._get_answer(query)
                assert answer and len(answer) > 10, (
                    f"Answer too short for query: {query}"
                )
                answers.append(answer)

            similarity = self._compute_cosine_similarity(answers)
            assert similarity >= self.MIN_SIMILARITY, (
                f"Answer consistency for query '{query[:50]}...' is {similarity:.3f}, "
                f"below threshold {self.MIN_SIMILARITY}"
            )
        except Exception as exc:
            if "CivilLawRAGAdapter" in str(exc) or "sentence" in str(exc):
                pytest.skip(f"Dependency unavailable: {exc}")
            raise
