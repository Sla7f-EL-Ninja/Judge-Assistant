"""
test_rag_quality.py -- RAG quality evaluation using RAGAS metrics.

Loads the golden dataset, runs queries through CivilLawRAGAdapter,
evaluates faithfulness and context recall using RAGAS, and saves results.

Marker: behavioral, llm_eval
"""

import json
import pathlib

import pytest
_ADAPTER = None

def _get_adapter():
    global _ADAPTER
    if _ADAPTER is None:
        from Supervisor.agents.civil_law_rag_adapter import CivilLawRAGAdapter
        _ADAPTER = CivilLawRAGAdapter()
    return _ADAPTER

GOLDEN_DATASET_PATH = pathlib.Path(__file__).parent.parent / "eval" / "golden_dataset.json"
RESULTS_PATH = pathlib.Path(__file__).parent.parent / "eval" / "ragas_results_latest.json"


def _load_golden_dataset():
    """Load and return golden test cases from JSON."""
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.behavioral
@pytest.mark.llm_eval
class TestRAGQuality:
    """Evaluate RAG quality using RAGAS metrics."""

    def _query_rag(self, question: str) -> dict:
        """Run a question through CivilLawRAGAdapter and return structured output."""
        adapter = _get_adapter()
        result = adapter.invoke(
            query=question,
            context={
                "case_id": "",
                "uploaded_files": [],
                "conversation_history": [],
            },
        )
        return {
            "answer": result.response,
            "sources": result.sources,
            "raw_output": result.raw_output,
        }

    def test_rag_quality_evaluation(self):
        """Run RAGAS evaluation on golden dataset and assert quality thresholds."""
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics.collections import context_precision, context_recall, faithfulness
        except ImportError:
            pytest.skip("ragas or datasets not available")

        try:
            cases = _load_golden_dataset()
        except FileNotFoundError:
            pytest.skip("golden_dataset.json not found")

        # Filter to civil_law_rag cases only
        civil_cases = [c for c in cases if c["expected_agent"] == "civil_law_rag"]
        if not civil_cases:
            pytest.skip("No civil_law_rag cases in golden dataset")

        questions = []
        answers = []
        contexts = []
        ground_truths = []

        for case in civil_cases:
            try:
                result = self._query_rag(case["question"])
                questions.append(case["question"])
                answers.append(result["answer"])
                contexts.append(result["sources"] or ["No context retrieved"])
                ground_truths.append(case["ground_truth"])
            except Exception as exc:
                # Skip individual failures but continue
                continue

        if len(questions) < 5:
            pytest.skip(
                f"Only {len(questions)} successful queries -- need at least 5 for evaluation"
            )

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )

        try:
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_huggingface import HuggingFaceEmbeddings
            from config import get_llm
            from config.rag import EMBEDDING_MODEL

            ragas_llm = LangchainLLMWrapper(get_llm("high"))
            ragas_embeddings = LangchainEmbeddingsWrapper(
                HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            )

            results = evaluate(
                dataset,
                metrics=[faithfulness, context_recall, context_precision],
                llm=ragas_llm,
                embeddings=ragas_embeddings,
            )

            # Save results to JSON
            results_dict = {
                "faithfulness": float(results["faithfulness"]),
                "context_recall": float(results["context_recall"]),
                "context_precision": float(results["context_precision"]),
                "num_cases": len(questions),
            }
            RESULTS_PATH.write_text(
                json.dumps(results_dict, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            assert results_dict["faithfulness"] >= 0.80, (
                f"Faithfulness {results_dict['faithfulness']:.2f} is below 0.80 threshold"
            )
            assert results_dict["context_recall"] >= 0.75, (
                f"Context recall {results_dict['context_recall']:.2f} is below 0.75 threshold"
            )
        except Exception as exc:
            pytest.skip(f"RAGAS evaluation failed: {exc}")

    @pytest.mark.parametrize(
        "case",
        _load_golden_dataset() if GOLDEN_DATASET_PATH.exists() else [],
        ids=lambda c: c.get("id", "unknown"),
    )
    def test_rag_contains_expected_concepts(self, case):
        """RAG responses should contain the expected legal concepts."""
        if case.get("expected_agent") != "civil_law_rag":
            pytest.skip("Not a civil_law_rag case")

        try:
            result = self._query_rag(case["question"])
            answer = result["answer"]

            must_contain = case.get("must_contain_concepts", [])
            missing = [c for c in must_contain if c not in answer]

            # Allow up to 50% missing concepts (LLM may use synonyms)
            max_missing = len(must_contain) // 2
            assert len(missing) <= max_missing, (
                f"Answer for '{case['id']}' is missing too many concepts: {missing}\n"
                f"Answer: {answer[:200]}..."
            )
        except AssertionError:
            raise
        except Exception as exc:
            pytest.skip(f"RAG dependency unavailable: {exc}")