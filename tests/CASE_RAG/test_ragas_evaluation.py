# """
# tests/CASE_RAG/test_ragas_evaluation.py

# Layer: Semantic Evaluation (LLM-as-a-Judge paired with Custom Reporting)
# Measures the linguistic quality, hallucination rate, and relevance of the RAG pipeline
# while feeding rich payloads into case_doc_rag_test_report.json.

# Run using:
# pytest -m ragas -s
# """

# from __future__ import annotations

# import pytest
# from datasets import Dataset
# from ragas import evaluate
# from ragas.metrics import answer_relevancy, faithfulness
# from config import cfg, get_llm
# from conftest import TEST_CASE_ID, invoke_graph
# LLM = get_llm("high")

# # ---------------------------------------------------------------------------
# # Golden Evaluation Dataset (Mapped to unique Test IDs matching happy path)
# # ---------------------------------------------------------------------------
# EVALUATION_DATASET = [
#      {
#         "test_id": "RAGAS_B6",
#         "question": "ما هي الأسانيد القانونية التي استندت إليها مذكرة دفاع المالك لتبرير صحة قرار لجنة التظلمات بالإزالة؟",
#         "ground_truth": "استندت مذكرة الدفاع إلى المادة 119 من قانون البناء الموحد لسنة 2008، وتحديداً المواد 90 و 92."
#     },
#     {
#         "test_id": "RAGAS_B7",
#         "question": "ما هو الدفع الرئيسي الذي أثاره المدعي (المستأجر) في صحيفة الدعوى لإثبات بطلان القرار فيما يخص صفة مقدم التظلم؟",
#         "ground_truth": "دفع المدعي بانعدام صفة مقدم التظلم لأنه ليس المالك الحقيقي للعقار المذكور في صحيفة الدعوى."
#     },

# ]
# from RAG.case_doc_rag.infrastructure import get_embedding_function

# # Initialize the embedding function using your project's settings
# embeddings = get_embedding_function()
# # ---------------------------------------------------------------------------
# # Ragas Evaluation Test with Reporting Registration
# # ---------------------------------------------------------------------------
# @pytest.mark.ragas
# @pytest.mark.timeout(180)  # Ragas LLM steps take slightly longer per item
# @pytest.mark.parametrize("item", EVALUATION_DATASET, ids=lambda x: x["test_id"])
# def test_ragas_semantic_quality(app, register_result, item):
#     """
#     Runs a single evaluation query through the graph, hooks the raw responses 
#     into the custom JSON tracking report, and asserts semantic metrics via Ragas.
#     """
#     query = item["question"]
#     ground_truth = item["ground_truth"]

#     # 1. Invoke the LangGraph pipeline
#     result = invoke_graph(app, query=query, case_id=TEST_CASE_ID)
    
#     # Extract pipeline variables for Ragas evaluation
#     final_answer = result.get("final_answer", "")
#     sub_answers = result.get("sub_answers", [])
#     contexts = [sa.get("answer", "") for sa in sub_answers if sa.get("found")]

#     # 2. Feed the rich data to the custom Pytest plugin report
#     # This populates the analytics dashboard and eliminates empty metric fields
#     register_result({
#         "layer": "B_ragas_evaluation",
#         "test_id": item["test_id"],
#         "query": query,
#         "answer_text": final_answer,
#         "sub_answers": sub_answers,
#         "doc_selection_mode": result.get("doc_selection_mode"),
#         "on_topic": result.get("on_topic", True),
#         "ragas_ground_truth": ground_truth,  # Injected custom metadata field
#     })

#     # 3. Construct the single-sample HuggingFace Dataset for Ragas
#     data_samples = {
#         "question": [query],
#         "answer": [final_answer],
#         "contexts": [contexts],
#         "ground_truth": [ground_truth]
#     }
#     dataset = Dataset.from_dict(data_samples)

#     # 4. Evaluate using LLM-as-a-judge
#     score = evaluate(
#         dataset,
#         metrics=[faithfulness, answer_relevancy],
#         llm=LLM,
#         embeddings= embeddings
#     )
#     score_dict = score.to_pandas().mean().to_dict()
    
#     # Print real-time scoring to stdout (visible with pytest -s)
#     print(f"\n=== RAGAS METRICS FOR {item['test_id']} ===")
#     print(f"  Faithfulness (Anti-Hallucination): {score_dict.get('faithfulness', 0.0):.2f}")
#     print(f"  Answer Relevancy (Alignment):     {score_dict.get('answer_relevancy', 0.0):.2f}")

#     # 5. Semantic quality thresholds
#     assert score_dict.get("faithfulness", 0) >= 0.85, (
#         f"Faithfulness score too low ({score_dict.get('faithfulness')}) - possible hallucination detected."
#     )
#     assert score_dict.get("answer_relevancy", 0) >= 0.80, (
#         f"Answer Relevancy score too low ({score_dict.get('answer_relevancy')}) - response drifted off-topic."
#     )