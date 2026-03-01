"""
Supervisor Agent - LangGraph-based orchestrator for the Judge Assistant.

Sits on top of the 5 worker agents (OCR, Summarization, Civil Law RAG,
Case Doc RAG, Case Reasoner), classifies judge intent, dispatches to
one or more workers, validates outputs, and maintains conversation history.
"""
