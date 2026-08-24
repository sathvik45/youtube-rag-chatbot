"""Retrieval pipeline: fetch, chunk, index, retrieve, cite.

Deliberately free of FastAPI and LangGraph imports -- graph nodes, the eval
runner and background workers all call into here, so a dependency in the other
direction would make the eval harness drag in the web framework.
"""
