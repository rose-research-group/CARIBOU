"""
RAG (Retrieval Augmented Generation) client initialization and management.

This module handles:
- Lazy initialization of the RAG singleton
- RAG client access for agents
"""
from __future__ import annotations

from rich.console import Console

from caribou.rag.RetrievalAugmentedGeneration import RetrievalAugmentedGeneration


# --- Lazily initialize RAG ---
_RAG_SINGLETON = None


def get_rag_client(console: Console) -> RetrievalAugmentedGeneration:
    """Get or initialize the RAG client singleton."""
    global _RAG_SINGLETON
    if _RAG_SINGLETON is None:
        console.print("[cyan]Initializing RAG model (this may take a moment)...[/cyan]")
        _RAG_SINGLETON = RetrievalAugmentedGeneration()
    return _RAG_SINGLETON
