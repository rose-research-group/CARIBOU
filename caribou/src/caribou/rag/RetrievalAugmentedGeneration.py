import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import redirect_stdout, redirect_stderr
import io # Import io for suppressing output

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ── Dependencies ─────────────────────────────────────────────
try:
    import re
    from sentence_transformers import SentenceTransformer
    from rich.console import Console
    import matplotlib.pyplot as plt
    import numpy as np
    # Import specific exceptions that might occur during download
    import requests.exceptions
    import urllib3.exceptions

except ImportError as e:
    print(f"Missing dependency: {e}", file=sys.stderr)
    sys.exit(1)

# ── Paths and Constants ─────────────────────────────────────────────
console = Console()

RAG_DIR = Path(__file__).resolve().parent.parent / "rag"
EMBEDDING_FILE = RAG_DIR / "embeddings.jsonl"
FUNCTIONS_FILE = RAG_DIR / "functions.jsonl"
MIN_SIMILARITY = 0.7

class RetrievalAugmentedGeneration():

    def __init__(self):
        self.embeddings = self.load_embeddings()
        self.functions = self.load_functions()
        self.model = None # Initialize model as None
        self.model_loaded = False # Flag to track model loading status
        self.queries = []
        self._load_model() # Attempt to load the model during initialization

    def _load_model(self):
        """Attempts to load the SentenceTransformer model."""
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        try:
            # Suppress potential stdout/stderr during model download/loading
            # (e.g., download progress bars, warnings)
            # Use io.StringIO to capture and discard output
            stdout_trap = io.StringIO()
            stderr_trap = io.StringIO()
            with redirect_stdout(stdout_trap), redirect_stderr(stderr_trap):
                self.model = SentenceTransformer(model_name)
            self.model_loaded = True
            console.log(f"[green]Successfully loaded SentenceTransformer model: {model_name}")
        # Catch specific network-related exceptions
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                urllib3.exceptions.MaxRetryError,
                urllib3.exceptions.NewConnectionError,
                OSError, # Can sometimes manifest as OSError (e.g., Name or service not known)
                ValueError # Can occur if model identifier is invalid or files missing
               ) as e:
            console.log(f"[yellow]⚠️ Failed to load SentenceTransformer model '{model_name}'. "
                        f"Network/Connection issue: {type(e).__name__}. RAG queries will be disabled.")
            console.log(f"[yellow]Error details: {e}")
            self.model = None
            self.model_loaded = False
        except Exception as e: # Catch any other unexpected errors during loading
             console.log(f"[red]❌ An unexpected error occurred loading SentenceTransformer model '{model_name}': {e}")
             self.model = None
             self.model_loaded = False


    def load_embeddings(self) -> List[np.ndarray]:
        if not EMBEDDING_FILE.exists():
            console.log(f"[yellow]Embeddings file not found at {EMBEDDING_FILE}. No embeddings loaded.")
            return []
        try:
            with open(EMBEDDING_FILE, "r", encoding="utf-8") as f:
                embeddings = [np.array(json.loads(line)) for line in f if line.strip()]
                console.log(f"[green]Loaded {len(embeddings)} embeddings from {EMBEDDING_FILE}")
                return embeddings
        except json.JSONDecodeError:
            console.log(f"[red]Embeddings file {EMBEDDING_FILE} is not valid JSONL.")
            return []
        except Exception as e:
            console.log(f"[red]Error loading embeddings from {EMBEDDING_FILE}: {e}")
            return []

    def load_functions(self) -> List[Dict[str, str]]:
        if not FUNCTIONS_FILE.exists():
            console.log(f"[yellow]Functions file not found at {FUNCTIONS_FILE}. No functions loaded.")
            return []
        try:
            with open(FUNCTIONS_FILE, "r", encoding="utf-8") as f:
                functions = [json.loads(line) for line in f if line.strip()]
                console.log(f"[green]Loaded {len(functions)} functions from {FUNCTIONS_FILE}")
                return functions
        except json.JSONDecodeError:
            console.log(f"[red]Functions file {FUNCTIONS_FILE} is not valid JSONL.")
            return []
        except Exception as e:
            console.log(f"[red]Error loading functions from {FUNCTIONS_FILE}: {e}")
            return []

    @staticmethod
    def cosine_similarity(A: np.ndarray, B: List[np.ndarray]) -> List[float]:
        # Ensure B is not empty and contains valid arrays
        if not B or not all(isinstance(emb, np.ndarray) and emb.size > 0 for emb in B):
            return []
        # Ensure A is valid
        if not isinstance(A, np.ndarray) or A.size == 0:
            return [0.0] * len(B)

        sims = []
        norm_A = np.linalg.norm(A)
        if norm_A == 0: # Avoid division by zero if A is a zero vector
             return [0.0] * len(B)

        for emb in B:
            norm_emb = np.linalg.norm(emb)
            if norm_emb == 0: # Avoid division by zero if emb is a zero vector
                sims.append(0.0)
            else:
                sims.append(np.dot(A, emb) / (norm_A * norm_emb))
        return sims

    def retrieve_function(self, name:str) -> Optional[str]:
        # Check if functions were loaded
        if not hasattr(self, 'functions') or not self.functions:
             console.log("[yellow]No functions loaded to retrieve from.")
             return None
        for function in self.functions:
            if "signature" in function and name in function["signature"]:
                return function["signature"]
        return None

    def query(self, text_query: str) -> Optional[str]: # Return type changed for clarity
        """
        Encodes a text query and finds the most similar function signature.

        Args:
            text_query: The natural language query string.

        Returns:
            The signature of the most similar function if found above the threshold,
            otherwise None. Returns None immediately if the model failed to load.
        """
        self.queries.append(text_query)

        # === Mocking Check ===
        if not self.model_loaded or self.model is None:
            # console.log("[yellow]Model not loaded. RAG query is mocked, returning None.")
            return None # Return None as requested for failed download

        # === Standard Query Logic ===
        if not self.embeddings:
            # console.log("[yellow]No embeddings loaded to compare against.")
            return None
        if not self.functions:
            # console.log("[yellow]No functions loaded to return.")
            return None

        try:
            query_embedding = self.model.encode([text_query])[0]
            sims = self.cosine_similarity(query_embedding, self.embeddings)

            if not sims: # Handle case where similarity calculation failed or returned empty
                # console.log("[yellow]Similarity calculation yielded no results.")
                return None

            idx = np.argmax(sims)

            # Ensure index is valid for the functions list
            if idx >= len(self.functions):
                 console.log(f"[red]Calculated index {idx} is out of bounds for functions list (length {len(self.functions)}).")
                 return None

            if sims[idx] >= MIN_SIMILARITY:
                # console.log(f"Query '{text_query[:50]}...' matched function {idx} with similarity {sims[idx]:.4f}")
                return self.functions[idx].get("signature") # Use .get for safety
            else:
                # console.log(f"Query '{text_query[:50]}...' - No function found above similarity threshold ({sims[idx]:.4f} < {MIN_SIMILARITY})")
                return None
        except Exception as e:
            console.log(f"[red]Error during RAG query processing: {e}")
            return None