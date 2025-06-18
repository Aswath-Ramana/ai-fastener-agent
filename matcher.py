# matcher.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from rapidfuzz import fuzz, process
import os
from typing import List, Tuple

# --- Model Loading ---
# Load model once at module level for efficiency. This is a great general-purpose model.
print("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded.")

def embed_text(texts: List[str]) -> np.ndarray:
    """Encodes a list of text strings into numpy array of embeddings."""
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Builds a FAISS index for fast similarity search."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search_top_k(index: faiss.Index, query_vec: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Searches the FAISS index for the top k most similar items."""
    # FAISS expects a 2D array for queries
    D, I = index.search(query_vec.reshape(1, -1), k)
    return I[0], D[0]

def fuzzy_match(query: str, choices: List[str], limit: int = 3) -> List[Tuple[str, int, int]]:
    """Performs fuzzy string matching using token_sort_ratio."""
    # process.extract returns list of (choice, score, index)
    return process.extract(query, choices, scorer=fuzz.token_sort_ratio, limit=limit)

def log_feedback(order_part: str, order_desc: str, matched_item: str, confirmed: bool, log_path: str = "feedback_log.csv"):
    """Logs user feedback to a CSV file."""
    log_entry = {
        'timestamp': pd.Timestamp.now(),
        'Order_Part#': order_part,
        'Order_Desc': order_desc,
        'Matched_Item': matched_item,
        'Confirmed_Match': confirmed
    }
    log_df = pd.DataFrame([log_entry])
    
    # Safely append to CSV, creating header only if file doesn't exist
    log_df.to_csv(log_path, mode='a', header=not os.path.exists(log_path), index=False)