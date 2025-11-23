"""Skill grouping helper using sentence-transformer embeddings.

Groups semantically similar skills while preserving original order. This keeps
question prompts manageable without losing niche skills.
"""
from __future__ import annotations

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def _embed_skills(skills: List[str]) -> np.ndarray:
    if not skills:
        return np.empty((0, 0))
    model = _get_model()
    embeddings = model.encode(skills, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32)


def cluster_skills(
    skills: List[str],
    *,
    max_group_size: int = 4,
    min_similarity: float = 0.6,
) -> List[List[str]]:
    """Group similar skills using greedy cosine similarity clustering.

    Args:
        skills: Ordered list of skill strings.
        max_group_size: Maximum number of items per group.
        min_similarity: Minimum cosine similarity to join an existing group.

    Returns:
        List of skill groups (each group preserves input order).
    """
    cleaned = [s.strip() for s in skills if s and s.strip()]
    if not cleaned:
        return []
    if len(cleaned) == 1:
        return [cleaned]

    embeddings = _embed_skills(cleaned)
    groups: List[List[str]] = []
    centroids: List[np.ndarray] = []

    for idx, skill in enumerate(cleaned):
        emb = embeddings[idx]
        best_idx = -1
        best_similarity = -1.0
        for g_idx, centroid in enumerate(centroids):
            if len(groups[g_idx]) >= max_group_size:
                continue
            sim = float(np.dot(centroid, emb))
            if sim > best_similarity:
                best_similarity = sim
                best_idx = g_idx
        if best_idx >= 0 and best_similarity >= min_similarity:
            groups[best_idx].append(skill)
            # Update centroid (normalized mean)
            old_size = len(groups[best_idx]) - 1
            new_centroid = (centroids[best_idx] * old_size + emb) / (old_size + 1)
            norm = np.linalg.norm(new_centroid)
            centroids[best_idx] = new_centroid / max(norm, 1e-8)
        else:
            groups.append([skill])
            centroids.append(emb)

    return groups


__all__ = ["cluster_skills"]
