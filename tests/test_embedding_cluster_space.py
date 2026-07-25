"""The embedding cluster fit must run in the PCA-reduced space.

`get_embedding_clusters` documents itself as clustering "the PCA-reduced
embeddings", and the cluster ids it produces are only ever consumed as the
marker colour of the web UI's PCA scatter plots. Fitting the mixture on the raw
1536-d embedding array instead colours those plots from a different space, and
the full-covariance fit is degenerate at that dimensionality. These tests pin
the input that reaches the mixture.
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from shinka.database import DatabaseConfig, Program, ProgramDatabase


def _program(program_id: str, embedding, island_idx: int = 0) -> Program:
    return Program(
        id=program_id,
        code=f"def f():\n    return {program_id!r}\n",
        correct=True,
        combined_score=1.0,
        generation=0,
        island_idx=island_idx,
        embedding=embedding,
    )


class _RecordingEmbeddingClient:
    """Stands in for EmbeddingClient, recording what each stage is handed."""

    def __init__(self):
        self.cluster_inputs = []
        self.reduction_dims = []

    def get_dim_reduction(self, embeddings, method="pca", dims=2):
        self.reduction_dims.append(dims)
        X = np.asarray(embeddings, dtype=float)
        # A deterministic stand-in for PCA: keep the leading `dims` columns.
        return X[:, :dims]

    def get_embedding_clusters(self, embeddings, num_clusters=4, verbose=False):
        arr = np.asarray(embeddings, dtype=float)
        self.cluster_inputs.append(arr)
        return np.arange(len(arr)) % num_clusters


def _seeded_db(tmpdir, n_programs=8, dim=32):
    rng = np.random.default_rng(0)
    db = ProgramDatabase(
        config=DatabaseConfig(db_path=str(Path(tmpdir) / "test.db"), num_islands=1),
        embedding_model="",
    )
    for i in range(n_programs):
        vec = rng.normal(size=dim).tolist()
        db.add(_program(f"p{i}", vec), defer_maintenance=True)
    return db


def _assert_clustered_in_reduced_space(client, dim):
    assert client.cluster_inputs, "the mixture was never fitted"
    fitted = client.cluster_inputs[-1]
    assert fitted.ndim == 2
    assert fitted.shape[1] == 3, (
        "the cluster fit must receive the 3-D PCA projection, not the raw "
        f"{fitted.shape[1]}-d embedding array"
    )
    assert fitted.shape[1] != dim


def test_sync_recompute_clusters_in_pca_space():
    with tempfile.TemporaryDirectory() as tmpdir:
        dim = 32
        db = _seeded_db(tmpdir, dim=dim)
        client = _RecordingEmbeddingClient()
        db.embedding_client = client
        db._ensure_embedding_client = lambda: client
        try:
            db._recompute_embeddings_and_clusters(num_clusters=4)
        finally:
            db.close()
        assert client.reduction_dims == [2, 3]
        _assert_clustered_in_reduced_space(client, dim)


def test_thread_safe_recompute_clusters_in_pca_space():
    with tempfile.TemporaryDirectory() as tmpdir:
        dim = 32
        db = _seeded_db(tmpdir, dim=dim)
        client = _RecordingEmbeddingClient()
        db.embedding_client = client
        db._ensure_embedding_client = lambda: client
        try:
            db._recompute_embeddings_and_clusters_thread_safe(num_clusters=4)
        finally:
            db.close()
        assert client.reduction_dims == [2, 3]
        _assert_clustered_in_reduced_space(client, dim)


def test_persisted_cluster_ids_and_pca_columns_stay_consistent():
    """Every program keeps a 2-D column, a 3-D column and a cluster id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _seeded_db(tmpdir, n_programs=8, dim=32)
        client = _RecordingEmbeddingClient()
        db.embedding_client = client
        db._ensure_embedding_client = lambda: client
        try:
            db._recompute_embeddings_and_clusters(num_clusters=4)
            db.cursor.execute(
                "SELECT embedding_pca_2d, embedding_pca_3d, embedding_cluster_id "
                "FROM programs"
            )
            rows = db.cursor.fetchall()
        finally:
            db.close()

        assert rows
        for row in rows:
            assert len(json.loads(row["embedding_pca_2d"])) == 2
            assert len(json.loads(row["embedding_pca_3d"])) == 3
            assert row["embedding_cluster_id"] is not None
