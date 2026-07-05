import tempfile
from pathlib import Path

from shinka.database import DatabaseConfig, Program, ProgramDatabase


def test_program_summary_excludes_full_embeddings_but_keeps_pca_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "summary.db"
        db = ProgramDatabase(
            config=DatabaseConfig(db_path=str(db_path), num_islands=1),
            embedding_model="",
            read_only=False,
        )
        try:
            db.add(
                Program(
                    id="p0",
                    code="def f():\n    return 1\n",
                    correct=True,
                    combined_score=1.0,
                    generation=0,
                    island_idx=0,
                    embedding=[0.1, 0.2, 0.3],
                    embedding_pca_2d=[1.0, 2.0],
                    embedding_pca_3d=[3.0, 4.0, 5.0],
                    embedding_cluster_id=7,
                )
            )

            summaries = db.get_programs_summary()

            assert len(summaries) == 1
            assert "embedding" not in summaries[0]
            assert summaries[0]["embedding_pca_2d"] == [1.0, 2.0]
            assert summaries[0]["embedding_pca_3d"] == [3.0, 4.0, 5.0]
            assert summaries[0]["embedding_cluster_id"] == 7
        finally:
            db.close()


def test_program_summary_includes_text_feedback_for_lightweight_details():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "summary.db"
        db = ProgramDatabase(
            config=DatabaseConfig(db_path=str(db_path), num_islands=1),
            embedding_model="",
            read_only=False,
        )
        try:
            db.add(
                Program(
                    id="failure-node",
                    code="",
                    correct=False,
                    combined_score=0.0,
                    generation=3,
                    text_feedback="proposal failed before evaluation",
                    metadata={"node_kind": "failed_proposal"},
                )
            )

            summaries = db.get_programs_summary()

            assert len(summaries) == 1
            assert summaries[0]["text_feedback"] == "proposal failed before evaluation"
            assert summaries[0]["metadata"]["node_kind"] == "failed_proposal"
        finally:
            db.close()
