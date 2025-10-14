import pandas as pd
import json
import sqlite3
from pathlib import Path
from typing import Optional


def load_programs_to_df(db_path_str: str) -> Optional[pd.DataFrame]:
    """Load program data with normalized component tables into a DataFrame."""
    db_file = Path(db_path_str)
    if not db_file.exists():
        print(f"Error: Database file not found at {db_path_str}")
        return None

    conn = None
    try:
        conn = sqlite3.connect(str(db_file))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM programs")
        program_rows = cursor.fetchall()

        if not program_rows:
            print(f"No programs found in the database: {db_path_str}")
            return pd.DataFrame()

        base_programs = [dict(row) for row in program_rows]
        program_ids = {row["id"] for row in base_programs}

        metrics_map = {pid: {"public": {}, "private": {}} for pid in program_ids}
        cursor.execute(
            "SELECT program_id, public_metrics, private_metrics FROM program_metrics"
        )
        for row in cursor.fetchall():
            program_id = row["program_id"]
            metrics_map.setdefault(program_id, {"public": {}, "private": {}})
            public_json = row["public_metrics"]
            private_json = row["private_metrics"]
            if public_json:
                metrics_map[program_id]["public"] = json.loads(public_json)
            if private_json:
                metrics_map[program_id]["private"] = json.loads(private_json)

        metadata_map = {pid: {} for pid in program_ids}
        cursor.execute("SELECT program_id, metadata FROM program_metadata")
        for row in cursor.fetchall():
            metadata_json = row["metadata"]
            metadata_map[row["program_id"]] = (
                json.loads(metadata_json) if metadata_json else {}
            )

        embedding_map = {pid: [] for pid in program_ids}
        cursor.execute(
            "SELECT program_id, embedding FROM program_embeddings"
        )
        for row in cursor.fetchall():
            embedding_json = row["embedding"]
            embedding_map[row["program_id"]] = (
                json.loads(embedding_json) if embedding_json else []
            )

        inspiration_map = {
            pid: {"archive": [], "top_k": []} for pid in program_ids
        }
        cursor.execute(
            "SELECT program_id, inspiration_id, source FROM program_inspirations"
        )
        for row in cursor.fetchall():
            entry = inspiration_map.setdefault(
                row["program_id"], {"archive": [], "top_k": []}
            )
            if row["source"] == "archive":
                entry["archive"].append(row["inspiration_id"])
            elif row["source"] == "top_k":
                entry["top_k"].append(row["inspiration_id"])

        programs_data = []
        for base in base_programs:
            program_id = base["id"]
            metrics_entry = metrics_map.get(program_id, {"public": {}, "private": {}})
            metadata_dict = metadata_map.get(program_id, {})
            embedding = embedding_map.get(program_id, [])
            inspirations = inspiration_map.get(
                program_id, {"archive": [], "top_k": []}
            )

            timestamp_val = base.get("timestamp")
            try:
                timestamp = (
                    pd.to_datetime(timestamp_val, unit="s")
                    if timestamp_val is not None
                    else None
                )
            except Exception:
                timestamp = None

            flat_data = {
                "id": program_id,
                "code": base.get("code"),
                "language": base.get("language"),
                "parent_id": base.get("parent_id"),
                "archive_inspiration_ids": inspirations.get("archive", []),
                "top_k_inspiration_ids": inspirations.get("top_k", []),
                "generation": base.get("generation"),
                "timestamp": timestamp,
                "complexity": base.get("complexity"),
                "embedding": embedding,
                "code_diff": base.get("code_diff"),
                "correct": bool(base.get("correct", False)),
                "combined_score": base.get("combined_score"),
                "children_count": base.get("children_count"),
                "island_idx": base.get("island_idx"),
                "embedding_cluster_id": base.get("embedding_cluster_id"),
                "text_feedback": base.get("text_feedback") or "",
            }

            flat_data.update(metadata_dict)
            flat_data.update(metrics_entry["public"])
            flat_data.update(metrics_entry["private"])

            programs_data.append(flat_data)

        return pd.DataFrame(programs_data)

    except sqlite3.Error as e:
        print(f"SQLite error while loading {db_path_str}: {e}")
        return None
    except json.JSONDecodeError as e:
        db_path = db_path_str
        print(f"JSON decoding error for metrics/metadata in {db_path}: {e}")
        return None
    finally:
        if conn:
            conn.close()


def get_path_to_best_node(
    df: pd.DataFrame, score_column: str = "combined_score"
) -> pd.DataFrame:
    """
    Finds the chronological path to the node with the highest score.

    Args:
        df: DataFrame containing program data
        score_column: The column name to use for finding the best node
                      (default: "combined_score")

    Returns:
        A DataFrame representing the chronological path to the
        best node, starting from the earliest ancestor and ending with the
        best node.
    """
    if df.empty:
        return pd.DataFrame()

    if score_column not in df.columns:
        raise ValueError(f"Column '{score_column}' not found in DataFrame")

    # Create a dictionary mapping id to row for quick lookups
    id_to_row = {row["id"]: row for _, row in df.iterrows()}

    print(f"Total rows: {len(df)}")
    # Only correct rows
    correct_df = df[df["correct"]]
    print(f"Correct rows: {len(correct_df)}")

    # Find the node with the maximum score
    best_node_row = correct_df.loc[correct_df[score_column].idxmax()]

    # Start building the path with the best node
    path = [best_node_row.to_dict()]
    current_id = best_node_row["parent_id"]

    # Trace back through parent_ids to construct the path
    while current_id is not None and current_id in id_to_row:
        parent_row = id_to_row[current_id]
        path.append(parent_row.to_dict())
        current_id = parent_row["parent_id"]

    # Reverse to get chronological order (oldest first)
    return pd.DataFrame(path[::-1])


def store_best_path(df: pd.DataFrame, results_dir: str):
    best_path = get_path_to_best_node(df)
    path_dir = Path(f"{results_dir}/best_path")
    path_dir.mkdir(exist_ok=True)
    patch_dir = Path(f"{path_dir}/patches")
    patch_dir.mkdir(exist_ok=True)
    code_dir = Path(f"{path_dir}/code")
    code_dir.mkdir(exist_ok=True)
    meta_dir = Path(f"{path_dir}/meta")
    meta_dir.mkdir(exist_ok=True)

    i = 0
    for _, row in best_path.iterrows():
        print(f"\nGeneration {row['generation']} - Score: {row['combined_score']:.2f}")

        if row["code_diff"] is not None:
            patch_path = patch_dir / f"patch_{i}.patch"
            patch_path.write_text(str(row["code_diff"]))
            print(f"Saved patch to {patch_path}")

        base_path = code_dir / f"main_{i}.py"
        base_path.write_text(str(row["code"]))

        # store row data as json, handle non-serializable types
        import datetime

        def default_serializer(obj):
            if isinstance(obj, (datetime.datetime, datetime.date)):
                return obj.isoformat()
            try:
                import pandas as pd

                if isinstance(obj, pd.Timestamp):
                    return obj.isoformat()
            except ImportError:
                pass
            return str(obj)

        row_data_path = meta_dir / f"meta_{i}.json"
        row_data_path.write_text(json.dumps(row.to_dict(), default=default_serializer))
        print(f"Saved meta data to {row_data_path}")
        print(f"Saved base code to {base_path}")
        print(row["patch_name"])
        i += 1
