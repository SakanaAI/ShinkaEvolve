import pandas as pd
import json
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, Union


def load_programs_to_df(
    db_path_str: str,
    default_db_name: str = "programs.sqlite",
    verbose: bool = True,
    include_prompts: bool = False,
) -> Union[
    Optional[pd.DataFrame], Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]
]:
    """
    Loads the 'programs' table from an SQLite database into a pandas DataFrame.

    Args:
        db_path_str: The path to the SQLite database file or results directory.
        default_db_name: Default database filename if db_path_str is a directory.
        verbose: Whether to print summary statistics.
        include_prompts: If True, also loads the prompt evolution database and
            returns a tuple of (programs_df, prompts_df).

    Returns:
        If include_prompts is False:
            A pandas DataFrame containing program data, or None if an error occurs.
        If include_prompts is True:
            A tuple of (programs_df, prompts_df) where each may be None on error.
        The 'metrics' JSON string is parsed and its key-value pairs are added
        as columns to the DataFrame.
    """
    db_file = Path(db_path_str)

    if db_path_str.endswith(".sqlite") or db_path_str.endswith(".db"):
        db_file = Path(db_path_str)
    else:
        db_file = Path(f"{db_path_str}/{default_db_name}")

    if not db_file.exists():
        print(f"Error: Database file not found at {db_path_str}")
        return None

    conn = None
    try:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM programs")  # Fetch all columns
        all_program_rows = cursor.fetchall()

        if not all_program_rows:
            print(f"No programs found in the database: {db_path_str}")
            return pd.DataFrame()  # Return empty DataFrame if no programs

        # Get column names from cursor.description
        column_names = [description[0] for description in cursor.description]
        # print(column_names)
        programs_data = []
        for row_tuple in all_program_rows:
            # Convert row tuple to dict
            p_dict = dict(zip(column_names, row_tuple))

            # Metrics and metadata are stored as JSON strings
            metrics_json = p_dict.get("metrics", "{}")
            metrics_dict = json.loads(metrics_json) if metrics_json else {}

            # Parse inspiration_ids JSON
            archive_insp_ids_json = p_dict.get("archive_inspiration_ids", "[]")
            archive_insp_ids = (
                json.loads(archive_insp_ids_json) if archive_insp_ids_json else []
            )
            top_k_insp_ids_json = p_dict.get("top_k_inspiration_ids", "[]")
            top_k_insp_ids = (
                json.loads(top_k_insp_ids_json) if top_k_insp_ids_json else []
            )
            metadata_json = p_dict.get("metadata", "{}")
            metadata_dict = json.loads(metadata_json) if metadata_json else {}

            # Parse public_metrics and private_metrics
            public_metrics_raw = p_dict.get("public_metrics", "{}")
            if isinstance(public_metrics_raw, str):
                public_metrics_dict = (
                    json.loads(public_metrics_raw) if public_metrics_raw else {}
                )
            else:
                public_metrics_dict = public_metrics_raw or {}

            private_metrics_raw = p_dict.get("private_metrics", "{}")
            if isinstance(private_metrics_raw, str):
                private_metrics_dict = (
                    json.loads(private_metrics_raw) if private_metrics_raw else {}
                )
            else:
                private_metrics_dict = private_metrics_raw or {}

            embedding = p_dict.get("embedding", [])
            if isinstance(embedding, str):
                embedding = json.loads(embedding)
            # Create a flat dictionary for the DataFrame
            try:
                timestamp = pd.to_datetime(p_dict.get("timestamp"), unit="s")
            except Exception:
                timestamp = None
            flat_data = {
                "id": p_dict.get("id"),
                "code": p_dict.get("code"),
                "language": p_dict.get("language"),
                "parent_id": p_dict.get("parent_id"),
                "archive_inspiration_ids": archive_insp_ids,
                "top_k_inspiration_ids": top_k_insp_ids,
                "generation": p_dict.get("generation"),
                "timestamp": timestamp,
                "complexity": p_dict.get("complexity"),
                "embedding": embedding,
                "code_diff": p_dict.get("code_diff"),
                "correct": bool(p_dict.get("correct", False)),
                "combined_score": p_dict.get("combined_score"),
                **metadata_dict,
                **public_metrics_dict,
                **private_metrics_dict,
                "text_feedback": p_dict.get("text_feedback", ""),
            }
            flat_data.update(metrics_dict)
            programs_data.append(flat_data)
        if verbose:
            print(f"Total program rows: {len(programs_data)}")
            print(
                f"Correct program rows: {len([p for p in programs_data if p['correct']])}"
            )
        programs_df = pd.DataFrame(programs_data)

        # Calculate total cost and add as a column to the DataFrame for each program individually
        # Compute total_cost while handling missing columns gracefully
        cost_cols = ["embed_cost", "novelty_cost", "api_costs", "meta_cost"]
        present_cols = [col for col in cost_cols if col in programs_df.columns]
        programs_df["total_cost"] = programs_df[present_cols].sum(axis=1)

        if verbose:
            print(f"Total cost: ${programs_df['total_cost'].sum():.2f}")
            print(f"Avg cost per program: ${programs_df['total_cost'].mean():.2f}")

        if include_prompts:
            # Determine prompts database path
            if db_path_str.endswith(".sqlite") or db_path_str.endswith(".db"):
                results_dir = Path(db_path_str).parent
            else:
                results_dir = Path(db_path_str)
            prompts_df = load_prompts_to_df(str(results_dir), verbose=verbose)
            return programs_df, prompts_df

        return programs_df

    except sqlite3.Error as e:
        print(f"SQLite error while loading {db_path_str}: {e}")
        return (None, None) if include_prompts else None
    except json.JSONDecodeError as e:
        db_path = db_path_str
        print(f"JSON decoding error for metrics/metadata in {db_path}: {e}")
        return (None, None) if include_prompts else None
    finally:
        if conn:
            conn.close()


def load_prompts_to_df(
    db_path_str: str,
    default_db_name: str = "prompts.sqlite",
    verbose: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Loads the 'system_prompts' table from the prompt evolution database
    into a pandas DataFrame.

    Args:
        db_path_str: Path to the prompts database file or results directory.
        default_db_name: Default database filename if path is a directory.
        verbose: Whether to print summary statistics.

    Returns:
        A pandas DataFrame containing prompt data, or None if an error occurs
        or no data is found.
    """
    if db_path_str.endswith(".sqlite") or db_path_str.endswith(".db"):
        db_file = Path(db_path_str)
    else:
        db_file = Path(f"{db_path_str}/{default_db_name}")

    if not db_file.exists():
        if verbose:
            print(f"Prompt database not found at {db_file}")
        return None

    conn = None
    try:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()

        # Check if prompt_archive table exists for in_archive status
        cursor.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='prompt_archive'"
        )
        has_archive_table = cursor.fetchone() is not None

        # Fetch prompts with archive status if available
        if has_archive_table:
            cursor.execute(
                """
                SELECT p.*,
                    CASE WHEN a.prompt_id IS NOT NULL
                         THEN 1 ELSE 0 END as in_archive
                FROM system_prompts p
                LEFT JOIN prompt_archive a ON p.id = a.prompt_id
                """
            )
        else:
            cursor.execute("SELECT *, 0 as in_archive FROM system_prompts")

        all_prompt_rows = cursor.fetchall()

        if not all_prompt_rows:
            if verbose:
                print(f"No prompts found in the database: {db_file}")
            return pd.DataFrame()

        # Get column names from cursor.description
        column_names = [description[0] for description in cursor.description]

        prompts_data = []
        for row_tuple in all_prompt_rows:
            p_dict = dict(zip(column_names, row_tuple))

            # Parse JSON fields
            program_ids_json = p_dict.get("program_ids", "[]")
            program_ids = json.loads(program_ids_json) if program_ids_json else []

            metadata_json = p_dict.get("metadata", "{}")
            metadata_dict = json.loads(metadata_json) if metadata_json else {}

            # Extract and flatten "llm" dict from metadata with prefix
            llm_dict = metadata_dict.pop("llm", {}) or {}
            llm_flat = {f"llm_{k}": v for k, v in llm_dict.items()}

            # Parse timestamp
            try:
                timestamp = pd.to_datetime(p_dict.get("timestamp"), unit="s")
            except Exception:
                timestamp = None

            flat_data = {
                "id": p_dict.get("id"),
                "prompt_text": p_dict.get("prompt_text"),
                "name": p_dict.get("name"),
                "description": p_dict.get("description"),
                "parent_id": p_dict.get("parent_id"),
                "generation": p_dict.get("generation"),
                "program_generation": p_dict.get("program_generation"),
                "patch_type": p_dict.get("patch_type"),
                "timestamp": timestamp,
                "program_count": p_dict.get("program_count", 0),
                "correct_program_count": p_dict.get("correct_program_count", 0),
                "total_improvement": p_dict.get("total_improvement", 0.0),
                "fitness": p_dict.get("fitness", 0.0),
                "program_ids": program_ids,
                "in_archive": bool(p_dict.get("in_archive", 0)),
                **metadata_dict,
                **llm_flat,
            }
            prompts_data.append(flat_data)

        if verbose:
            print(f"Total prompt rows: {len(prompts_data)}")
            archive_count = len([p for p in prompts_data if p["in_archive"]])
            print(f"Prompts in archive: {archive_count}")

        return pd.DataFrame(prompts_data)

    except sqlite3.Error as e:
        print(f"SQLite error while loading prompts from {db_file}: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON decoding error for prompts metadata in {db_file}: {e}")
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

    # Only correct rows
    correct_df = df[df["correct"]]

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
