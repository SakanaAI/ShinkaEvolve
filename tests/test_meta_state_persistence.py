"""The meta summarizer keeps its state in the run directory."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from shinka.core.async_summarizer import AsyncMetaSummarizer
from shinka.core.summarizer import META_STATE_FILENAME, MetaSummarizer
from shinka.database import Program


@dataclass
class _Response:
    content: str
    cost: float = 0.0


class _AsyncMetaLLM:
    """Summarizes every program and answers the two follow-up queries."""

    def __init__(self) -> None:
        self.updates = 0

    async def batch_kwargs_query_indexed(self, num_samples, msg, system_msg):
        return [(i, _Response(f"summary {i}")) for i in range(num_samples)]

    async def query(self, msg, system_msg):
        self.updates += 1
        step = "insights" if self.updates % 2 else "1. recommendation"
        return _Response(f"{step} (update {(self.updates + 1) // 2})")


def _program(program_id: str, generation: int = 1) -> Program:
    return Program(
        id=program_id,
        code=f"def f():\n    return {generation}\n",
        generation=generation,
        combined_score=float(generation),
        correct=True,
        embedding=[0.25] * 64,
        metadata={"patch_name": f"patch-{program_id}"},
    )


def _summarizer(results_dir: Path) -> AsyncMetaSummarizer:
    summarizer = AsyncMetaSummarizer(MetaSummarizer(async_mode=True), _AsyncMetaLLM())
    summarizer.persist_state_in(str(results_dir))
    return summarizer


def _run_update(summarizer: AsyncMetaSummarizer, results_dir: Path) -> None:
    async def run():
        recommendations, _ = await summarizer.update_meta_memory_async()
        assert recommendations
        await summarizer.write_meta_output_async(str(results_dir))

    asyncio.run(run())


def _resumed(results_dir: Path) -> AsyncMetaSummarizer:
    summarizer = _summarizer(results_dir)
    assert summarizer.restore_persisted_state()
    return summarizer


def test_state_is_not_written_without_a_run_directory(tmp_path):
    summarizer = MetaSummarizer()

    summarizer.add_evaluated_program(_program("a"))

    assert list(tmp_path.iterdir()) == []
    assert not summarizer.restore_persisted_state()


def test_tracked_programs_survive_an_interrupted_session(tmp_path):
    first = _summarizer(tmp_path)
    first.add_evaluated_program(_program("a", 1))
    first.add_evaluated_program(_program("b", 2))

    resumed = _resumed(tmp_path)

    tracked = resumed.evaluated_since_last_meta
    assert [(p.id, p.generation, p.code) for p in tracked] == [
        ("a", 1, "def f():\n    return 1\n"),
        ("b", 2, "def f():\n    return 2\n"),
    ]
    assert tracked[0].metadata["patch_name"] == "patch-a"
    assert resumed.should_update_meta(2)


def test_state_file_leaves_out_embeddings(tmp_path):
    summarizer = _summarizer(tmp_path)

    summarizer.add_evaluated_program(_program("a"))

    state = json.loads((tmp_path / META_STATE_FILENAME).read_text())
    assert state["unprocessed_programs"][0]["embedding"] == []
    assert summarizer.evaluated_since_last_meta[0].embedding == [0.25] * 64


def test_update_results_survive_an_interrupted_session(tmp_path):
    first = _summarizer(tmp_path)
    first.add_evaluated_program(_program("a", 1))
    first.add_evaluated_program(_program("b", 2))
    _run_update(first, tmp_path)
    first.add_evaluated_program(_program("c", 3))

    resumed = _resumed(tmp_path)

    assert resumed.get_current() == first.get_current()
    assert resumed.meta_scratch_pad == "insights (update 1)"
    assert resumed.meta_recommendations == "1. recommendation (update 1)"
    assert resumed.meta_recommendations_history == ["1. recommendation (update 1)"]
    assert resumed.total_programs_processed == 2
    assert [p.id for p in resumed.evaluated_since_last_meta] == ["c"]


def test_resumed_session_does_not_overwrite_earlier_meta_output(tmp_path):
    first = _summarizer(tmp_path)
    first.add_evaluated_program(_program("a", 1))
    first.add_evaluated_program(_program("b", 2))
    _run_update(first, tmp_path)
    first_output = (tmp_path / "meta" / "meta_2.txt").read_text()

    resumed = _resumed(tmp_path)
    resumed.add_evaluated_program(_program("c", 3))
    resumed.add_evaluated_program(_program("d", 4))
    _run_update(resumed, tmp_path)

    assert (tmp_path / "meta" / "meta_2.txt").read_text() == first_output
    second_output = (tmp_path / "meta" / "meta_4.txt").read_text()
    # The new session builds on the earlier summaries instead of starting over.
    assert second_output.count("**Program Identifier:**") == 4


def test_replayed_program_is_tracked_once(tmp_path):
    first = _summarizer(tmp_path)
    first.add_evaluated_program(_program("a"))

    resumed = _resumed(tmp_path)
    resumed.add_evaluated_program(_program("a"))
    resumed.add_evaluated_program(_program("b"))

    assert [p.id for p in resumed.evaluated_since_last_meta] == ["a", "b"]


def test_state_is_recovered_from_meta_output_of_older_sessions(tmp_path):
    """Sessions before the state file existed left only meta/meta_<N>.txt."""
    first = _summarizer(tmp_path)
    for index in range(12):
        first.add_evaluated_program(_program(f"p{index}", index))
        if index in (4, 11):
            _run_update(first, tmp_path)
    (tmp_path / META_STATE_FILENAME).unlink()
    assert {p.name for p in (tmp_path / "meta").iterdir()} == {
        "meta_5.txt",
        "meta_12.txt",
    }

    resumed = _resumed(tmp_path)

    assert resumed.total_programs_processed == 12
    assert resumed.meta_summary == first.meta_summary
    assert resumed.meta_scratch_pad == "insights (update 2)"
    assert resumed.meta_recommendations == "1. recommendation (update 2)"
    assert resumed.meta_recommendations_history == ["1. recommendation (update 2)"]
    assert resumed.evaluated_since_last_meta == []


def test_newer_meta_output_wins_over_a_stale_state_file(tmp_path):
    """A state file left by an old release can predate later meta outputs."""
    first = _summarizer(tmp_path)
    for index in range(4):
        first.add_evaluated_program(_program(f"p{index}", index))
        if index == 1:
            _run_update(first, tmp_path)
            first.add_evaluated_program(_program("pending", 9))
            stale_state = (tmp_path / META_STATE_FILENAME).read_text()
    _run_update(first, tmp_path)
    (tmp_path / META_STATE_FILENAME).write_text(stale_state)

    resumed = _resumed(tmp_path)

    assert resumed.total_programs_processed == 5
    assert resumed.meta_summary == first.meta_summary
    assert resumed.meta_scratch_pad == "insights (update 2)"
    assert resumed.meta_recommendations == "1. recommendation (update 2)"
    assert resumed.evaluated_since_last_meta == []


def test_nothing_to_restore_in_a_new_run_directory(tmp_path):
    summarizer = _summarizer(tmp_path)

    assert not summarizer.restore_persisted_state()
    assert summarizer.total_programs_processed == 0


def test_sync_update_persists_its_result(tmp_path):
    class _SyncMetaLLM:
        def get_kwargs(self):
            return {}

        def batch_kwargs_query_indexed(self, num_samples, msg, system_msg):
            return [(i, _Response(f"summary {i}")) for i in range(num_samples)]

        def query(self, msg, system_msg, llm_kwargs):
            return _Response("1. sync recommendation")

    summarizer = MetaSummarizer(meta_llm_client=_SyncMetaLLM())  # type: ignore[arg-type]
    summarizer.persist_state_in(str(tmp_path))
    summarizer.add_evaluated_program(_program("a"))
    recommendations, _ = summarizer.update_meta_memory()
    assert recommendations == "1. sync recommendation"

    resumed = MetaSummarizer()
    resumed.persist_state_in(str(tmp_path))

    assert resumed.restore_persisted_state()
    assert resumed.meta_recommendations == "1. sync recommendation"
    assert resumed.total_programs_processed == 1
    assert resumed.evaluated_since_last_meta == []


# -- damaged files ------------------------------------------------------------


def _old_session_output(results_dir: Path) -> AsyncMetaSummarizer:
    """A session of a release without the state file: only meta/meta_2.txt."""
    first = _summarizer(results_dir)
    first.add_evaluated_program(_program("a", 1))
    first.add_evaluated_program(_program("b", 2))
    _run_update(first, results_dir)
    (results_dir / META_STATE_FILENAME).unlink()
    return first


def test_meta_output_cut_inside_a_character_is_still_restored(tmp_path):
    """The output file is not written atomically; a kill can cut it anywhere."""
    first = _old_session_output(tmp_path)
    output = tmp_path / "meta" / "meta_2.txt"
    output.write_bytes(output.read_bytes() + "é".encode()[:1])

    resumed = _resumed(tmp_path)

    assert resumed.total_programs_processed == 2
    assert resumed.meta_summary == first.meta_summary
    assert resumed.meta_scratch_pad == "insights (update 1)"


def test_files_that_only_look_like_meta_output_are_ignored(tmp_path):
    _old_session_output(tmp_path)
    (tmp_path / "meta" / "meta_²3.txt").write_text("not a count")
    (tmp_path / "meta" / "meta_final.txt").write_text("not a count")

    assert _resumed(tmp_path).total_programs_processed == 2


@pytest.mark.parametrize("stored_total", [None, "2", "many"])
def test_state_file_with_an_unusable_program_count_is_restored(tmp_path, stored_total):
    first = _summarizer(tmp_path)
    first.add_evaluated_program(_program("a", 1))
    first.add_evaluated_program(_program("b", 2))
    _run_update(first, tmp_path)
    state_path = tmp_path / META_STATE_FILENAME
    state = json.loads(state_path.read_text())
    state["total_programs_meta_processed"] = stored_total
    state_path.write_text(json.dumps(state))

    resumed = _resumed(tmp_path)

    # The count is taken from the meta output when the state file has none.
    assert resumed.total_programs_processed == 2
    assert resumed.meta_recommendations == "1. recommendation (update 1)"


def test_unreadable_meta_output_does_not_stop_a_resume(tmp_path):
    _old_session_output(tmp_path)
    output = tmp_path / "meta" / "meta_2.txt"
    output.unlink()
    output.mkdir()

    summarizer = _summarizer(tmp_path)

    assert not summarizer.restore_persisted_state()
