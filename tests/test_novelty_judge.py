from dataclasses import dataclass

import pytest

from shinka.core.novelty_judge import NoveltyJudge
from shinka.database import Program


@dataclass
class DummyResponse:
    content: str
    cost: float = 0.0


class DummyNoveltyLLM:
    def __init__(self, responses=None, raise_on_query=None):
        self.responses = list(responses or [])
        self.raise_on_query = raise_on_query

    def get_kwargs(self):
        return {}

    def query(self, msg, system_msg, llm_kwargs):
        if self.raise_on_query is not None:
            raise self.raise_on_query
        if not self.responses:
            return None

        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class DummyIslandManager:
    def __init__(self, initialized=True):
        self.initialized = initialized

    def are_all_islands_initialized(self):
        return self.initialized


class DummyDatabase:
    def __init__(
        self,
        similarity_sequences,
        most_similar_program=None,
        island_initialized=True,
    ):
        self._similarity_sequences = list(similarity_sequences)
        self.most_similar_program = most_similar_program
        self.island_manager = DummyIslandManager(initialized=island_initialized)

    def compute_similarity(self, code_embedding, island_idx):
        if not self._similarity_sequences:
            return []
        if len(self._similarity_sequences) == 1:
            return self._similarity_sequences[0]
        return self._similarity_sequences.pop(0)

    def get_most_similar_program(self, code_embedding, island_idx):
        return self.most_similar_program


def make_program(program_id="prog-1", island_idx=0):
    return Program(
        id=program_id,
        code="def solve():\n    return 1\n",
        language="python",
        generation=1,
        island_idx=island_idx,
    )


def test_should_check_novelty_rejects_missing_prereqs():
    judge = NoveltyJudge()
    database = DummyDatabase([[]])
    parent_program = make_program()

    assert not judge.should_check_novelty([], 1, parent_program, database)
    assert not judge.should_check_novelty([0.1], 0, parent_program, database)
    assert not judge.should_check_novelty([0.1], 1, None, database)


def test_should_check_novelty_requires_initialized_islands():
    judge = NoveltyJudge()
    parent_program = make_program(island_idx=0)

    initialized_db = DummyDatabase([[]], island_initialized=True)
    assert judge.should_check_novelty([0.1], 1, parent_program, initialized_db)

    uninitialized_db = DummyDatabase([[]], island_initialized=False)
    assert not judge.should_check_novelty([0.1], 1, parent_program, uninitialized_db)

    parent_without_island = make_program(island_idx=None)
    assert not judge.should_check_novelty(
        [0.1], 1, parent_without_island, initialized_db
    )


def test_assess_novelty_accepts_when_similarity_is_empty(tmp_path):
    judge = NoveltyJudge(similarity_threshold=0.9, max_novelty_attempts=3)
    database = DummyDatabase([[]])
    parent_program = make_program()

    exec_file = tmp_path / "candidate.py"
    exec_file.write_text("def candidate():\n    return 123\n", encoding="utf-8")

    accepted, metadata = judge.assess_novelty_with_rejection_sampling(
        exec_fname=str(exec_file),
        code_embedding=[0.1, 0.2],
        parent_program=parent_program,
        database=database,
    )

    assert accepted
    assert metadata["similarity_scores"] == []
    assert metadata["novelty_checks_performed"] == 0
    assert metadata["novelty_total_cost"] == 0.0


def test_assess_novelty_rejects_after_exhausting_attempts(tmp_path):
    judge = NoveltyJudge(similarity_threshold=0.9, max_novelty_attempts=3)
    database = DummyDatabase([[0.95], [0.96], [0.97]])
    parent_program = make_program()

    exec_file = tmp_path / "candidate.py"
    exec_file.write_text("def candidate():\n    return 1\n", encoding="utf-8")

    accepted, metadata = judge.assess_novelty_with_rejection_sampling(
        exec_fname=str(exec_file),
        code_embedding=[0.3, 0.4],
        parent_program=parent_program,
        database=database,
    )

    assert not accepted
    assert metadata["novelty_checks_performed"] == 0
    assert metadata["novelty_total_cost"] == 0.0
    assert metadata["max_similarity"] == pytest.approx(0.97)


def test_assess_novelty_accepts_high_similarity_when_llm_marks_novel(tmp_path):
    novelty_llm = DummyNoveltyLLM(
        responses=[DummyResponse(content="NOVEL: meaningful redesign", cost=0.12)]
    )
    judge = NoveltyJudge(
        novelty_llm_client=novelty_llm,
        similarity_threshold=0.9,
        max_novelty_attempts=3,
    )

    most_similar_program = make_program(program_id="existing")
    database = DummyDatabase(
        similarity_sequences=[[0.99]],
        most_similar_program=most_similar_program,
    )
    parent_program = make_program(program_id="parent")

    exec_file = tmp_path / "candidate.py"
    exec_file.write_text("def candidate():\n    return 42\n", encoding="utf-8")

    accepted, metadata = judge.assess_novelty_with_rejection_sampling(
        exec_fname=str(exec_file),
        code_embedding=[0.5, 0.6],
        parent_program=parent_program,
        database=database,
    )

    assert accepted
    assert metadata["novelty_checks_performed"] == 1
    assert metadata["novelty_total_cost"] == pytest.approx(0.12)
    assert metadata["novelty_explanation"].startswith("NOVEL")


def test_check_llm_novelty_handles_empty_response_and_exception():
    similar_program = make_program(program_id="existing")

    empty_llm = DummyNoveltyLLM(responses=[DummyResponse(content=None, cost=0.5)])
    empty_judge = NoveltyJudge(novelty_llm_client=empty_llm)

    is_novel, explanation, cost = empty_judge.check_llm_novelty(
        proposed_code="def x():\n    return 0\n",
        most_similar_program=similar_program,
    )

    assert is_novel
    assert "empty" in explanation.lower()
    assert cost == 0.0

    failing_llm = DummyNoveltyLLM(raise_on_query=RuntimeError("network down"))
    failing_judge = NoveltyJudge(novelty_llm_client=failing_llm)

    is_novel, explanation, cost = failing_judge.check_llm_novelty(
        proposed_code="def x():\n    return 0\n",
        most_similar_program=similar_program,
    )

    assert is_novel
    assert "network down" in explanation
    assert cost == 0.0
