from dataclasses import dataclass

import pytest

from shinka.core.summarizer import MetaSummarizer
from shinka.database import Program


@dataclass
class DummyResponse:
    content: str
    cost: float = 0.0


class DummyMetaLLM:
    def __init__(self, batch_responses=None, query_responses=None):
        self.batch_responses = list(batch_responses or [])
        self.query_responses = list(query_responses or [])

    def get_kwargs(self):
        return {"temperature": 0.0}

    def batch_kwargs_query(self, num_samples, msg, system_msg):
        return self.batch_responses

    def query(self, msg, system_msg, llm_kwargs):
        if not self.query_responses:
            return None
        response = self.query_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def make_program(program_id, generation, patch_name="patch", correct=True):
    return Program(
        id=program_id,
        code=f"def fn_{generation}():\\n    return {generation}\\n",
        language="python",
        generation=generation,
        combined_score=float(generation),
        public_metrics={"score": float(generation)},
        text_feedback="good",
        correct=correct,
        metadata={"patch_name": patch_name},
    )


def test_should_update_meta_respects_interval_and_mode():
    sync_no_client = MetaSummarizer(meta_llm_client=None, async_mode=False)
    sync_no_client.add_evaluated_program(make_program("p1", generation=1))
    assert not sync_no_client.should_update_meta(meta_rec_interval=1)

    async_no_client = MetaSummarizer(meta_llm_client=None, async_mode=True)
    async_no_client.add_evaluated_program(make_program("p2", generation=1))
    assert async_no_client.should_update_meta(meta_rec_interval=1)

    with_client = MetaSummarizer(meta_llm_client=DummyMetaLLM(), async_mode=False)
    with_client.add_evaluated_program(make_program("p3", generation=1))
    assert not with_client.should_update_meta(meta_rec_interval=2)
    assert with_client.should_update_meta(meta_rec_interval=1)


def test_update_meta_memory_successfully_updates_state():
    llm_client = DummyMetaLLM(
        batch_responses=[
            DummyResponse(content="Summary for generation 2", cost=0.1),
            DummyResponse(content="Summary for generation 1", cost=0.2),
        ],
        query_responses=[
            DummyResponse(content="Global insight block", cost=0.3),
            DummyResponse(content="1. Recommendation A\n2. Recommendation B", cost=0.4),
        ],
    )

    summarizer = MetaSummarizer(meta_llm_client=llm_client, max_recommendations=2)
    summarizer.add_evaluated_program(
        make_program("prog-2", generation=2, patch_name="p2")
    )
    summarizer.add_evaluated_program(
        make_program("prog-1", generation=1, patch_name="p1")
    )

    recommendations, total_cost = summarizer.update_meta_memory()

    assert recommendations == "1. Recommendation A\n2. Recommendation B"
    assert total_cost == pytest.approx(1.0)
    assert summarizer.meta_scratch_pad == "Global insight block"
    assert summarizer.total_programs_processed == 2
    assert summarizer.get_unprocessed_program_count() == 0
    assert summarizer.get_recommendations_history_count() == 1

    assert "Generation 1" in summarizer.meta_summary
    assert "Generation 2" in summarizer.meta_summary
    assert summarizer.meta_summary.find("Generation 1") < summarizer.meta_summary.find(
        "Generation 2"
    )


def test_update_meta_memory_returns_none_without_client_or_programs():
    no_client = MetaSummarizer(meta_llm_client=None)
    assert no_client.update_meta_memory() == (None, 0.0)

    no_programs = MetaSummarizer(meta_llm_client=DummyMetaLLM())
    assert no_programs.update_meta_memory() == (None, 0.0)


def test_get_sampled_recommendation_parses_multiline(monkeypatch):
    summarizer = MetaSummarizer()
    summarizer.meta_recommendations = (
        "1. First recommendation\nextra detail\n2. Second recommendation"
    )

    monkeypatch.setattr("shinka.core.summarizer.random.choice", lambda items: items[0])

    sampled = summarizer.get_sampled_recommendation()

    assert sampled.startswith("First recommendation")
    assert "extra detail" in sampled
    assert not sampled.startswith("1.")


def test_save_and_load_meta_state_round_trip(tmp_path):
    summarizer = MetaSummarizer()
    tracked_program = make_program("tracked", generation=3, patch_name="tracked_patch")

    summarizer.add_evaluated_program(tracked_program)
    summarizer.meta_summary = "summary"
    summarizer.meta_scratch_pad = "scratch"
    summarizer.meta_recommendations = "1. rec"
    summarizer.meta_recommendations_history = ["1. old"]
    summarizer.total_programs_processed = 7

    state_path = tmp_path / "meta" / "state.json"
    summarizer.save_meta_state(str(state_path))

    loaded = MetaSummarizer()
    assert loaded.load_meta_state(str(state_path))

    assert loaded.meta_summary == "summary"
    assert loaded.meta_scratch_pad == "scratch"
    assert loaded.meta_recommendations == "1. rec"
    assert loaded.meta_recommendations_history == ["1. old"]
    assert loaded.total_programs_processed == 7
    assert len(loaded.evaluated_since_last_meta) == 1
    assert loaded.evaluated_since_last_meta[0].id == "tracked"


def test_load_meta_state_invalid_json_returns_false(tmp_path):
    bad_state = tmp_path / "bad_meta.json"
    bad_state.write_text("{broken json", encoding="utf-8")

    summarizer = MetaSummarizer()
    assert not summarizer.load_meta_state(str(bad_state))


def test_perform_final_summary_writes_output_file(tmp_path):
    llm_client = DummyMetaLLM(
        batch_responses=[DummyResponse(content="Program summary", cost=0.1)],
        query_responses=[
            DummyResponse(content="Global insights", cost=0.2),
            DummyResponse(content="1. Keep successful pattern", cost=0.3),
        ],
    )
    summarizer = MetaSummarizer(meta_llm_client=llm_client, max_recommendations=1)
    program = make_program("prog-final", generation=1, patch_name="pf")
    summarizer.add_evaluated_program(program)

    results_dir = tmp_path / "results"
    assert summarizer.perform_final_summary(str(results_dir), best_program=program)

    output_file = results_dir / "meta" / "meta_1.txt"
    assert output_file.exists()
    output_text = output_file.read_text(encoding="utf-8")

    assert "# INDIVIDUAL PROGRAM SUMMARIES" in output_text
    assert "# GLOBAL INSIGHTS SCRATCHPAD" in output_text
    assert "# META RECOMMENDATIONS" in output_text
