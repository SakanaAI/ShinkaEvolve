from shinka.database import DatabaseConfig, Program, ProgramDatabase


def make_programs():
    first = Program(
        id="p0",
        code="print(0)",
        generation=0,
        correct=True,
        combined_score=1.0,
        public_metrics={
            "score": 1.0,
            "nested": {"accuracy": 0.5},
            "bulky_text": "not a chart metric",
        },
        private_metrics={"hidden": 2.0},
        metadata={
            "api_costs": 0.10,
            "embed_cost": 0.01,
            "novelty_cost": 0.02,
            "meta_cost": 0.03,
            "patch_type": "init",
            "pipeline_seconds": 2.0,
            "evaluation_seconds": 1.5,
            "model_name": "headless/test-agent",
            "headless_usage_status": "reported",
            "headless_usage_unknown": False,
            "headless_pricing_status": "missing",
            "headless_pricing_unknown": True,
            "headless_cost_basis": None,
            "headless_pricing_source": None,
            "secret_token": "must-not-leak",
        },
    )
    second = Program(
        id="p1",
        code="print(1)",
        generation=1,
        parent_id="p0",
        correct=False,
        combined_score=0.25,
        public_metrics={"score": 0.25},
        metadata={
            "api_costs": 0.20,
            "embed_cost": 0.02,
            "novelty_cost": 0.03,
            "meta_cost": 0.04,
            "patch_type": "diff",
            "llm_result": {"model": "fallback-model"},
            "headless_usage_status": "stale",
            "headless_pricing_unknown": True,
            "embedding": [1.0, 2.0],
        },
    )
    return first, second


def make_db(tmp_path):
    db = ProgramDatabase(DatabaseConfig(db_path=str(tmp_path / "programs.sqlite")))
    first, second = make_programs()
    db.add(first, defer_maintenance=True)
    db.add(second, defer_maintenance=True)
    return db, first, second
