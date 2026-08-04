from shinka.database import Program
from shinka.prompts.prompts_cross import get_cross_component


def _program(program_id, embedding):
    return Program(id=program_id, code=program_id, embedding=embedding)


def test_cross_component_selects_most_distant_inspiration():
    parent = _program("parent", [1.0, 0.0])
    near = _program("near", [0.99, 0.1])
    far = _program("far", [0.0, 1.0])

    component = get_cross_component(
        [near], [far], parent=parent, distance_metric="cosine"
    )

    assert "far" in component
    assert "near" not in component


def test_cross_component_supports_configured_distance_metric():
    parent = _program("parent", [0.0, 1.0])
    near = _program("near", [0.0, 2.0])
    far = _program("far", [4.0, 0.0])

    component = get_cross_component(
        [near], [far], parent=parent, distance_metric="euclidean"
    )

    assert "far" in component
    assert "near" not in component


def test_cross_component_falls_back_to_random_without_embeddings(monkeypatch):
    parent = _program("parent", [])
    random_inspiration = _program("random", [])

    monkeypatch.setattr(
        "shinka.prompts.prompts_cross.random.choice",
        lambda inspirations: random_inspiration,
    )

    component = get_cross_component([random_inspiration], [], parent=parent)

    assert "random" in component
