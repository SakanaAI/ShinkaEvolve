from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VIZ_TREE_HTML = REPO_ROOT / "shinka" / "webui" / "viz_tree.html"


def test_runtime_timeline_layout_reserves_space_for_legend():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert "function getRuntimeTimelineLayout(" in html
    assert "margin: { l: 150, r: 10, t: 90, b: 140 }" in html
    assert "orientation: 'h'" in html
    assert "xanchor: 'left'" in html
    assert "yanchor: 'bottom'" in html
    assert "y: 1.02" in html
