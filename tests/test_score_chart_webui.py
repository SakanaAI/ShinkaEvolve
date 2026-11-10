from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VIZ_TREE_HTML = REPO_ROOT / "shinka" / "webui" / "viz_tree.html"


def test_webui_score_chart_click_selects_individual_solutions_dataset():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    # Datasets are: 0 Cumulative Max, 1 Best Path, 2 Cumulative Mean, 3 Individual Solutions
    assert "if (datasetIndex === 3) { // Individual Solution Scores" in html
    assert "if (datasetIndex === 2) { // Individual Solution Scores" not in html


def test_webui_tree_render_cache_does_not_drop_pending_node_selection():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert "if (!forceRender && !window.pendingNodeSelectionId && currentHash === lastRenderedDataHash" in html


def test_webui_score_charts_hold_previous_max_until_improvement_is_evaluated():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    # Chart.js 'before' holds the previous score until the next point's x value.
    assert "stepped: 'before'," in html
    assert "const isCumulativeMaxImprovement = (ctx) => {" in html
    assert html.count("...cumulativeMaxStepStyle") == 3
