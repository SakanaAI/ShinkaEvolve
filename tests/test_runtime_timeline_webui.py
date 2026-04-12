from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VIZ_TREE_HTML = REPO_ROOT / "shinka" / "webui" / "viz_tree.html"


def test_runtime_timeline_layout_reserves_space_for_legend():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert "function getRuntimeTimelineLayout(" in html
    assert "margin: { l: 150, r: 10, t: 60, b: 105 }" in html
    assert "laneCount = null" in html
    assert "layout.yaxis.range = [laneCount - 0.5, -0.5];" in html
    assert "orientation: 'h'" in html
    assert "xanchor: 'left'" in html
    assert "yanchor: 'bottom'" in html
    assert "y: 1.01" in html


def test_embeddings_heatmap_uses_scroll_wrapper_for_full_size_matrix():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert '.attr("id", "main-heatmap-scroll")' in html
    assert '.style("overflow", "auto")' in html
    assert '.style("width", "max-content")' in html


def test_embeddings_heatmap_requires_full_hydration_before_render():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert "function datasetHasCompleteEmbeddings(data)" in html
    assert "const programsNeedingEmbeddings = filteredData.filter(" in html
    assert "return programsNeedingEmbeddings.every(" in html
    assert "if (datasetHasCompleteEmbeddings(window.treeData)) {" in html
    assert "window.fullProgramDataByDb = window.fullProgramDataByDb || {};" in html
    assert "window.fullProgramDataByDb[window.currentDbPath] = fullData;" in html


def test_runtime_timeline_dedupes_source_jobs_and_deprioritizes_island_copies():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert "function getRuntimeTimelineRowPriority(row)" in html
    assert "isIslandCopy: Boolean(meta._spawned_island || meta._is_island_copy)" in html
    assert "const dedupeKey = row.sourceJobId || row.id;" in html
    assert "if (rowPriority > existingPriority)" in html


def test_runtime_timeline_infers_stage_lanes_when_worker_ids_are_missing():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert "function assignStageTimelineLaneIds(" in html
    assert "fallbackLaneKey" in html
    assert "row.samplingWorkerId || row.samplingLaneId" in html
    assert "row.evaluationWorkerId || row.evaluationLaneId" in html
    assert "row.postprocessWorkerId || row.postprocessLaneId" in html


def test_throughput_tab_contains_runtime_and_utilization_sections():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert 'data-tab="throughput"' in html
    assert 'id="throughput"' in html
    assert 'id="throughput-summary"' in html
    assert 'id="throughput-runtime-plot"' in html
    assert 'id="throughput-occupancy-plot"' in html
    assert 'id="throughput-occupancy-percent-plot"' in html
    assert 'id="throughput-eval-distribution-plot"' in html
    assert 'id="throughput-completion-rate-plot"' in html
    assert 'id="throughput-duration-table"' in html
    assert 'id="throughput-utilization-table"' in html
    assert "function updateThroughputTab(selectedNodeId = null)" in html
    assert "function renderThroughputOccupancyPercentPlot(rows, capacities)" in html


def test_meta_panel_uses_update_wording_instead_of_generation_wording():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert '<label for="generation-slider">Meta Update:</label>' in html
    assert "Meta analysis for update ${generation} is not available." in html
    assert 'Meta analysis for this update is not available.' in html
    assert "Failed to load meta analysis for update ${generation}." in html
    assert 'Loading meta analysis for update:' in html
    assert "Load the highest update file by default" in html
    assert "const currentGen = metaData.processed_count ?? metaData.generation;" in html
    assert "Scratchpad - Update ${currentGen}" in html
