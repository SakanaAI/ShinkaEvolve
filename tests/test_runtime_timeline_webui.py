from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VIZ_TREE_HTML = REPO_ROOT / "shinka" / "webui" / "viz_tree.html"


def test_runtime_timeline_layout_reserves_space_for_legend():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert "function getRuntimeTimelineLayout(" in html
    assert "const RUNTIME_TIMELINE_ROW_HEIGHT = 34;" in html
    assert "const RUNTIME_TIMELINE_MIN_HEIGHT = 220;" in html
    assert "const RUNTIME_TIMELINE_BASE_CHROME = 140;" in html
    assert "function getRuntimeTimelinePlotHeight(laneCount)" in html
    assert "RUNTIME_TIMELINE_BASE_CHROME + (safeLaneCount * RUNTIME_TIMELINE_ROW_HEIGHT)" in html
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


def test_embeddings_heatmap_requires_hydrated_embedding_fields_before_render():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert "function datasetHasHydratedEmbeddingFields(data)" in html
    assert "const programsNeedingEmbeddings = filteredData.filter(" in html
    assert "Object.prototype.hasOwnProperty.call(program, 'embedding')" in html
    assert "return programsNeedingEmbeddings.every(" in html
    assert "if (datasetHasHydratedEmbeddingFields(window.treeData)) {" in html
    assert "window.fullProgramDataByDb = window.fullProgramDataByDb || {};" in html
    assert "window.fullProgramDataByDb[window.currentDbPath] = fullData;" in html


def test_embeddings_heatmap_refetches_when_cached_full_data_is_stale():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert "function fullProgramDataCoversEmbeddingHydration(existingData, fullData)" in html
    assert "const cachedFullData = window.fullProgramDataByDb[window.currentDbPath];" in html
    assert (
        "if (cachedFullData && fullProgramDataCoversEmbeddingHydration(window.treeData, cachedFullData))"
        in html
    )
    assert "console.log(\"[DEBUG] Cached full embedding data is stale, refetching\")" in html


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


def test_runtime_timeline_uses_adaptive_height_and_fixed_lane_thickness():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert 'id="throughput-runtime-plot" style="width: 100%; overflow: hidden;"' in html
    assert "const runtimePlotHeight = getRuntimeTimelinePlotHeight(laneLabels.length);" in html
    assert "const runtimePlotHeight = getRuntimeTimelinePlotHeight(laneCount);" in html
    assert "const poolLineWidth = getRuntimeTimelineLineWidth();" in html
    assert "const legacyBarWidth = getRuntimeTimelineBarWidth();" in html
    assert "width: poolLineWidth" in html
    assert "width: rows.map(() => legacyBarWidth)" in html
    assert "const runtimePlotHeight = 420;" not in html


def test_runtime_timeline_can_be_toggled_visible_by_default():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert 'id="throughput-runtime-toggle"' in html
    assert 'onclick="toggleThroughputRuntimePlot()"' in html
    assert 'id="throughput-runtime-plot-wrapper"' in html
    assert "window.isThroughputRuntimePlotVisible = true;" in html
    assert "function setThroughputRuntimePlotVisibility(isVisible, rerender = false)" in html
    assert "function toggleThroughputRuntimePlot()" in html
    assert "button.textContent = isVisible ? 'Hide Plot' : 'Show Plot';" in html
    assert "wrapper.style.display = isVisible ? 'block' : 'none';" in html
    assert "renderRuntimeTimelinePlot(window.treeData, window.selectedNodeId || null, 'throughput-runtime-summary', 'throughput-runtime-plot');" in html


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


def test_throughput_distribution_and_completion_plots_stack_vertically():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert (
        '<div style="display: grid; grid-template-columns: 1fr; gap: 10px; min-width: 0;">'
        in html
    )
    assert "<h3>Evaluation Occupancy Distribution</h3>" in html
    assert "<h3>Completed Evaluations per Minute</h3>" in html
    assert '<div style="min-width: 0; overflow: hidden;">' in html
    assert (
        'id="throughput-eval-distribution-plot" style="width: 100%; max-width: 100%; height: 240px; overflow: hidden;"'
        in html
    )
    assert (
        'id="throughput-completion-rate-plot" style="width: 100%; max-width: 100%; height: 240px; overflow: hidden;"'
        in html
    )
    assert "#throughput-eval-distribution-plot .svg-container," in html
    assert "#throughput-completion-rate-plot .svg-container {" in html
    assert "max-width: 100% !important;" in html
    assert "overflow: hidden !important;" in html
    assert "height: 240," in html
    assert "margin: { l: 50, r: 10, t: 30, b: 38 }" in html
    assert "tickfont: { size: 10 }" in html
    assert "titlefont: { size: 11 }" in html


def test_tab_labels_use_shortened_scratch_and_eval_text():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert '<div class="tab" data-tab="meta-analysis">Scratch</div>' in html
    assert '<div class="tab" data-tab="log-output">Eval</div>' in html
    assert '<div class="tab" data-tab="meta-analysis">Scratchpad</div>' not in html
    assert '<div class="tab" data-tab="log-output">Evaluation</div>' not in html


def test_throughput_tab_uses_usage_label():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert '<div class="tab" data-tab="throughput">Usage</div>' in html
    assert '<div class="tab" data-tab="throughput">Throughput</div>' not in html


def test_normalized_occupancy_plot_uses_short_util_label():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert "name: `${stage.label} Util.`" in html
    assert "name: `${stage.label} Utilization`" not in html
    assert "name: '100% Cap.'" in html
    assert "name: '100% Capacity'" not in html
    assert (
        "legend: { orientation: 'h', x: 0.01, y: 1.12, font: { size: 11 } }" in html
    )


def test_meta_panel_uses_update_wording_instead_of_generation_wording():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert "function getCurrentResultsDir()" in html
    assert '<h3 id="meta-info-title">${metaTitle}</h3>' in html
    assert "Info: <code>${escapeHtml(resultsDir)}</code>" in html
    assert '<label for="generation-slider">Meta Update:</label>' in html
    assert "Meta analysis for update ${generation} is not available." in html
    assert 'Meta analysis for this update is not available.' in html
    assert "Failed to load meta analysis for update ${generation}." in html
    assert 'Loading meta analysis for update:' in html
    assert "Load the highest update file by default" in html
    assert "const currentGen = metaData.processed_count ?? metaData.generation;" in html
    assert "Scratchpad - Update ${currentGen}" in html
