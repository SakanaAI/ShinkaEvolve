from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VIZ_TREE_HTML = REPO_ROOT / "shinka" / "webui" / "viz_tree.html"


def test_webui_defines_failed_proposal_node_helpers_and_details_section():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert "function isFailedProposalNode(program)" in html
    assert "program.metadata.node_kind === 'failed_proposal'" in html
    assert "function getFailureStatusLabel(program)" in html
    assert "function buildFailureSummaryHtml(data)" in html
    assert "<h5>Failure Summary</h5>" in html
    assert "failure_json_path" in html


def test_webui_program_table_includes_status_column_for_failed_nodes():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert "<th>Status</th>" in html
    assert "const statusLabel = getFailureStatusLabel(prog);" in html
    assert "const isFailedProposal = isFailedProposalNode(prog);" in html


def test_webui_tree_styles_failed_proposal_nodes_distinctly():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert "if (isFailedProposalNode(d.data)) return '#ffb3b3';" in html
    assert "if (isFailedProposalNode(d.data)) return '#c0392b';" in html
    assert ".style(\"stroke-dasharray\", d => isFailedProposalNode(d.data) ? \"6,3\" : null)" in html


def test_webui_tree_resizer_rerenders_graph_instead_of_reloading_data():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert "renderGraph(window.treeData, true);" in html
    assert "window.treeResizeTimeout" in html
    assert "if (window.isResizing) {" in html
