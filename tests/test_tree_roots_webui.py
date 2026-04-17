from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VIZ_TREE_HTML = REPO_ROOT / "shinka" / "webui" / "viz_tree.html"


def test_webui_always_virtualizes_remaining_multiple_roots():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert "const hasUnifiedRoot = rootNodes.some(n => n.isUnifiedRoot);" not in html
    assert "if (rootNodes.length > 1) {" in html
    assert "parent_id: null" in html
