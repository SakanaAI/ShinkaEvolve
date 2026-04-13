from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = REPO_ROOT / "shinka" / "webui" / "index.html"


def test_dashboard_supports_sorting_by_setting():
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "onclick=\"toggleSort('setting')\"" in html
    assert "let currentSortKey = 'task';" in html
    assert "function getResultDisplayParts(result)" in html
    assert "function compareResults(a, b)" in html
    assert "if (currentSortKey === 'setting')" in html
    assert "filteredResults.sort(compareResults);" in html
