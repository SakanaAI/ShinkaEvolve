from pathlib import Path


VIZ_TREE_HTML = Path(__file__).parents[1] / "shinka" / "webui" / "viz_tree.html"


def test_prompt_program_rows_escape_content_and_use_data_event_binding():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert 'onclick="selectProgramFromPromptView(\'${p.id}\')"' not in html
    assert 'data-program-id="${escapeHtml(p.id)}"' in html
    assert 'title="${escapeHtml(name)}">${escapeHtml(name)}' in html
    assert "bindPromptProgramRowHandlers();" in html


def test_full_prompt_escapes_prompt_and_parent_names():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert "${escapeHtml(prompt.name)}" in html
    assert "${escapeHtml(parentPrompt.name ||" in html


def test_viz_tree_defines_one_shared_escape_helper():
    html = VIZ_TREE_HTML.read_text(encoding="utf-8")

    assert html.count("function escapeHtml(text)") == 1
