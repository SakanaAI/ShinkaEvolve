from __future__ import annotations

from shinka.logo import full_shinka_ascii, get_logo_ascii, minimal_shinka_ascii


def test_minimal_logo_has_leading_blank_line_and_shinka_cli_wordmark():
    assert minimal_shinka_ascii.startswith("\n")
    assert "░██████╗" in minimal_shinka_ascii
    assert "░█████╗░██╗░░░░░██╗" in minimal_shinka_ascii
    assert "@" not in minimal_shinka_ascii


def test_full_logo_preserves_existing_full_banner_art():
    assert full_shinka_ascii.startswith("  @@@@@@@@@@@@@@@@@@@@@")
    assert "@" in full_shinka_ascii
    assert "░██████╗" in full_shinka_ascii


def test_get_logo_ascii_supports_full_and_minimal_styles():
    assert get_logo_ascii("full") == full_shinka_ascii
    assert get_logo_ascii("minimal") == minimal_shinka_ascii
