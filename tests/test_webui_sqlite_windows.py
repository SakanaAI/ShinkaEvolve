import pytest

from shinka.webui.visualization import DatabaseRequestHandler, PathValidationError


@pytest.mark.parametrize(
    ("path", "expected_target"),
    [
        (
            r"C:\Users\Rob Lange\programs.sqlite",
            ("file:///C:/Users/Rob%20Lange/programs.sqlite?mode=ro", True),
        ),
        (
            r"\\server\results\programs.sqlite",
            ("file:////server/results/programs.sqlite?mode=ro", True),
        ),
        (
            r"\\?\C:\results\programs.sqlite",
            ("file:///C:/results/programs.sqlite?mode=ro", True),
        ),
        (
            r"\\?\UNC\server\results\programs #1.sqlite",
            (
                "file:////server/results/programs%20%231.sqlite?mode=ro",
                True,
            ),
        ),
        (
            r"\\?\unc\server\results\programs.sqlite",
            ("file:////server/results/programs.sqlite?mode=ro", True),
        ),
    ],
)
def test_sqlite_read_only_target_supports_windows_paths(path, expected_target):
    assert DatabaseRequestHandler._sqlite_read_only_target(path) == expected_target


@pytest.mark.parametrize(
    "path",
    [
        r"\\.\C:\results\programs.sqlite",
        r"\\?\GLOBALROOT\Device\HarddiskVolume1\programs.sqlite",
        r"\\?\Volume{01234567-89ab-cdef-0123-456789abcdef}\programs.sqlite",
    ],
)
def test_sqlite_read_only_target_rejects_device_namespaces(path):
    with pytest.raises(PathValidationError, match="device paths"):
        DatabaseRequestHandler._sqlite_read_only_target(path)
