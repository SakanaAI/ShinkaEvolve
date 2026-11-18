import pytest

from shinka.utils import parse_time_to_seconds


@pytest.mark.parametrize(
    ("time_limit", "seconds"),
    [
        ("01:00:00", 3600),
        ("00:00:00", 0),
        ("00:10:30", 630),
        ("90", 5400),
        ("30:00", 1800),
        ("90:15", 5415),
        ("1-00:00:00", 86400),
        ("2-12", 216000),
        ("1-06:30", 109800),
        ("1-00:00:05", 86405),
        (" 00:05:00 ", 300),
        (30, 1800),
    ],
)
def test_time_limit_accepts_every_slurm_format(time_limit, seconds: int) -> None:
    assert parse_time_to_seconds(time_limit) == seconds


@pytest.mark.parametrize(
    "time_limit",
    ["", "abc", "1:2:3:4", "1-", "-1", "1-2-3", "01:00:xx", "1.5", "00::10", "²"],
)
def test_time_limit_rejects_what_slurm_rejects(time_limit: str) -> None:
    with pytest.raises(ValueError, match="Invalid time limit"):
        parse_time_to_seconds(time_limit)
