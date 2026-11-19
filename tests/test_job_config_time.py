import pytest

from shinka.launch import LocalJobConfig, SlurmCondaJobConfig, SlurmDockerJobConfig


@pytest.mark.parametrize("config", [SlurmCondaJobConfig, SlurmDockerJobConfig])
@pytest.mark.parametrize(
    "time_limit", ["1-00:00:00", "30:00", "120", "UNLIMITED", "infinite", "-1"]
)
def test_slurm_config_accepts_slurm_time_formats(config, time_limit: str) -> None:
    assert config(time=time_limit).time == time_limit


@pytest.mark.parametrize("config", [SlurmCondaJobConfig, SlurmDockerJobConfig])
def test_slurm_config_rejects_an_invalid_time_before_any_job_runs(config) -> None:
    with pytest.raises(ValueError, match="Invalid time limit 'one hour'"):
        config(time="one hour")


def test_local_config_validates_a_configured_time() -> None:
    assert LocalJobConfig().time is None
    assert LocalJobConfig(time="").time == ""
    assert LocalJobConfig(time="1-00:00:00").time == "1-00:00:00"
    with pytest.raises(ValueError, match="Invalid time limit 'soon'"):
        LocalJobConfig(time="soon")
