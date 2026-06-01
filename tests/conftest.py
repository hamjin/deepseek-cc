import pytest

from app.config import Settings

LOGGING_ENV_KEYS = ("LOG_FULL_REQUEST", "LOG_DIR")


@pytest.fixture(autouse=True)
def isolate_settings_logging_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in LOGGING_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
