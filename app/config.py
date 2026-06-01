from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    upstream_base_url: str = Field(
        default="https://api.deepseek.com/anthropic", alias="UPSTREAM_BASE_URL"
    )
    db_path: Path = Field(default=Path("./deepseek_anthropic_proxy.sqlite3"), alias="DB_PATH")
    request_timeout_seconds: float = Field(default=120.0, alias="REQUEST_TIMEOUT_SECONDS")
    enable_thinking_repair: bool = Field(default=True, alias="ENABLE_THINKING_REPAIR")
    repair_strict: bool = Field(default=False, alias="REPAIR_STRICT")
    log_full_request: bool = Field(default=False, alias="LOG_FULL_REQUEST")
    log_dir: Path = Field(default=Path("./log"), alias="LOG_DIR")
    store_all_assistant_content: bool = Field(
        default=False, alias="STORE_ALL_ASSISTANT_CONTENT"
    )

    @property
    def messages_url(self) -> str:
        return self.upstream_url("/v1/messages")

    def upstream_url(self, path: str) -> str:
        return f"{self.upstream_base_url.rstrip('/')}/{path.lstrip('/')}"
