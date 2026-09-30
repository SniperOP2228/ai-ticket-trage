from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    database_url: str = "postgresql://postgres:password@localhost:5432/ticket_triage"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_debug: bool = False
    confidence_threshold: float = 0.80
    model_version: str = "1.0.0"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
