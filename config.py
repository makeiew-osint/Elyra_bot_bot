import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    hf_token: str
    gemini_api_key: str
    bot_name: str
    database_url: str
    admin_user_ids: tuple[int, ...]
    daily_message_limit: int
    max_file_size_mb: int
    healthcheck_interval_minutes: int
    maintenance_mode: bool

    @classmethod
    def from_env(cls) -> "Settings":
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        hf_token = os.getenv("HF_TOKEN", "").strip()
        gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not telegram_token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")
        if not hf_token:
            raise RuntimeError("HF_TOKEN is not set in .env")
        admin_ids = tuple(
            int(value.strip())
            for value in os.getenv("ADMIN_USER_IDS", "1881090493").split(",")
            if value.strip().isdigit()
        ) or (1881090493,)
        return cls(
            telegram_token=telegram_token,
            hf_token=hf_token,
            gemini_api_key=gemini_api_key,
            bot_name=os.getenv("BOT_NAME", "Elyra_bot_bot").strip() or "Elyra_bot_bot",
            database_url=os.getenv("DATABASE_URL", "").strip(),
            admin_user_ids=admin_ids,
            daily_message_limit=max(0, int(os.getenv("DAILY_MESSAGE_LIMIT", "100"))),
            max_file_size_mb=max(1, int(os.getenv("MAX_FILE_SIZE_MB", "20"))),
            healthcheck_interval_minutes=max(1, int(os.getenv("HEALTHCHECK_INTERVAL_MINUTES", "15"))),
            maintenance_mode=os.getenv("MAINTENANCE_MODE", "true").strip().lower()
            in ("1", "true", "yes", "on", "вкл"),
        )


TEXT_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
REASONING_MODEL = "zai-org/GLM-5.3-Flash"
GEMINI_MODEL = "gemini-3.6-flash"
OCR_MODEL = "zai-org/GLM-OCR"
IMAGE_MODEL = "krea/Krea-2-Turbo"
IMAGE_REALISTIC_MODEL = "black-forest-labs/FLUX.1-dev"
IMAGE_FAST_MODEL = "Tongyi-MAI/Z-Image-Turbo"
EDIT_MODEL = "black-forest-labs/FLUX.2-dev"
SUPPORT_USERNAME = "Makeiew"
