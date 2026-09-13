import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    telegram_token: str
    hf_token: str
    bot_name: str

    @classmethod
    def from_env(cls) -> "Settings":
        telegram_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        hf_token = os.getenv("HF_TOKEN", "").strip()
        if not telegram_token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")
        if not hf_token:
            raise RuntimeError("HF_TOKEN is not set in .env")
        return cls(
            telegram_token=telegram_token,
            hf_token=hf_token,
            bot_name=os.getenv("BOT_NAME", "Elyra_bot_bot").strip() or "Elyra_bot_bot",
        )


TEXT_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
OCR_MODEL = "zai-org/GLM-OCR"
IMAGE_MODEL = "krea/Krea-2-Turbo"
EDIT_MODEL = "black-forest-labs/FLUX.2-dev"
SUPPORT_USERNAME = "Makeiew"
