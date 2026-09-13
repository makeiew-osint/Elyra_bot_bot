import asyncio
import base64
import logging
import tempfile
from enum import Enum
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
from aiogram.exceptions import TelegramAPIError
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    BotCommand,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from huggingface_hub import InferenceClient
from google import genai
from google.genai import types

from config import (
    EDIT_MODEL,
    GEMINI_MODEL,
    IMAGE_MODEL,
    OCR_MODEL,
    REASONING_MODEL,
    SUPPORT_USERNAME,
    TEXT_MODEL,
    Settings,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
router = Router()


class Mode(str, Enum):
    CHAT = "chat"
    CODE = "code"
    AGENT = "agent"
    OCR = "ocr"
    IMAGE = "image"
    EDIT = "edit"


class UserFlow(StatesGroup):
    waiting_for_prompt = State()
    waiting_for_image_prompt = State()
    waiting_for_edit_image = State()
    waiting_for_edit_prompt = State()


def main_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🧠 Умный чат", callback_data="mode:chat"),
                InlineKeyboardButton(text="💻 Решить код", callback_data="mode:code"),
            ],
            [
                InlineKeyboardButton(text="🤖 Агент", callback_data="mode:agent"),
                InlineKeyboardButton(text="📷 Решить фото", callback_data="mode:ocr"),
            ],
            [
                InlineKeyboardButton(text="🎨 Создать картинку", callback_data="mode:image"),
                InlineKeyboardButton(text="✏️ Изменить фото", callback_data="mode:edit"),
            ],
            [
                InlineKeyboardButton(text="ℹ️ О боте", callback_data="about"),
                InlineKeyboardButton(text="🆘 Поддержка", url=f"https://t.me/{SUPPORT_USERNAME}"),
            ],
        ]
    )


def back_menu() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="⬅️ В меню", callback_data="menu")],
            [InlineKeyboardButton(text="🆘 Поддержка", url=f"https://t.me/{SUPPORT_USERNAME}")],
        ]
    )


ABOUT_TEXT = (
    "✨ <b>Elyra</b> — твой AI-помощник в Telegram.\n\n"
    "🧠 <b>Умный чат</b> — вопросы и объяснения\n"
    "💻 <b>Решить код</b> — программирование\n"
    "🤖 <b>Агент</b> — планы и сложные задачи\n"
    "📷 <b>Решить фото</b> — условие с картинки + решение\n"
    "🎨 <b>Создать картинку</b> — генерация\n"
    "✏️ <b>Изменить фото</b> — редактирование\n\n"
    "Модели: DeepSeek V4.1 Flash, GLM-OCR, Krea-2-Turbo и FLUX.2-dev.\n"
    "Поддержка: @Makeiew"
)


def prompt_for_mode(mode: Mode) -> str:
    return {
        Mode.CHAT: "Напишите вопрос или задачу:",
        Mode.CODE: "Опишите задачу по коду. Укажите язык и ожидаемый результат:",
        Mode.AGENT: "Опишите цель. Я разложу ее на шаги и предложу решение:",
        Mode.OCR: "Отправьте фото или документ для распознавания текста:",
        Mode.IMAGE: "Опишите изображение, которое нужно создать:",
        Mode.EDIT: "Сначала отправьте изображение, затем напишите, что изменить:",
    }[mode]


async def hf_text(client: InferenceClient, prompt: str, mode: Mode) -> str:
    system = {
        Mode.CHAT: "Отвечай на русском ясно и полезно.",
        Mode.CODE: "Ты опытный разработчик. Дай рабочий код и кратко объясни решение.",
        Mode.AGENT: "Ты агент-планировщик. Разбей задачу на безопасные проверяемые шаги.",
    }[mode]
    result = await asyncio.to_thread(
        client.chat_completion,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        model=TEXT_MODEL if mode != Mode.AGENT else REASONING_MODEL,
        max_tokens=2048,
        temperature=0.7,
    )
    return result.choices[0].message.content


def choose_text_backend(settings: Settings, mode: Mode, prompt: str) -> str:
    if mode == Mode.CODE:
        return "deepseek"
    if mode == Mode.AGENT:
        return "glm"
    if settings.gemini_api_key and len(prompt) > 1200:
        return "gemini"
    return "gemini" if settings.gemini_api_key else "deepseek"


async def gemini_text(settings: Settings, prompt: str, mode: Mode) -> str:
    client = genai.Client(api_key=settings.gemini_api_key)
    instruction = {
        Mode.CHAT: "Отвечай по-русски ясно, точно и полезно.",
        Mode.CODE: "Реши задачу по программированию. Дай рабочий код и объяснение.",
        Mode.AGENT: "Разбей задачу на проверяемые шаги и предложи надежное решение.",
    }[mode]
    result = await asyncio.to_thread(
        client.models.generate_content,
        model=GEMINI_MODEL,
        contents=f"{instruction}\n\n{prompt}",
    )
    return result.text


async def hf_image(client: InferenceClient, prompt: str, model: str) -> bytes:
    image = await asyncio.to_thread(client.text_to_image, prompt=prompt, model=model)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        image.save(handle, format="PNG")
        return Path(handle.name).read_bytes()


def temporary_image_path(content: bytes) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        handle.write(content)
        return Path(handle.name)


def data_uri_for_image(content: bytes) -> str:
    if content.startswith(b"\x89PNG"):
        mime_type = "image/png"
    elif content.startswith(b"GIF8"):
        mime_type = "image/gif"
    elif content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        mime_type = "image/webp"
    else:
        mime_type = "image/jpeg"
    return f"data:{mime_type};base64," + base64.b64encode(content).decode("ascii")


async def hf_edit(client: InferenceClient, content: bytes, prompt: str) -> bytes:
    image = await asyncio.to_thread(
        client.image_to_image,
        image=content,
        prompt=prompt,
        model=EDIT_MODEL,
    )
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        image.save(handle, format="PNG")
        return Path(handle.name).read_bytes()


async def hf_ocr(client: InferenceClient, content: bytes) -> str:
    result = await asyncio.to_thread(
        client.chat_completion,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Распознай весь текст на изображении без пропусков. "
                            "Сохрани номера, формулы и переносы строк."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_uri_for_image(content)
                        },
                    },
                ],
            }
        ],
        model=OCR_MODEL,
        max_tokens=2048,
    )
    return result.choices[0].message.content


async def hf_vision_answer(
    client: InferenceClient, content: bytes, prompt: str
) -> str:
    result = await asyncio.to_thread(
        client.chat_completion,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri_for_image(content)},
                    },
                ],
            }
        ],
        model=TEXT_MODEL,
        max_tokens=2048,
    )
    return result.choices[0].message.content


async def gemini_vision_answer(
    settings: Settings, content: bytes, prompt: str
) -> str:
    client = genai.Client(api_key=settings.gemini_api_key)
    result = await asyncio.to_thread(
        client.models.generate_content,
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=content, mime_type="image/jpeg"),
            prompt,
        ],
    )
    return result.text


async def send_error(message: Message, error: Exception) -> None:
    logging.exception("Hugging Face request failed", exc_info=error)
    await message.answer(
        "Не удалось обработать запрос. Проверьте доступность модели Hugging Face "
        "или обратитесь в поддержку @Makeiew.",
        reply_markup=back_menu(),
    )


@router.message(CommandStart())
async def start(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer(
        "✨ <b>Привет! Я Elyra</b>\n\n"
        "Помогу решить задачу текстом или по фотографии. "
        "Выберите нужный режим:",
        reply_markup=main_menu(),
        parse_mode="HTML",
    )


@router.message(Command("help"))
async def help_command(message: Message) -> None:
    await message.answer(
        "Выберите режим в меню и отправьте запрос. Для OCR отправьте фото или PDF.\n"
        "Команда /cancel сбрасывает текущий режим.",
        reply_markup=main_menu(),
    )


@router.message(Command("about"))
async def about_command(message: Message) -> None:
    await message.answer(ABOUT_TEXT, reply_markup=main_menu(), parse_mode="HTML")


@router.message(Command("cancel"))
async def cancel_command(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Текущий режим отменен.", reply_markup=main_menu())


@router.callback_query(F.data == "menu")
async def menu_callback(callback: CallbackQuery, state: FSMContext) -> None:
    await state.clear()
    await callback.message.edit_text(
        "✨ <b>Elyra</b>\n\nВыберите режим работы:",
        reply_markup=main_menu(),
        parse_mode="HTML",
    )
    await callback.answer()


@router.callback_query(F.data == "about")
async def about_callback(callback: CallbackQuery) -> None:
    await callback.message.edit_text(ABOUT_TEXT, reply_markup=back_menu(), parse_mode="HTML")
    await callback.answer()


@router.callback_query(F.data.startswith("mode:"))
async def mode_callback(callback: CallbackQuery, state: FSMContext) -> None:
    mode = Mode(callback.data.split(":", 1)[1])
    await state.clear()
    if mode == Mode.EDIT:
        await state.set_state(UserFlow.waiting_for_edit_image)
    elif mode in (Mode.IMAGE,):
        await state.set_state(UserFlow.waiting_for_image_prompt)
    elif mode == Mode.OCR:
        await state.set_state(UserFlow.waiting_for_prompt)
    else:
        await state.set_state(UserFlow.waiting_for_prompt)
    await state.update_data(mode=mode.value)
    await callback.message.edit_text(prompt_for_mode(mode), reply_markup=back_menu())
    await callback.answer()


@router.message(UserFlow.waiting_for_prompt, F.text)
async def text_request(message: Message, state: FSMContext, settings: Settings) -> None:
    data = await state.get_data()
    mode = Mode(data["mode"])
    if mode == Mode.OCR:
        await message.answer("Для OCR отправьте изображение или PDF.", reply_markup=back_menu())
        return
    try:
        backend = choose_text_backend(settings, mode, message.text)
        if backend == "gemini":
            try:
                answer = await gemini_text(settings, message.text, mode)
            except Exception:
                logging.warning("Gemini failed; falling back to DeepSeek", exc_info=True)
                answer = await hf_text(
                    InferenceClient(token=settings.hf_token), message.text, mode
                )
        else:
            answer = await hf_text(InferenceClient(token=settings.hf_token), message.text, mode)
        await message.answer(answer[:4000], reply_markup=back_menu())
    except Exception as error:
        await send_error(message, error)


@router.message(UserFlow.waiting_for_prompt, F.photo)
async def ocr_photo(message: Message, bot: Bot, state: FSMContext, settings: Settings) -> None:
    data = await state.get_data()
    try:
        mode = Mode(data["mode"])
        file = await bot.get_file(message.photo[-1].file_id)
        buffer = await bot.download_file(file.file_path)
        content = buffer.read()
        client = InferenceClient(token=settings.hf_token)
        if settings.gemini_api_key:
            try:
                extracted = await gemini_vision_answer(
                    settings,
                    content,
                    "Распознай текст и условие задачи на фотографии. "
                    "Сохрани формулы, номера и все важные детали.",
                )
            except Exception:
                logging.warning("Gemini vision failed; using Hugging Face OCR", exc_info=True)
                extracted = await hf_ocr(client, content)
        else:
            try:
                extracted = await hf_ocr(client, content)
            except Exception:
                logging.warning("GLM-OCR failed; using multimodal DeepSeek fallback", exc_info=True)
                extracted = await hf_vision_answer(
                    client,
                    content,
                    "Точно распознай текст на фотографии. Сохрани условие задачи, "
                    "формулы, номера и все важные детали.",
                )
        if not extracted.strip():
            await message.answer("Не удалось распознать текст на фото.", reply_markup=back_menu())
            return
        if mode == Mode.OCR:
            answer = extracted
        else:
            prompt = (
                f"Реши задачу, распознанную с фотографии. "
                f"Покажи ход решения и итоговый ответ.\n\n{extracted}"
            )
            if settings.gemini_api_key:
                try:
                    answer = await gemini_text(settings, prompt, mode)
                except Exception:
                    logging.warning("Gemini failed; falling back to DeepSeek", exc_info=True)
                    answer = await hf_text(client, prompt, mode)
            else:
                answer = await hf_text(client, prompt, mode)
        await message.answer(answer[:4000], reply_markup=back_menu())
    except Exception as error:
        await send_error(message, error)


@router.message(UserFlow.waiting_for_prompt, F.document)
async def ocr_document(message: Message, bot: Bot, state: FSMContext, settings: Settings) -> None:
    data = await state.get_data()
    if data.get("mode") != Mode.OCR.value:
        await message.answer("В этом режиме нужен текстовый запрос.", reply_markup=back_menu())
        return
    try:
        file = await bot.get_file(message.document.file_id)
        buffer = await bot.download_file(file.file_path)
        client = InferenceClient(token=settings.hf_token)
        answer = await hf_ocr(client, buffer.read())
        await message.answer(answer[:4000] or "Текст не найден.", reply_markup=back_menu())
    except Exception as error:
        await send_error(message, error)


@router.message(UserFlow.waiting_for_image_prompt, F.text)
async def image_request(message: Message, state: FSMContext, settings: Settings) -> None:
    try:
        client = InferenceClient(token=settings.hf_token)
        image = await hf_image(client, message.text, IMAGE_MODEL)
        path = temporary_image_path(image)
        await message.answer_photo(FSInputFile(path), caption="Готово ✨", reply_markup=back_menu())
        path.unlink(missing_ok=True)
        await state.clear()
    except Exception as error:
        await send_error(message, error)


@router.message(UserFlow.waiting_for_edit_image, F.photo)
async def edit_image_received(message: Message, state: FSMContext) -> None:
    await state.update_data(file_id=message.photo[-1].file_id)
    await state.set_state(UserFlow.waiting_for_edit_prompt)
    await message.answer("Теперь напишите, что изменить на изображении:", reply_markup=back_menu())


@router.message(UserFlow.waiting_for_edit_image)
async def edit_image_expected(message: Message) -> None:
    await message.answer("Отправьте изображение как фото.", reply_markup=back_menu())


@router.message(UserFlow.waiting_for_edit_prompt, F.text)
async def edit_request(
    message: Message, bot: Bot, state: FSMContext, settings: Settings
) -> None:
    await message.answer(
        "Запрос на редактирование принят, обрабатываю изображение…"
    )
    try:
        data = await state.get_data()
        file = await bot.get_file(data["file_id"])
        buffer = await bot.download_file(file.file_path)
        client = InferenceClient(token=settings.hf_token)
        image = await hf_edit(client, buffer.read(), message.text)
        path = temporary_image_path(image)
        await message.answer_photo(FSInputFile(path), caption="Готово ✨", reply_markup=back_menu())
        path.unlink(missing_ok=True)
        await state.clear()
    except Exception as error:
        await send_error(message, error)


async def main() -> None:
    settings = Settings.from_env()
    bot = Bot(settings.telegram_token)
    try:
        await bot.set_my_commands(
            [
                BotCommand(command="start", description="Главное меню"),
                BotCommand(command="help", description="Помощь"),
                BotCommand(command="about", description="О боте"),
                BotCommand(command="cancel", description="Отменить режим"),
            ]
        )
    except TelegramAPIError:
        logging.warning("Bot command setup skipped; Telegram rate limit or API error")
    dp = Dispatcher()
    dp["settings"] = settings
    dp.include_router(router)
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logging.info("Bot stopped")
