import asyncio
import logging
import tempfile
from enum import Enum
from pathlib import Path

from aiogram import Bot, Dispatcher, F, Router
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

from config import (
    EDIT_MODEL,
    IMAGE_MODEL,
    OCR_MODEL,
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
        model=TEXT_MODEL,
        max_tokens=2048,
        temperature=0.7,
    )
    return result.choices[0].message.content


async def hf_image(client: InferenceClient, prompt: str, model: str) -> bytes:
    image = await asyncio.to_thread(client.text_to_image, prompt=prompt, model=model)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        image.save(handle, format="PNG")
        return Path(handle.name).read_bytes()


def temporary_image_path(content: bytes) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        handle.write(content)
        return Path(handle.name)


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
        client.image_to_text,
        image=content,
        model=OCR_MODEL,
    )
    return getattr(result, "text", str(result))


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
    await message.answer(ABOUT_TEXT, reply_markup=main_menu())


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
        client = InferenceClient(token=settings.hf_token)
        answer = await hf_text(client, message.text, mode)
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
        client = InferenceClient(token=settings.hf_token)
        extracted = await hf_ocr(client, buffer.read())
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
    await bot.set_my_name(name=settings.bot_name)
    await bot.set_my_description(
        description="AI-помощник: чат, кодинг, OCR, генерация и редактирование изображений."
    )
    await bot.set_my_commands(
        [
            BotCommand(command="start", description="Главное меню"),
            BotCommand(command="help", description="Помощь"),
            BotCommand(command="about", description="О боте"),
            BotCommand(command="cancel", description="Отменить режим"),
        ]
    )
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
