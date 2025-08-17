import asyncio
import time
from pathlib import Path
from voicetrigger import (
    VoiceTrigger,
    Filter, TextContext,
    ColorLogger, Mode
)

bot = VoiceTrigger(
    model_path="model_small",  # Путь к моделе
    keywords=["Алиса"],  # Не обязательно, может брать автоматически с Filter
    quick_words=["стоп", "назад", "вперед"],  # Не обязательно, может брать автоматически с Filter
    calibration_path=Path("voice_calibration.json"),  # Если не указывать будет пытаться брать из "voice_calibration.json", а если файла не будет то определение будет работать по системным параметрам
    device=None,  # Устройство ввода, если не указывать выберет системное
    logger=ColorLogger(level="debug")  # Логгер
)


state = {"active_until": 0.0}


@bot.keyword(Filter("Алиса"))
async def on_alisa(ctx: TextContext):
    bot.log.info(f"[KW] {ctx.match} mode={ctx.mode}")
    bot.start_recognition_main()
    bot.stop_recognition_keywords()
    state["active_until"] = time.time() + 10.0


@bot.quick(Filter(["стоп", "пауза"]))
async def on_quick(ctx: TextContext):
    bot.log.info(f"[QUICK] {ctx.match} mode={ctx.mode}")
    if ctx.match and ctx.match.lower() == "стоп":
        bot.stop_recognition_main()
        bot.start_recognition_keywords()
        state["active_until"] = 0.0


@bot.text()
async def on_all_text(ctx: TextContext):
    if ctx.match is None and ctx.text:
        bot.log.info(f"[TEXT] mode={ctx.mode} text='{ctx.text}'")


@bot.text(Filter(["привет", "здарова"], lv=10, mode=Mode.normal))
async def on_greeting(ctx: TextContext):
    bot.log.info(f"[GREETING] {ctx.match} text='{ctx.text}' mode={ctx.mode}")


@bot.on_silence()  # Возвращает время с последнего quick_words
async def handle_silence_main(sec: float):
    now = time.time()
    if 0 < state["active_until"] <= now and bot.active_main and sec >= 10.0:
        bot.log.info(f"[Silence] {sec:.1f}s -> back to keywords")
        bot.stop_recognition_main()
        bot.start_recognition_keywords()
        state["active_until"] = 0.0


# @bot.on_kw_silence()  # Возвращает время с последнего keywords
# async def handle_kw_silence(sec: float):
#     if sec >= 5.0:
#         bot.log.debug(f"[KW Silence] {sec:.1f}s with no keywords")


if __name__ == "__main__":
    devices = bot.list_input_devices()  # Вывод всех аудио устройств
    bot.log.debug(f"Available input devices: {devices}")
    try:
        asyncio.run(bot.run(initial_keywords_mode=True))  # Запуск с в начале включенным keywords_mode
    except KeyboardInterrupt:
        bot.log.info("Interrupted by user.")