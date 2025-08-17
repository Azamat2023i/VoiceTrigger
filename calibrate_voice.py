"""
calibrate_voice.py

Калибровка голоса с усреднением. Для каждого уровня (quiet, normal, loud) делает несколько
записей, усредняет RMS и HF, и сохраняет адаптивные пороги в voice_calibration.json.
"""

import json
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

OUT_FILE = Path("voice_calibration.json")


def record_seconds(seconds: float, samplerate: int = 16000, channels: int = 1) -> np.ndarray:
    print(f"Recording {seconds:.1f}s... speak now.")
    rec = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=channels, dtype='int16')
    sd.wait()
    if rec.ndim > 1:
        rec = rec.mean(axis=1)
    return rec.astype(np.int16).astype(np.float32) / 32768.0


def windowed_rms_db(data: np.ndarray, samplerate: int, win_sec: float = 0.2):
    win_len = max(1, int(win_sec * samplerate))
    n = len(data)
    if n < win_len:
        rms = np.sqrt(np.mean(np.square(data)))
        return np.array([20 * np.log10(rms + 1e-12)])
    dbs = []
    for start in range(0, n - win_len + 1, win_len):
        w = data[start:start + win_len]
        rms = np.sqrt(np.mean(np.square(w)))
        dbs.append(20 * np.log10(rms + 1e-12))
    return np.array(dbs)


def windowed_hf_ratio(data: np.ndarray, samplerate: int, win_sec: float = 0.2):
    from numpy.fft import rfft, rfftfreq
    win_len = max(4, int(win_sec * samplerate))
    n = len(data)
    if n < win_len:
        fft = rfft(data)
        mag = np.abs(fft)
        freqs = rfftfreq(len(data), 1 / samplerate)
        low = mag[freqs < 1000].sum() + 1e-12
        high = mag[freqs >= 1000].sum() + 1e-12
        return np.array([high / low])
    res = []
    for start in range(0, n - win_len + 1, win_len):
        w = data[start:start + win_len]
        fft = rfft(w)
        mag = np.abs(fft)
        freqs = rfftfreq(len(w), 1 / samplerate)
        low = mag[freqs < 1000].sum() + 1e-12
        high = mag[freqs >= 1000].sum() + 1e-12
        res.append(high / low)
    return np.array(res)


def summarize_sample(data: np.ndarray, sr: int):
    dbs = windowed_rms_db(data, sr)
    hfs = windowed_hf_ratio(data, sr)
    return {
        "db_mean": float(np.mean(dbs)),
        "db_std": float(np.std(dbs)),
        "hf_mean": float(np.mean(hfs)),
        "hf_std": float(np.std(hfs)),
        "windows": int(len(dbs))
    }


def multi_record(level_name: str, repeats: int = 3, seconds: float = 4.0, samplerate: int = 16000):
    samples = []
    for i in range(repeats):
        print(f"\nЗапись {i+1}/{repeats} для уровня '{level_name}'")
        input(f"Нажми Enter и говори {level_name} ...")
        time.sleep(0.3)
        data = record_seconds(seconds, samplerate)
        samples.append(summarize_sample(data, samplerate))
    # усреднение
    avg = {}
    for key in ["db_mean", "db_std", "hf_mean", "hf_std"]:
        avg[key] = float(np.mean([s[key] for s in samples]))
    avg["windows"] = int(np.sum([s["windows"] for s in samples]))
    return avg


def main():
    sr = 16000
    repeats = 3
    seconds = 4.0

    print("Калибровка голоса с усреднением нескольких записей")
    quiet = multi_record("тихо/шёпот", repeats, seconds, sr)
    normal = multi_record("нормальная речь", repeats, seconds, sr)
    loud = multi_record("громко", repeats, seconds, sr)

    out = {
        "samplerate": sr,
        "seconds_per_sample": seconds,
        "repeats": repeats,
        "quiet": quiet,
        "normal": normal,
        "loud": loud,
        "created_at": time.time()
    }

    OUT_FILE.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nКалибровка сохранена в {OUT_FILE.resolve()}")
    print("Средние dB (db_mean):")
    print(f" quiet:  {quiet['db_mean']:.2f}")
    print(f" normal: {normal['db_mean']:.2f}")
    print(f" loud:   {loud['db_mean']:.2f}")
    print("HF средние:", f"{quiet['hf_mean']:.2f}", f"{normal['hf_mean']:.2f}", f"{loud['hf_mean']:.2f}")


if __name__ == "__main__":
    main()
