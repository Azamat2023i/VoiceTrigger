import asyncio
import signal
import time
import math
import json
import threading
from collections import deque
from enum import Enum
from typing import Callable, Optional, List, Union, Any, Coroutine
import inspect
from pathlib import Path

import numpy as np
import sounddevice as sd
import vosk

import logging
import sys
from colorlog import ColoredFormatter

# Optional libs
try:
    import noisereduce as nr
    _NOISEREDUCE_AVAILABLE = True
except Exception:
    _NOISEREDUCE_AVAILABLE = False

try:
    from scipy.signal import butter, filtfilt
    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False


# ----------------- Utility: ColorLogger -----------------
class ColorLogger:
    """
    Simple colored logger wrapper. Uses English messages only.
    """
    LEVELS = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }

    DEFAULT_COLORS = {
        "DEBUG": "cyan",
        "INFO": "blue",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red,bg_white"
    }

    def __init__(self, name="AppLogger", level="info", colors=None):
        self.logger = logging.getLogger(name)
        self.logger.propagate = False
        self.logger.setLevel(self.LEVELS.get(level, logging.INFO))

        if colors is None:
            colors = self.DEFAULT_COLORS

        formatter = ColoredFormatter(
            "%(log_color)s%(levelname)s | %(message)s",
            log_colors=colors
        )

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def set_level(self, level: str):
        self.logger.setLevel(self.LEVELS.get(level, logging.INFO))

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        # convenience wrapper for full stacktrace logging
        self.logger.exception(msg, *args, **kwargs)


# ----------------- Filter / Mode / Context -----------------
class Mode(Enum):
    whisper = "whisper"
    normal = "normal"
    shout = "shout"


class Filter:
    def __init__(self,
                 phrases: Union[str, List[str], None] = None,
                 lv: int = 10,
                 mode: Optional[Mode] = None):
        if isinstance(phrases, str):
            self.phrases = [phrases]
        elif isinstance(phrases, list):
            self.phrases = list(phrases)
        else:
            self.phrases = []
        self.lv = max(0, int(lv))
        self.mode = mode

    def is_wildcard(self):
        return len(self.phrases) == 0


class TextContext:
    def __init__(self, text: str, mode: str, match: Optional[str], ts: float):
        self.text = text
        self.mode = mode
        self.match = match
        self.timestamp = ts


# ----------------- Levenshtein -----------------
def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def is_match_by_lev(word: str, pattern: str, lv_percent: int) -> bool:
    word = word.lower()
    pattern = pattern.lower()
    dist = levenshtein_distance(word, pattern)
    max_errors = max(1, math.ceil(len(pattern) * lv_percent / 100))
    return dist <= max_errors


# ----------------- VoiceLevelDetector (kept mostly as-is, improved) -----------------
class VoiceLevelDetector:
    """
    Adaptive Voice Level Detector.

    - Accepts `calibration_path` for flexibility.
    - Returns dominant level from recent buffer.
    """
    DEFAULT_CALIB_FILE = Path("voice_calibration.json")

    def __init__(self,
                 samplerate=16000,
                 blocksize=2000,
                 rms_thresholds=None,
                 hf_ratio_threshold=1.5,
                 silence_db=-45,
                 buffer_seconds=None,
                 compute_every_n_blocks=1,
                 hf_weight=0.12,
                 calibration_path: Optional[Path] = None,
                 logger: Optional[ColorLogger] = None):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.hf_ratio_threshold = hf_ratio_threshold
        self.silence_db = silence_db

        self.rms_thresholds = rms_thresholds or {
            "whisper": -43.0,
            "normal": -15.0,
            "shout": 0.0
        }

        self.hf_weight = float(hf_weight)

        if buffer_seconds is not None:
            approx_blocks = max(1, int((samplerate * buffer_seconds) / max(1, blocksize)))
            maxlen = approx_blocks
        else:
            maxlen = 50

        self.audio_buffer = deque(maxlen=maxlen)
        self._lock = threading.Lock()

        self.compute_every_n_blocks = max(1, int(compute_every_n_blocks))
        self._blocks_since_compute = self.compute_every_n_blocks
        self._last_computed_level = "normal"

        self._calib = None
        self.calibration_path = calibration_path or self.DEFAULT_CALIB_FILE
        self.logger = logger or ColorLogger()
        self._load_calibration_if_exists(self.calibration_path)

    def _load_calibration_if_exists(self, path: Path):
        if path and path.exists():
            try:
                raw = json.loads(path.read_text(encoding='utf-8'))
                self._calib = raw
                self._compute_thresholds_from_calib(raw)
                qhf = raw.get("quiet", {}).get("hf_mean")
                nhf = raw.get("normal", {}).get("hf_mean")
                if qhf is not None and nhf is not None:
                    self.hf_ratio_threshold = max(1.0, float(qhf + 0.6 * (nhf - qhf)))
                self.logger.info(f"Voice calibration loaded from {path}")
            except Exception as e:
                self.logger.exception(f"Failed to load calibration from {path}: {e}")
                self._calib = None

    def _compute_thresholds_from_calib(self, calib: dict):
        try:
            q = float(calib["quiet"]["db_mean"])
            n = float(calib["normal"]["db_mean"])
            l = float(calib["loud"]["db_mean"])
        except Exception:
            self.logger.debug("Incomplete calibration data; skipping threshold computation.")
            return

        if not (q <= n <= l):
            sorted_vals = sorted([q, n, l])
            q, n, l = sorted_vals

        whisper_thr = q + max(1.5, (n - q) * 0.25)
        normal_thr = n + max(2.0, (l - n) * 0.5)

        if whisper_thr >= normal_thr - 1.0:
            whisper_thr = n - max(1.0, (n - q) * 0.2)

        self.rms_thresholds = {
            "whisper": float(whisper_thr),
            "normal": float(normal_thr),
            "shout": float(normal_thr)
        }

        self.silence_db = min(self.silence_db, q - 6.0)

    def rms_db(self, data: np.ndarray):
        rms = np.sqrt(np.mean(np.square(data)))
        db = 20 * np.log10(rms + 1e-6)
        return rms, db

    def hf_ratio(self, data: np.ndarray):
        try:
            fft = np.fft.rfft(data)
            mag = np.abs(fft)
            freqs = np.fft.rfftfreq(len(data), 1 / self.samplerate)
            low = mag[freqs < 1000].sum() + 1e-6
            high = mag[freqs >= 1000].sum() + 1e-6
            return float(high / low)
        except Exception:
            return 1.0

    def _decide_by_hybrid(self, db: float, hf: float) -> str:
        whisper_thr = self.rms_thresholds["whisper"]
        normal_thr = self.rms_thresholds["normal"]

        if db < self.silence_db:
            return "silence"
        if db < whisper_thr:
            return "whisper"
        if db >= normal_thr:
            return "shout"

        denom = (normal_thr - whisper_thr) if (normal_thr - whisper_thr) != 0 else 1.0
        db_norm = float(np.clip((db - whisper_thr) / denom, 0.0, 1.0))

        hf_base = 1.0
        hf_thr = max(self.hf_ratio_threshold, hf_base + 0.01)
        hf_score = float(np.clip((hf - hf_base) / (hf_thr - hf_base), 0.0, 1.0))

        alpha = 1.0 - self.hf_weight
        combined = alpha * db_norm + (1.0 - alpha) * hf_score

        if combined < 0.25:
            return "whisper"
        elif combined < 0.75:
            return "normal"
        else:
            return "shout"

    def process_block(self, data_bytes: bytes):
        try:
            data = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception:
            # attempt to handle if bytes are float32 array or numpy array passed
            try:
                arr = np.frombuffer(data_bytes, dtype=np.float32)
                data = arr
            except Exception:
                self.logger.debug("Failed to interpret audio bytes for voice level calculation.", exc_info=True)
                return

        self._blocks_since_compute += 1
        if self._blocks_since_compute >= self.compute_every_n_blocks:
            self._blocks_since_compute = 0
            try:
                rms, db = self.rms_db(data)
                hf = self.hf_ratio(data)
                level = self._decide_by_hybrid(db, hf)
                self._last_computed_level = level
            except Exception:
                self.logger.exception("Error while computing voice level.")
                level = self._last_computed_level
        else:
            level = self._last_computed_level

        with self._lock:
            self.audio_buffer.append(level)

    def get_dominant_level(self):
        with self._lock:
            counts = {}
            for lvl in self.audio_buffer:
                if lvl != "silence":
                    counts[lvl] = counts.get(lvl, 0) + 1
        if not counts:
            return "normal"
        return max(counts, key=counts.get)


# ----------------- AudioStreamManager -----------------
class AudioStreamManager:
    """
    Wraps sounddevice RawInputStream and handles starting/stopping and restart attempts.
    Allows specifying device (index or name).
    """
    def __init__(self, samplerate=16000, blocksize=2000, dtype='int16', channels=1, callback=None, device: Optional[Union[int, str]] = None, logger: Optional[ColorLogger] = None):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.dtype = dtype
        self.channels = channels
        self.callback = callback
        self._stream = None
        self._lock = threading.Lock()
        self.logger = logger or ColorLogger()
        self.device = device  # device index or name, passed to sounddevice

    def _wrap_callback(self, indata, frames, time_info, status):
        if status:
            self.logger.debug(f"Audio stream status: {status}")
        try:
            # If raw stream returns bytes, pass as-is; if numpy array, convert to bytes
            data_bytes = indata if isinstance(indata, (bytes, bytearray)) else bytes(indata)
        except Exception:
            # fallback: try to get buffer
            try:
                data_bytes = indata.tobytes()
            except Exception:
                self.logger.debug("Failed to convert audio input to bytes.", exc_info=True)
                return
        if self.callback:
            try:
                self.callback(data_bytes, frames, time_info, status)
            except Exception:
                self.logger.exception("Error in user audio callback.")

    def start(self):
        with self._lock:
            if self._stream:
                return
            try:
                self._stream = sd.RawInputStream(
                    samplerate=self.samplerate,
                    blocksize=self.blocksize,
                    dtype=self.dtype,
                    channels=self.channels,
                    callback=self._wrap_callback,
                    device=self.device
                )
                self._stream.start()
                self.logger.info(f"Audio stream started. device={self.device!r}")
            except Exception as e:
                self._stream = None
                self.logger.exception(f"Failed to start audio stream (device={self.device!r}): {e}")
                raise

    def stop(self):
        with self._lock:
            if not self._stream:
                return
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                self.logger.exception("Error while stopping audio stream.")
            finally:
                self._stream = None
                self.logger.info("Audio stream stopped.")

    def is_active(self):
        with self._lock:
            return self._stream is not None


# ----------------- SpeechRecognizer (wraps Vosk recognizers) -----------------
class SpeechRecognizer:
    """
    Wraps Vosk Model and multiple KaldiRecognizer instances for main/keyword/quick.
    Provides safe methods to accept waveform and get results.
    """
    def __init__(self, model_path: str, sample_rate: int = 16000, logger: Optional[ColorLogger] = None):
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.model = None
        self.logger = logger or ColorLogger()
        self._load_model()

        # recognizers (can be re-created if model changes)
        self.rec_main = self._make_recognizer()
        self.rec_kw = self._make_recognizer()
        self.rec_quick = self._make_recognizer()

    def _load_model(self):
        try:
            self.model = vosk.Model(self.model_path)
            self.logger.info(f"Vosk model loaded from {self.model_path}")
        except Exception as e:
            self.model = None
            self.logger.exception(f"Failed to load Vosk model from {self.model_path}: {e}")
            exit()

    def _make_recognizer(self):
        if self.model:
            try:
                return vosk.KaldiRecognizer(self.model, self.sample_rate)
            except Exception:
                self.logger.exception("Failed to create KaldiRecognizer.")
                return None
        return None

    def reload_model(self, new_model_path: Optional[str] = None):
        if new_model_path:
            self.model_path = new_model_path
        self._load_model()
        self.rec_main = self._make_recognizer()
        self.rec_kw = self._make_recognizer()
        self.rec_quick = self._make_recognizer()
        self.logger.info("Model reloaded.")

    def reset(self):
        try:
            if self.rec_main:
                self.rec_main.Reset()
            if self.rec_kw:
                self.rec_kw.Reset()
            if self.rec_quick:
                self.rec_quick.Reset()
        except Exception:
            self.logger.exception("Error while resetting recognizers.")

    # Helper accept/process methods: returns tuple(result_text, partial_text)
    def process_main(self, processed_bytes: bytes):
        text = ""
        partial = ""
        try:
            if self.rec_main and self.rec_main.AcceptWaveform(processed_bytes):
                result = json.loads(self.rec_main.Result())
                text = result.get("text", "")
            else:
                partial = json.loads(self.rec_main.PartialResult()).get("partial", "")
        except Exception:
            self.logger.debug("Error in main recognizer processing.", exc_info=True)
        return text, partial

    def process_kw(self, processed_bytes: bytes):
        # kw recognizer: mainly partials for keyword detection
        try:
            if self.rec_kw and self.rec_kw.AcceptWaveform(processed_bytes):
                _ = json.loads(self.rec_kw.Result())
            partial = json.loads(self.rec_kw.PartialResult()).get("partial", "")
            return "", partial
        except Exception:
            self.logger.debug("Error in keyword recognizer processing.", exc_info=True)
            return "", ""

    def process_quick(self, processed_bytes: bytes):
        try:
            if self.rec_quick and self.rec_quick.AcceptWaveform(processed_bytes):
                _ = json.loads(self.rec_quick.Result())
            partial = json.loads(self.rec_quick.PartialResult()).get("partial", "")
            return "", partial
        except Exception:
            self.logger.debug("Error in quick recognizer processing.", exc_info=True)
            return "", ""


# ----------------- RealtimeRecognizer (orchestrator) -----------------
class VoiceTrigger:
    def __init__(self, model_path,
                 sample_rate=16000, blocksize=2000,
                 keywords: Optional[List[str]] = None,
                 quick_words: Optional[List[str]] = None,
                 buffer_time_seconds=6, buffer_max_chars=1000,
                 noise_reduction=False, hp_cutoff=100,
                 batch_blocks=6,
                 voice_detector_buffer_seconds=5,
                 calibration_path: Optional[Path] = None,
                 device: Optional[Union[int, str]] = None,
                 logger: Optional[ColorLogger] = None):
        self.sample_rate = sample_rate
        self.blocksize = blocksize

        self.log = logger or ColorLogger(level="info")

        # speech recognizer wrapper
        self.speech = SpeechRecognizer(model_path=model_path, sample_rate=sample_rate, logger=self.log)

        # initial lists (unique)
        self.keywords = list(dict.fromkeys(keywords or []))
        self.quick_words = list(dict.fromkeys(quick_words or []))

        # flags
        self.active_main = False
        self.active_kw = False
        self.active_quick = False

        # buffer for dynamic text
        self.text_buffer = []
        self.buffer_active = False
        self.buffer_time_seconds = buffer_time_seconds
        self.buffer_max_chars = buffer_max_chars

        # voice detector
        self.voice_detector = VoiceLevelDetector(
            samplerate=sample_rate,
            blocksize=blocksize,
            buffer_seconds=voice_detector_buffer_seconds,
            calibration_path=calibration_path,
            logger=self.log
        )

        # audio manager (store device selection)
        self.device = device
        self.audio_manager = AudioStreamManager(
            samplerate=sample_rate,
            blocksize=blocksize,
            dtype='int16',
            channels=1,
            callback=self._audio_callback,
            device=self.device,
            logger=self.log
        )

        # async queue/loop/stream
        self.async_q = None
        self.loop = None

        self._silence_start_main = time.time()
        self._silence_start_kw = time.time()
        self.latest_text = ""

        # noise settings
        self.noise_reduction = noise_reduction and _NOISEREDUCE_AVAILABLE
        self.hp_cutoff = hp_cutoff
        self.noise_floor_db = -50.0

        # batch
        self.batch_blocks = max(1, int(batch_blocks))

        # callbacks
        self._text_handlers = []
        self._keyword_handlers = []
        self._quick_handlers = []
        self._silence_handlers = []
        self._kw_silence_handlers = []

        self._handlers_lock = threading.Lock()

        # internal state for last matched quick/keyword
        self._last_keyword = None
        self._last_quick = None

        # For test mode: external injection of raw audio bytes (bypass stream)
        self._test_mode = False

    # ----------- Public decorator API (keeps original API) -----------
    def text(self, flt: Optional[Filter] = None):
        if flt is None:
            flt = Filter(None)

        def decorator(func: Callable[[TextContext], Coroutine[Any, Any, Any]]):
            with self._handlers_lock:
                self._text_handlers.append((flt, func))
            return func
        return decorator

    def keyword(self, flt: Optional[Filter] = None):
        if flt is None:
            flt = Filter(None)

        def decorator(func: Callable[[TextContext], Coroutine[Any, Any, Any]]):
            with self._handlers_lock:
                if flt.phrases:
                    for p in flt.phrases:
                        if p not in self.keywords:
                            self.keywords.append(p)
                self._keyword_handlers.append((flt, func))
            return func
        return decorator

    def quick(self, flt: Optional[Filter] = None):
        if flt is None:
            flt = Filter(None)

        def decorator(func: Callable[[TextContext], Coroutine[Any, Any, Any]]):
            with self._handlers_lock:
                if flt.phrases:
                    for p in flt.phrases:
                        if p not in self.quick_words:
                            self.quick_words.append(p)
                self._quick_handlers.append((flt, func))
            return func
        return decorator

    def on_silence(self):
        def decorator(func: Callable[[float], Coroutine[Any, Any, Any]]):
            with self._handlers_lock:
                self._silence_handlers.append(func)
            return func
        return decorator

    def on_kw_silence(self):
        def decorator(func: Callable[[float], Coroutine[Any, Any, Any]]):
            with self._handlers_lock:
                self._kw_silence_handlers.append(func)
            return func
        return decorator

    # --------- control operations (start/stop/reload) ----------
    def start_recognition_main(self):
        try:
            self.speech.reset()
            self.active_main = True
            self.active_quick = True
            self.buffer_active = False
            self.log.debug("Main recognition enabled.")
        except Exception:
            self.log.exception("Failed to enable main recognition.")

    def stop_recognition_main(self):
        self.active_main = False
        self.active_quick = False
        self.log.debug("Main recognition disabled.")

    def start_recognition_keywords(self):
        try:
            self.speech.reset()
            self.active_kw = True
            self.log.debug("Keyword recognition enabled.")
        except Exception:
            self.log.exception("Failed to enable keyword recognition.")

    def stop_recognition_keywords(self):
        self.active_kw = False
        self.log.debug("Keyword recognition disabled.")

    def reload_model(self, model_path: Optional[str] = None):
        try:
            self.speech.reload_model(new_model_path=model_path)
        except Exception:
            self.log.exception("Failed to reload model.")

    # --------- device management ----------
    def set_input_device(self, device: Optional[Union[int, str]], restart_stream: bool = False):
        """
        Set input device (index or name). If restart_stream=True and stream is active,
        it will stop and restart stream with the new device.
        """
        self.log.info(f"Set input device to {device!r}")
        self.device = device
        self.audio_manager.device = device
        if restart_stream:
            was_active = self.audio_manager.is_active()
            try:
                if was_active:
                    self.log.info("Restarting audio stream to apply device change...")
                    self.audio_manager.stop()
                    # small delay to ensure OS releases device (non-blocking)
                    time.sleep(0.1)
                    self.audio_manager.start()
                    self.log.info("Audio stream restarted with new device.")
            except Exception:
                self.log.exception("Failed to restart audio stream after device change.")

    @staticmethod
    def list_input_devices() -> List[dict]:
        """
        Returns a list of available input devices (each item: {'index': int, 'name': str, 'max_input_channels': int})
        """
        out = []
        try:
            devs = sd.query_devices()
            for i, d in enumerate(devs):
                if d.get('max_input_channels', 0) > 0:
                    out.append({
                        'index': i,
                        'name': d.get('name'),
                        'max_input_channels': d.get('max_input_channels', 0)
                    })
        except Exception:
            # If query fails, return empty but log
            ColorLogger().exception("Failed to query sounddevice devices.")
        return out

    # --------- audio callback (from AudioStreamManager) ----------
    def _audio_callback(self, data_bytes, frames, time_info, status):
        try:
            # Feed voice detector and async queue
            try:
                self.voice_detector.process_block(data_bytes)
            except Exception:
                self.log.debug("Voice detector failed for a block.", exc_info=True)

            if self.loop and self.async_q:
                try:
                    # put bytes to async queue from audio thread
                    self.loop.call_soon_threadsafe(self.async_q.put_nowait, data_bytes)
                except Exception:
                    self.log.debug("Failed to put audio block into async queue.", exc_info=True)
        except Exception:
            self.log.exception("Unhandled error inside audio callback.")

    # --------- preprocessing ----------
    def _highpass_filter(self, data_float: np.ndarray):
        if _SCIPY_AVAILABLE:
            nyq = 0.5 * self.sample_rate
            normal_cutoff = self.hp_cutoff / nyq
            try:
                b, a = butter(1, normal_cutoff, btype='high', analog=False)
                return filtfilt(b, a, data_float)
            except Exception:
                self.log.debug("SciPy highpass failed; using fallback filter.", exc_info=True)
        alpha = 0.995
        y = np.zeros_like(data_float)
        prev_x = 0.0
        prev_y = 0.0
        for i, x in enumerate(data_float):
            y[i] = alpha * (prev_y + x - prev_x)
            prev_x = x
            prev_y = y[i]
        return y

    def _simple_noise_gate_and_normalize(self, data_float: np.ndarray):
        if data_float.size == 0:
            return data_float
        rms = np.sqrt(np.mean(data_float ** 2))
        db = 20 * np.log10(rms + 1e-12)
        try:
            if db < (self.noise_floor_db):
                self.noise_floor_db = 0.95 * self.noise_floor_db + 0.05 * db
            else:
                self.noise_floor_db = 0.999 * self.noise_floor_db + 0.001 * db
        except Exception:
            self.log.debug("Noise floor update failed.", exc_info=True)

        if db < (self.noise_floor_db + 6.0):
            data_float = data_float * (rms / (rms + 1e-6)) * 0.1
        peak = np.max(np.abs(data_float)) + 1e-12
        if peak > 0.95:
            data_float = data_float / peak * 0.95
        return data_float

    def _preprocess_audio(self, data_bytes: bytes) -> bytes:
        try:
            data_int16 = np.frombuffer(data_bytes, dtype=np.int16)
            data_float = data_int16.astype(np.float32) / 32768.0
            data_float = self._highpass_filter(data_float)
            if self.noise_reduction and _NOISEREDUCE_AVAILABLE:
                try:
                    nr_out = nr.reduce_noise(y=data_float, sr=self.sample_rate, stationary=False)
                    data_float = nr_out.astype(np.float32)
                except Exception:
                    self.log.debug("noisereduce failed, falling back to simple gate.", exc_info=True)
                    data_float = self._simple_noise_gate_and_normalize(data_float)
            else:
                data_float = self._simple_noise_gate_and_normalize(data_float)
            clipped = np.clip(data_float, -1.0, 1.0)
            out_int16 = (clipped * 32767.0).astype(np.int16)
            return out_int16.tobytes()
        except Exception:
            self.log.debug("Preprocessing failed for audio block; returning raw bytes.", exc_info=True)
            return data_bytes

    # --------- buffering text ----------
    def _append_to_buffer(self, text: str):
        ts = time.time()
        self.text_buffer.append((ts, text))
        cutoff = ts - self.buffer_time_seconds
        self.text_buffer = [(t, txt) for (t, txt) in self.text_buffer if t >= cutoff]
        total_chars = sum(len(txt) for (_, txt) in self.text_buffer)
        while total_chars > self.buffer_max_chars and len(self.text_buffer) > 1:
            self.text_buffer.pop(0)
            total_chars = sum(len(txt) for (_, txt) in self.text_buffer)

    def get_buffered_phrase(self):
        if not self.text_buffer:
            return None
        text_all = " ".join(txt for (_, txt) in self.text_buffer)
        for kw in self.keywords:
            idx = text_all.find(kw)
            if idx != -1:
                return text_all[idx:]
        return text_all

    # --------- General matching + dispatch utilities ----------
    def _match_filter_against_text(self, flt: Filter, text: str, voice_mode: str) -> Optional[str]:
        text_low = text.lower()
        if flt.is_wildcard():
            return None
        for pattern in flt.phrases:
            pat_low = pattern.lower()
            if pat_low in text_low:
                return pattern
            words = [w for w in ''.join(ch if ch.isalnum() else ' ' for ch in text_low).split() if w]
            for w in words:
                if is_match_by_lev(w, pat_low, flt.lv):
                    return pattern
            if is_match_by_lev(text_low, pat_low, flt.lv):
                return pattern
        return None

    async def _invoke_handler(self, handler: Callable[..., Any], arg):
        """
        Invoke a handler safely:
        - If it's an async function, schedule it on the event loop.
        - If it's synchronous, run in default executor to avoid blocking the loop.
        """
        try:
            if inspect.iscoroutinefunction(handler):
                # schedule coroutine
                asyncio.create_task(handler(arg))
            else:
                # run sync handler in executor to avoid blocking
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, handler, arg)
        except Exception:
            self.log.exception("Handler invocation failed.")

    async def _dispatch_handlers_generic(self, handlers, text_or_kw: str, voice_mode: str, is_phrase_source=False):
        """
        Generic dispatcher for text/keyword/quick handlers.
        If flt.is_wildcard() -> call with match=None or match=the kw (for keywords/quick).
        """
        ts = time.time()
        with self._handlers_lock:
            handlers_copy = list(handlers)
        for flt, handler in handlers_copy:
            try:
                if flt.mode is not None and voice_mode != flt.mode.value:
                    continue

                if flt.is_wildcard():
                    # wildcard means we accept everything
                    match = text_or_kw if is_phrase_source else None
                    ctx = TextContext(text=text_or_kw, mode=voice_mode, match=match, ts=ts)
                    await self._invoke_handler(handler, ctx)
                    continue

                match = self._match_filter_against_text(flt, text_or_kw, voice_mode)
                if match is not None:
                    ctx = TextContext(text=text_or_kw, mode=voice_mode, match=match, ts=ts)
                    await self._invoke_handler(handler, ctx)
            except Exception:
                self.log.exception("Error while dispatching to handler.")

    # --------- silence invocation helpers ----------
    async def _call_silence_handlers(self, silence_main: float):
        with self._handlers_lock:
            handlers = list(self._silence_handlers)
        for h in handlers:
            try:
                if inspect.iscoroutinefunction(h):
                    asyncio.create_task(h(silence_main))
                else:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, h, silence_main)
            except Exception:
                self.log.exception("Failure while calling silence handler.")

    async def _call_kw_silence_handlers(self, silence_kw: float):
        with self._handlers_lock:
            handlers = list(self._kw_silence_handlers)
        for h in handlers:
            try:
                if inspect.iscoroutinefunction(h):
                    asyncio.create_task(h(silence_kw))
                else:
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, h, silence_kw)
            except Exception:
                self.log.exception("Failure while calling keyword silence handler.")

    # --------- main processing loop ----------
    async def process_audio(self):
        """
        Pull from async queue, process audio in batches, run recognizers and dispatch handlers.
        Returns tuple similar to previous implementation.
        """
        main_text = None
        keyword_found = None
        quick_found = None

        if self.async_q is None:
            return None, None, None, time.time() - self._silence_start_main, time.time() - self._silence_start_kw

        while not self.async_q.empty():
            available = self.async_q.qsize()
            n = min(available, self.batch_blocks)
            if n <= 0:
                break
            frames = []
            for _ in range(n):
                try:
                    frames.append(self.async_q.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if not frames:
                break
            batch_bytes = b"".join(frames)
            processed = self._preprocess_audio(batch_bytes)

            # main recognition
            if self.active_main:
                try:
                    text, partial = self.speech.process_main(processed)
                    # Prefer final text over partials for buffer
                    if text and not (self._last_quick and is_match_by_lev(text, self._last_quick, 10)):
                        main_text = text
                        self.latest_text = text
                        self._append_to_buffer(text)
                        self._silence_start_main = time.time()
                        if self.buffer_active:
                            self.buffer_active = False
                    else:
                        # partial handling: keep buffer and latest_text updated
                        if partial:
                            self.latest_text = partial
                            self._append_to_buffer(partial)
                            self._silence_start_main = time.time()
                except Exception:
                    self.log.exception("Error while running main recognition.")

            # keywords
            if self.active_kw:
                try:
                    _, partial_kw = self.speech.process_kw(processed)
                    if partial_kw:
                        self._silence_start_kw = time.time()
                        for kw in self.keywords:
                            if is_match_by_lev(partial_kw, kw, 10) or kw.lower() in partial_kw.lower():
                                if kw != getattr(self, "_last_keyword", None):
                                    keyword_found = kw
                                    self._last_keyword = kw
                                    self.buffer_active = True
                                    self.text_buffer = [(time.time(), kw)]
                                break
                    else:
                        # reset last keyword if no partial matches
                        if not any(is_match_by_lev(partial_kw, kw, 10) or kw.lower() in partial_kw.lower() for kw in self.keywords):
                            self._last_keyword = None
                except Exception:
                    self.log.exception("Error while running keyword recognition.")

            # quick words
            if self.active_quick:
                try:
                    _, partial_q = self.speech.process_quick(processed)
                    if partial_q:
                        for qw in self.quick_words:
                            if is_match_by_lev(partial_q, qw, 10) or qw.lower() in partial_q.lower():
                                if qw != getattr(self, "_last_quick", None):
                                    quick_found = qw
                                    self._last_quick = qw
                                break
                    else:
                        if not any(is_match_by_lev(partial_q, qw, 10) or qw.lower() in partial_q.lower() for qw in self.quick_words):
                            self._last_quick = None
                except Exception:
                    self.log.exception("Error while running quick-word recognition.")

        silence_time_main = time.time() - self._silence_start_main
        silence_time_kw = time.time() - self._silence_start_kw

        voice_mode = self.voice_detector.get_dominant_level()

        # dispatching via unified generic dispatcher
        if main_text:
            await self._dispatch_handlers_generic(self._text_handlers, main_text, voice_mode, is_phrase_source=False)
        if keyword_found:
            await self._dispatch_handlers_generic(self._keyword_handlers, keyword_found, voice_mode, is_phrase_source=True)
        if quick_found:
            await self._dispatch_handlers_generic(self._quick_handlers, quick_found, voice_mode, is_phrase_source=True)

        # call silence handlers (non-blocking)
        await self._call_silence_handlers(silence_time_main)
        await self._call_kw_silence_handlers(silence_time_kw)

        return main_text, keyword_found, quick_found, silence_time_main, silence_time_kw

    # --------- run loop & streaming ----------
    async def start_stream(self):
        if self._test_mode:
            self.log.info("Test mode active: not starting audio stream.")
            return
        if self.async_q is None:
            self.loop = asyncio.get_running_loop()
            self.async_q = asyncio.Queue()
        try:
            self.audio_manager.start()
            self.log.info("Audio stream started (async).")
        except Exception:
            self.log.exception("Failed to start audio stream.")

    async def stop_stream(self):
        try:
            self.audio_manager.stop()
            if self.async_q:
                # clear queue
                while not self.async_q.empty():
                    try:
                        self.async_q.get_nowait()
                    except Exception:
                        break
            self.log.info("Audio stream stopped (async).")
        except Exception:
            self.log.exception("Error while stopping audio stream.")

    # Allow injecting raw audio blocks for testing (bypasses the actual microphone)
    async def inject_audio_block(self, data_bytes: bytes):
        if self.async_q is None:
            self.async_q = asyncio.Queue()
            self.loop = asyncio.get_running_loop()
        await self.async_q.put(data_bytes)

    async def run(self, initial_keywords_mode=True):
        """
        Main run loop. The class itself does not perform switching logic (left to handlers),
        but it supervises stream and processing.
        """
        await self.start_stream()
        if initial_keywords_mode:
            self.start_recognition_keywords()
        else:
            self.start_recognition_main()

        self.log.info("Say something... (Ctrl+C to exit)")

        stop_event = asyncio.Event()

        def _signal_handler(sig, frame):
            stop_event.set()

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        try:
            while not stop_event.is_set():
                await asyncio.sleep(0.05)
                # user logic (switching) must be in handlers; class does not switch itself
                await self.process_audio()
        except Exception:
            self.log.exception("Unexpected error in main loop.")
        finally:
            self.stop_recognition_main()
            self.stop_recognition_keywords()
            await self.stop_stream()
            self.log.info("Stopping speech recognition")
