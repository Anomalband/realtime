import io
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

from utils.logger import logger


@dataclass
class PipelineTimings:
    asr_seconds: float = 0.0
    llm_ttf_token_seconds: float = 0.0
    llm_ttf_chunk_seconds: float = 0.0


class WhisperASR:
    def __init__(self, opt):
        self.model_name = getattr(opt, "asr_whisper_model", "medium")
        self.language = getattr(opt, "asr_language", "en")
        wanted_device = getattr(opt, "asr_device", "cuda" if torch.cuda.is_available() else "cpu")
        if wanted_device == "cuda" and not torch.cuda.is_available():
            logger.warning("ASR requested cuda but cuda is not available. Falling back to cpu.")
            wanted_device = "cpu"
        self.device = wanted_device
        self.compute_type = getattr(
            opt,
            "asr_compute_type",
            "int8_float16" if self.device == "cuda" else "int8",
        )
        if self.device != "cuda" and self.compute_type == "int8_float16":
            self.compute_type = "int8"
        self.vad_filter = bool(getattr(opt, "asr_vad_filter", True))
        self._model = None
        self._lock = threading.Lock()

    def ensure_loaded(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from faster_whisper import WhisperModel

            t0 = time.perf_counter()
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            logger.info(
                "ASR ready: model=%s device=%s compute_type=%s load=%.2fs",
                self.model_name,
                self.device,
                self.compute_type,
                time.perf_counter() - t0,
            )

    def transcribe_bytes(self, audio_bytes: bytes, language: str | None = None) -> tuple[str, float]:
        self.ensure_loaded()
        lang = language or self.language
        t0 = time.perf_counter()
        segments, info = self._model.transcribe(
            io.BytesIO(audio_bytes),
            language=lang,
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            vad_filter=self.vad_filter,
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments if seg.text and seg.text.strip()).strip()
        took = time.perf_counter() - t0
        logger.info(
            "ASR done: lang=%s prob=%.3f dur=%.2fs rt=%.3fs text_len=%d",
            info.language,
            info.language_probability,
            info.duration,
            took,
            len(text),
        )
        return text, took


class Qwen25QuantizedLLM:
    def __init__(self, opt):
        self.model_id = getattr(opt, "llm_model_id", "Qwen/Qwen2.5-3B-Instruct")
        self.max_new_tokens = int(getattr(opt, "llm_max_new_tokens", 128))
        self.temperature = float(getattr(opt, "llm_temperature", 0.4))
        self.top_p = float(getattr(opt, "llm_top_p", 0.9))
        self.repetition_penalty = float(getattr(opt, "llm_repetition_penalty", 1.05))
        self.system_prompt = getattr(
            opt,
            "llm_system_prompt",
            "You are a concise, natural English speaking assistant. Keep answers short.",
        )
        self.quant_bits = int(getattr(opt, "llm_quant_bits", 4))
        self._tokenizer = None
        self._model = None
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()

    def ensure_loaded(self):
        if self._model is not None and self._tokenizer is not None:
            return
        with self._load_lock:
            if self._model is not None and self._tokenizer is not None:
                return

            t0 = time.perf_counter()
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

            model_kwargs = {
                "device_map": "auto",
                "trust_remote_code": True,
            }
            if self.quant_bits == 4:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["quantization_config"] = bnb_config
                model_kwargs["torch_dtype"] = torch.float16
            elif self.quant_bits == 8:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs["quantization_config"] = bnb_config
            else:
                model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

            try:
                self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
            except Exception:
                logger.exception(
                    "LLM quantized load failed for %s. Falling back to non-quantized load.",
                    self.model_id,
                )
                fallback_kwargs = {
                    "device_map": "auto",
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                }
                self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **fallback_kwargs)
            self._model.eval()
            logger.info(
                "LLM ready: model=%s quant=%dbit load=%.2fs",
                self.model_id,
                self.quant_bits,
                time.perf_counter() - t0,
            )

    def stream_text(self, user_text: str) -> tuple[Iterator[str], Callable[[], float]]:
        self.ensure_loaded()

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(input_ids, "to"):
            input_ids = input_ids.to(self._model.device)

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            "input_ids": input_ids,
            "streamer": streamer,
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "eos_token_id": self._tokenizer.eos_token_id,
        }

        first_token_time = {"value": 0.0}
        start = time.perf_counter()

        def run_generate():
            with self._generate_lock:
                self._model.generate(**generation_kwargs)

        gen_thread = threading.Thread(target=run_generate, daemon=True)
        gen_thread.start()

        def iterator():
            first = True
            for piece in streamer:
                if first:
                    first = False
                    first_token_time["value"] = time.perf_counter() - start
                    logger.info("LLM first token: %.3fs", first_token_time["value"])
                yield piece
            gen_thread.join()

        def first_token_elapsed() -> float:
            return first_token_time["value"]

        return iterator(), first_token_elapsed


class LowLatencySpeechPipeline:
    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self, opt):
        self.opt = opt
        self.asr = WhisperASR(opt)
        self.llm = Qwen25QuantizedLLM(opt)
        self.chunk_min_chars = int(getattr(opt, "llm_chunk_min_chars", 2))
        self.chunk_max_chars = int(getattr(opt, "llm_chunk_max_chars", 16))
        self.chunk_max_wait_ms = int(getattr(opt, "llm_chunk_max_wait_ms", 40))
        self.llm_generation_prewarm = bool(getattr(opt, "llm_generation_prewarm", True))

    @classmethod
    def get(cls, opt):
        if cls._instance is not None:
            return cls._instance
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(opt)
            return cls._instance

    def prewarm(self):
        try:
            self.asr.ensure_loaded()
            self.llm.ensure_loaded()
            if self.llm_generation_prewarm:
                t = self.stream_llm("Hi.", lambda _: None)
                logger.info(
                    "LLM generation prewarm done: first_token=%.3fs first_chunk=%.3fs",
                    t.llm_ttf_token_seconds,
                    t.llm_ttf_chunk_seconds,
                )
            logger.info("Speech pipeline prewarmed.")
        except Exception:
            logger.exception("Speech pipeline prewarm failed")

    def asr_then_stream_llm(self, audio_bytes: bytes, on_chunk: Callable[[str], None], language: str | None = None) -> PipelineTimings:
        text, asr_seconds = self.asr.transcribe_bytes(audio_bytes, language=language)
        if not text:
            logger.warning("ASR returned empty text")
            return PipelineTimings(asr_seconds=asr_seconds)
        return self.stream_llm(text, on_chunk, asr_seconds=asr_seconds)

    def stream_llm(self, user_text: str, on_chunk: Callable[[str], None], asr_seconds: float = 0.0) -> PipelineTimings:
        stream, first_token = self.llm.stream_text(user_text)
        timings = PipelineTimings(asr_seconds=asr_seconds)
        t0 = time.perf_counter()
        buffer = ""
        first_chunk_sent = False
        last_emit = t0
        punctuations = {".", "!", "?", ";", ":", ",", "\n", "。", "！", "？", "；", "：", "，"}

        for piece in stream:
            if not piece:
                continue
            buffer += piece

            while True:
                split_idx = -1
                for i, ch in enumerate(buffer):
                    if ch in punctuations and i + 1 >= self.chunk_min_chars:
                        split_idx = i + 1
                        break
                if split_idx <= 0:
                    break
                chunk = buffer[:split_idx].strip()
                buffer = buffer[split_idx:]
                if chunk:
                    on_chunk(chunk)
                    if not first_chunk_sent:
                        first_chunk_sent = True
                        timings.llm_ttf_chunk_seconds = time.perf_counter() - t0
                        logger.info("LLM first chunk: %.3fs", timings.llm_ttf_chunk_seconds)
                    last_emit = time.perf_counter()

            now = time.perf_counter()
            wait_ms = (now - last_emit) * 1000.0
            if len(buffer) >= self.chunk_max_chars or (
                len(buffer) >= self.chunk_min_chars and wait_ms >= self.chunk_max_wait_ms
            ):
                cut = buffer.rfind(" ")
                if cut <= 0:
                    cut = min(len(buffer), self.chunk_max_chars)
                chunk = buffer[:cut].strip()
                buffer = buffer[cut:]
                if chunk:
                    on_chunk(chunk)
                    if not first_chunk_sent:
                        first_chunk_sent = True
                        timings.llm_ttf_chunk_seconds = time.perf_counter() - t0
                        logger.info("LLM first chunk: %.3fs", timings.llm_ttf_chunk_seconds)
                    last_emit = time.perf_counter()

        if buffer.strip():
            on_chunk(buffer.strip())
            if not first_chunk_sent:
                timings.llm_ttf_chunk_seconds = time.perf_counter() - t0

        timings.llm_ttf_token_seconds = first_token()
        return timings
