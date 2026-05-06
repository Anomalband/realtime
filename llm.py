import os
import time
from typing import TYPE_CHECKING

from utils.logger import logger

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar


def _ensure_runtime_metrics(avatar_session: "BaseAvatar") -> dict:
    metrics = getattr(avatar_session, "runtime_metrics", None)
    if not isinstance(metrics, dict):
        metrics = {}
        avatar_session.runtime_metrics = metrics
    return metrics


def _emit_chunk_to_tts(avatar_session: "BaseAvatar", datainfo: dict, chunk: str):
    if chunk:
        logger.info("llm chunk: %s", chunk)
        metrics = _ensure_runtime_metrics(avatar_session)
        if metrics.get("llm_ms") is None:
            query_start_ts = metrics.get("query_start_ts")
            if isinstance(query_start_ts, (int, float)):
                metrics["llm_ms"] = (time.perf_counter() - query_start_ts) * 1000.0
        prev = str(metrics.get("last_answer", "")).strip()
        metrics["last_answer"] = (f"{prev} {chunk}".strip() if prev else chunk.strip())
        metrics["updated_at"] = time.time()
        avatar_session.put_msg_txt(chunk, datainfo)


def _dashscope_stream_llm(message: str, avatar_session: "BaseAvatar", datainfo: dict):
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model=os.getenv("DASHSCOPE_MODEL", "qwen-plus"),
        messages=[
            {
                "role": "system",
                "content": "You are a concise assistant. Reply naturally and keep it short.",
            },
            {"role": "user", "content": message},
        ],
        stream=True,
        stream_options={"include_usage": True},
    )

    result = ""
    first_token = True
    first_chunk = True
    start = time.perf_counter()
    punct = ",.!?;:\n"
    metrics = _ensure_runtime_metrics(avatar_session)

    for chunk in completion:
        if len(chunk.choices) == 0:
            continue
        msg = chunk.choices[0].delta.content
        if not msg:
            continue
        if first_token:
            first_token = False
            logger.info("dashscope llm first token: %.3fs", time.perf_counter() - start)
        result += msg

        while True:
            cut = -1
            for i, ch in enumerate(result):
                if ch in punct and i + 1 >= 10:
                    cut = i + 1
                    break
            if cut <= 0:
                break
            chunk_text = result[:cut].strip()
            result = result[cut:]
            _emit_chunk_to_tts(avatar_session, datainfo, chunk_text)
            if first_chunk:
                first_chunk = False
                metrics["llm_ms"] = (time.perf_counter() - start) * 1000.0

    if result.strip():
        _emit_chunk_to_tts(avatar_session, datainfo, result.strip())
        if first_chunk:
            metrics["llm_ms"] = (time.perf_counter() - start) * 1000.0
    metrics["updated_at"] = time.time()


def llm_response(message, avatar_session: "BaseAvatar", datainfo: dict | None = None):
    try:
        if datainfo is None:
            datainfo = {}

        metrics = _ensure_runtime_metrics(avatar_session)
        metrics["query_start_ts"] = time.perf_counter()
        metrics["last_answer"] = ""
        metrics["asr_ms"] = None
        metrics["llm_ms"] = None
        metrics["tts_ms"] = None
        metrics["e2e_ms"] = None
        metrics["rtf"] = None
        metrics["updated_at"] = time.time()

        opt = avatar_session.opt
        backend = getattr(opt, "llm_backend", "local_qwen")
        if backend == "dashscope":
            _dashscope_stream_llm(message, avatar_session, datainfo)
            return

        from pipeline.speech_pipeline import LowLatencySpeechPipeline

        pipeline = LowLatencySpeechPipeline.get(opt)
        timings = pipeline.stream_llm(
            message,
            lambda chunk: _emit_chunk_to_tts(avatar_session, datainfo, chunk),
        )
        logger.info(
            "pipeline llm timings: first_token=%.3fs first_chunk=%.3fs",
            timings.llm_ttf_token_seconds,
            timings.llm_ttf_chunk_seconds,
        )
        metrics["llm_ms"] = timings.llm_ttf_chunk_seconds * 1000.0
        metrics["updated_at"] = time.time()
    except Exception:
        logger.exception("llm exception:")


def llm_response_from_audio(
    audio_bytes: bytes,
    avatar_session: "BaseAvatar",
    datainfo: dict | None = None,
    language: str | None = None,
):
    try:
        if datainfo is None:
            datainfo = {}

        metrics = _ensure_runtime_metrics(avatar_session)
        metrics["query_start_ts"] = time.perf_counter()
        metrics["last_answer"] = ""
        metrics["asr_ms"] = None
        metrics["llm_ms"] = None
        metrics["tts_ms"] = None
        metrics["e2e_ms"] = None
        metrics["rtf"] = None
        metrics["updated_at"] = time.time()

        from pipeline.speech_pipeline import LowLatencySpeechPipeline

        opt = avatar_session.opt
        pipeline = LowLatencySpeechPipeline.get(opt)
        timings = pipeline.asr_then_stream_llm(
            audio_bytes,
            lambda chunk: _emit_chunk_to_tts(avatar_session, datainfo, chunk),
            language=language,
        )
        logger.info(
            "pipeline timings: asr=%.3fs first_token=%.3fs first_chunk=%.3fs",
            timings.asr_seconds,
            timings.llm_ttf_token_seconds,
            timings.llm_ttf_chunk_seconds,
        )
        metrics["asr_ms"] = timings.asr_seconds * 1000.0
        metrics["llm_ms"] = timings.llm_ttf_chunk_seconds * 1000.0
        metrics["updated_at"] = time.time()
    except Exception:
        logger.exception("llm_response_from_audio exception:")
