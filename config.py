###############################################################################
# Config parsing
###############################################################################

import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="LiveTalking Digital Human Server")

    # Audio/video timing
    parser.add_argument("--fps", type=int, default=25, help="video fps")
    parser.add_argument("-l", type=int, default=10)
    parser.add_argument("-m", type=int, default=8)
    parser.add_argument("-r", type=int, default=10)

    # Avatar model
    parser.add_argument("--model", type=str, default="wav2lip", help="musetalk/wav2lip/ultralight")
    parser.add_argument("--avatar_id", type=str, default="wav2lip256_avatar1", help="avatar id in data/avatars")
    parser.add_argument("--batch_size", type=int, default=16, help="infer batch")
    parser.add_argument("--modelres", type=int, default=192)
    parser.add_argument("--modelfile", type=str, default="")

    # Custom choreography
    parser.add_argument("--customvideo_config", type=str, default="", help="custom action json")

    # TTS
    parser.add_argument(
        "--tts",
        type=str,
        default="edgetts",
        help="edgetts/gpt-sovits/cosyvoice/fishtts/tencent/doubao/indextts2/azuretts/qwentts/xtts",
    )
    parser.add_argument("--REF_FILE", type=str, default="zh-CN-YunxiaNeural", help="reference file or voice id")
    parser.add_argument("--REF_TEXT", type=str, default=None)
    parser.add_argument("--TTS_SERVER", type=str, default="http://127.0.0.1:9880")
    parser.add_argument("--xtts_language", type=str, default="en", help="xtts language, e.g. en")
    parser.add_argument("--xtts_stream_chunk_size", type=int, default=2, help="xtts stream chunk size for lower latency")
    parser.add_argument("--xtts_prewarm", action="store_true", default=True, help="prewarm xtts stream endpoint at startup")
    parser.add_argument("--no_xtts_prewarm", dest="xtts_prewarm", action="store_false", help="disable xtts prewarm")
    parser.add_argument("--xtts_prewarm_rounds", type=int, default=2, help="number of xtts prewarm rounds")

    # Transport
    parser.add_argument("--transport", type=str, default="webrtc", help="rtcpush/webrtc/rtmp/virtualcam")
    parser.add_argument(
        "--push_url",
        type=str,
        default="http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream",
    )
    parser.add_argument("--max_session", type=int, default=1)
    parser.add_argument("--listenport", type=int, default=8010, help="web listen port")

    # ASR (Whisper)
    parser.add_argument("--asr_whisper_model", type=str, default="medium", help="faster-whisper model size")
    parser.add_argument("--asr_language", type=str, default="en", help="asr language")
    parser.add_argument("--asr_device", type=str, default="cuda", help="cuda/cpu")
    parser.add_argument("--asr_compute_type", type=str, default="int8_float16", help="int8_float16/int8/float16")
    parser.add_argument("--asr_vad_filter", action="store_true", default=True, help="enable whisper VAD filter")
    parser.add_argument("--asr_no_vad_filter", dest="asr_vad_filter", action="store_false", help="disable whisper VAD filter")

    # LLM (Qwen2.5)
    parser.add_argument("--llm_backend", type=str, default="local_qwen", help="local_qwen/dashscope")
    parser.add_argument("--llm_model_id", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--llm_quant_bits", type=int, default=4, help="4/8/16")
    parser.add_argument("--llm_max_new_tokens", type=int, default=128)
    parser.add_argument("--llm_temperature", type=float, default=0.3)
    parser.add_argument("--llm_top_p", type=float, default=0.9)
    parser.add_argument("--llm_repetition_penalty", type=float, default=1.05)
    parser.add_argument(
        "--llm_system_prompt",
        type=str,
        default="You are a concise, natural English speaking assistant. Keep answers short.",
    )

    # LLM->TTS chunking for low first-audio latency
    parser.add_argument("--llm_chunk_min_chars", type=int, default=2)
    parser.add_argument("--llm_chunk_max_chars", type=int, default=16)
    parser.add_argument("--llm_chunk_max_wait_ms", type=int, default=40)

    # Pipeline prewarm
    parser.add_argument("--pipeline_prewarm", action="store_true", default=True, help="preload ASR+LLM at startup")
    parser.add_argument("--no_pipeline_prewarm", dest="pipeline_prewarm", action="store_false", help="disable pipeline prewarm")
    parser.add_argument("--llm_generation_prewarm", action="store_true", default=True, help="run one dummy LLM generation at startup")
    parser.add_argument("--no_llm_generation_prewarm", dest="llm_generation_prewarm", action="store_false", help="disable LLM generation prewarm")

    opt = parser.parse_args()

    opt.customopt = []
    if opt.customvideo_config:
        with open(opt.customvideo_config, "r", encoding="utf-8") as f:
            opt.customopt = json.load(f)

    return opt
