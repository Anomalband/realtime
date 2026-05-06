import time
import json
import numpy as np
import resampy
import requests
from typing import Iterator
from pathlib import Path

from utils.logger import logger
from .base_tts import BaseTTS, State
from registry import register

@register("tts", "xtts")
class XTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt,parent)
        self.default_language = getattr(opt, "xtts_language", "en")
        self.default_stream_chunk_size = str(getattr(opt, "xtts_stream_chunk_size", 2))
        self.speaker = self.get_speaker(opt.REF_FILE, opt.TTS_SERVER)
        if getattr(opt, "xtts_prewarm", True):
            self._prewarm_stream()

    def txt_to_audio(self,msg:tuple[str, dict]):
        text,textevent = msg  
        tts_cfg = textevent.get("tts", {})
        language = tts_cfg.get("language", self.default_language)
        stream_chunk_size = str(tts_cfg.get("stream_chunk_size", self.default_stream_chunk_size))

        self.stream_tts(
            self.xtts(
                text,
                self.speaker,
                language,
                self.opt.TTS_SERVER, #"http://localhost:9000", #args.server_url,
                stream_chunk_size
            ),
            msg
        )

    def get_speaker(self,ref_audio,server_url):
        if not ref_audio:
            logger.warning("xtts REF_FILE is empty; attempting server default speaker.")
            return self._get_server_default_speaker(server_url)

        ref_path = Path(ref_audio)
        if ref_path.suffix.lower() == ".json" and ref_path.exists():
            try:
                with ref_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                if "gpt_cond_latent" in payload and "speaker_embedding" in payload:
                    logger.info("xtts loaded speaker embeddings from %s", ref_audio)
                    return payload
                logger.warning("xtts speaker json missing required keys: %s", ref_audio)
            except Exception:
                logger.exception("xtts failed to load speaker json: %s", ref_audio)

        if not ref_path.exists():
            logger.warning("xtts REF_FILE not found: %s; attempting server default speaker.", ref_audio)
            return self._get_server_default_speaker(server_url)

        with ref_path.open("rb") as f:
            files = {"wav_file": ("reference.wav", f)}
            response = requests.post(f"{server_url}/clone_speaker", files=files, timeout=30)
            response.raise_for_status()
            return response.json()

    def _get_server_default_speaker(self, server_url):
        try:
            response = requests.get(f"{server_url}/studio_speakers", timeout=10)
            response.raise_for_status()
            speakers = response.json() or {}
            if speakers:
                first_name = next(iter(speakers.keys()))
                logger.info("xtts using server default speaker: %s", first_name)
                return speakers[first_name]
        except Exception:
            logger.exception("xtts failed to fetch server default speaker.")
        return {}

    def _prewarm_stream(self):
        if not (self.speaker and "speaker_embedding" in self.speaker and "gpt_cond_latent" in self.speaker):
            logger.warning("xtts prewarm skipped: no speaker embeddings.")
            return

        rounds = max(1, int(getattr(self.opt, "xtts_prewarm_rounds", 2)))
        start = time.perf_counter()
        try:
            for i in range(rounds):
                round_start = time.perf_counter()
                payload = dict(self.speaker)
                payload["text"] = "ok"
                payload["language"] = self.default_language
                payload["stream_chunk_size"] = self.default_stream_chunk_size
                payload["add_wav_header"] = True
                with requests.post(
                    f"{self.opt.TTS_SERVER}/tts_stream",
                    json=payload,
                    stream=True,
                    timeout=30,
                ) as res:
                    res.raise_for_status()
                    for chunk in res.iter_content(chunk_size=None):
                        if chunk:
                            logger.info(
                                "xtts prewarm first chunk: round=%d/%d %.3fs",
                                i + 1,
                                rounds,
                                time.perf_counter() - round_start,
                            )
                            break
            logger.info("xtts prewarm done in %.3fs", time.perf_counter() - start)
        except Exception:
            logger.exception("xtts prewarm failed")

    def xtts(self,text, speaker, language, server_url, stream_chunk_size) -> Iterator[bytes]:
        start = time.perf_counter()
        if not (speaker and "speaker_embedding" in speaker and "gpt_cond_latent" in speaker):
            logger.error("xtts speaker embedding is missing; cannot call /tts_stream")
            return

        payload = dict(speaker or {})
        payload["text"] = text
        payload["language"] = language
        payload["stream_chunk_size"] = stream_chunk_size  # lower values reduce first-audio latency
        try:
            res = requests.post(
                f"{server_url}/tts_stream",
                json=payload,
                stream=True,
                timeout=60,
            )
            end = time.perf_counter()
            logger.info(f"xtts Time to make POST: {end-start}s")

            if res.status_code != 200:
                print("Error:", res.text)
                return

            first = True
        
            for chunk in res.iter_content(chunk_size=None): #24K*20ms*2
                if first:
                    end = time.perf_counter()
                    logger.info(f"xtts Time to first chunk: {end-start}s")
                    first = False
                if chunk:
                    yield chunk
        except Exception as e:
            print(e)
    
    def stream_tts(self,audio_stream,msg:tuple[str, dict]):
        text,textevent = msg
        first = True
        total_samples = 0
        last_stream = np.array([],dtype=np.float32)
        start = time.perf_counter()
        for chunk in audio_stream:
            if self.state != State.RUNNING:
                break
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                stream = np.concatenate((last_stream,stream))
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint={}
                    if first:
                        eventpoint={'status':'start','text':text}
                        first_audio_s = time.perf_counter() - start
                        logger.info("xtts first audio frame: %.3fs", first_audio_s)
                        metrics = getattr(self.parent, "runtime_metrics", None)
                        if isinstance(metrics, dict):
                            metrics["tts_ms"] = first_audio_s * 1000.0
                            query_start_ts = metrics.get("query_start_ts")
                            if isinstance(query_start_ts, (int, float)):
                                metrics["e2e_ms"] = (time.perf_counter() - query_start_ts) * 1000.0
                            metrics["updated_at"] = time.time()
                        first = False
                    eventpoint.update(**textevent) 
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    total_samples += self.chunk
                    streamlen -= self.chunk
                    idx += self.chunk
                last_stream = stream[idx:] #get the remain stream
        eventpoint={'status':'end','text':text}
        eventpoint.update(**textevent) 
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)  
        metrics = getattr(self.parent, "runtime_metrics", None)
        if isinstance(metrics, dict) and total_samples > 0:
            elapsed = time.perf_counter() - start
            audio_seconds = total_samples / float(self.sample_rate)
            if audio_seconds > 0:
                metrics["rtf"] = elapsed / audio_seconds
                metrics["updated_at"] = time.time()
