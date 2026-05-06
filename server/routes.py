###############################################################################
#  Server routes
###############################################################################

import asyncio
import json
import os

from aiohttp import web

from llm import llm_response_from_audio
from server.session_manager import session_manager
from utils.logger import logger


def json_ok(data=None):
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.Response(content_type="application/json", text=json.dumps(body))


def json_error(msg: str, code: int = -1):
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": code, "msg": str(msg)}),
    )


def get_session(request, sessionid: str):
    return session_manager.get_session(sessionid)


async def assistant_ui(request):
    ui_path = os.path.join("web", "assistant-ui.html")
    return web.FileResponse(ui_path)


async def human(request):
    """Text input (echo/chat)"""
    try:
        params: dict = await request.json()

        sessionid: str = params.get("sessionid", "")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        if params.get("interrupt"):
            avatar_session.flush_talk()

        datainfo = {}
        if params.get("tts"):
            datainfo["tts"] = params.get("tts")

        if params["type"] == "echo":
            avatar_session.put_msg_txt(params["text"], datainfo)
        elif params["type"] == "chat":
            llm_response = request.app.get("llm_response")
            if llm_response:
                asyncio.get_event_loop().run_in_executor(
                    None, llm_response, params["text"], avatar_session, datainfo
                )

        return json_ok()
    except Exception as e:
        logger.exception("human route exception:")
        return json_error(str(e))


async def interrupt_talk(request):
    try:
        params = await request.json()
        sessionid = params.get("sessionid", "")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.flush_talk()
        return json_ok()
    except Exception as e:
        logger.exception("interrupt_talk exception:")
        return json_error(str(e))


async def humanaudio(request):
    """Upload audio file.

    type=echo: playback audio directly.
    type=chat: run ASR(whisper) -> LLM(qwen2.5) -> TTS streaming.
    """
    try:
        form = await request.post()
        sessionid = str(form.get("sessionid", ""))
        fileobj = form["file"]
        filebytes = fileobj.file.read()

        datainfo = {}
        if form.get("tts"):
            try:
                datainfo["tts"] = json.loads(form.get("tts"))
            except Exception:
                datainfo["tts"] = form.get("tts")

        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")

        req_type = str(form.get("type", "echo")).lower()
        if req_type == "chat":
            language = str(form.get("language", "en")).strip() or "en"
            asyncio.get_event_loop().run_in_executor(
                None,
                llm_response_from_audio,
                filebytes,
                avatar_session,
                datainfo,
                language,
            )
        else:
            avatar_session.put_audio_file(filebytes, datainfo)

        return json_ok()
    except Exception as e:
        logger.exception("humanaudio exception:")
        return json_error(str(e))


async def set_audiotype(request):
    try:
        params = await request.json()
        sessionid = params.get("sessionid", "")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.set_custom_state(params["audiotype"])
        return json_ok()
    except Exception as e:
        logger.exception("set_audiotype exception:")
        return json_error(str(e))


async def record(request):
    try:
        params = await request.json()
        sessionid = params.get("sessionid", "")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        if params["type"] == "start_record":
            avatar_session.start_recording()
        elif params["type"] == "end_record":
            avatar_session.stop_recording()
        return json_ok()
    except Exception as e:
        logger.exception("record exception:")
        return json_error(str(e))


async def is_speaking(request):
    params = await request.json()
    sessionid = params.get("sessionid", "")
    avatar_session = get_session(request, sessionid)
    if avatar_session is None:
        return json_error("session not found")
    return json_ok(data=avatar_session.is_speaking())


async def metrics(request):
    sessionid = str(request.query.get("sessionid", "")).strip()
    if not sessionid:
        return json_error("sessionid required")
    avatar_session = get_session(request, sessionid)
    if avatar_session is None:
        return json_error("session not found")
    payload = getattr(avatar_session, "runtime_metrics", {})
    if not isinstance(payload, dict):
        payload = {}
    return json_ok(data=payload)


def setup_routes(app):
    app.router.add_get("/", assistant_ui)
    app.router.add_get("/assistant-ui", assistant_ui)
    app.router.add_get("/metrics", metrics)
    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/is_speaking", is_speaking)
    app.router.add_static("/", path="web")
