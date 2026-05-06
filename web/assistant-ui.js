let pc = null;
let sessionId = "";
let metricsTimer = null;

let mediaRecorder = null;
let recorderStream = null;
let recordedChunks = [];
let isRecording = false;

let fpsCurrent = null;
let fpsCounter = 0;
let fpsLastTs = 0;
let fpsLoopStarted = false;

const videoEl = document.getElementById("avatar-video");
const audioEl = document.getElementById("avatar-audio");
const talkBtn = document.getElementById("talk-btn");
const answerBox = document.getElementById("answer-box");
const statusLine = document.getElementById("status-line");
const chipOnline = document.getElementById("chip-online");

const metricEls = {
    fps: document.getElementById("metric-fps"),
    llm: document.getElementById("metric-llm"),
    tts: document.getElementById("metric-tts"),
    rtf: document.getElementById("metric-rtf"),
    e2e: document.getElementById("metric-e2e"),
};

function setStatus(text) {
    statusLine.textContent = text;
}

function setOnline(isOnline) {
    chipOnline.style.opacity = isOnline ? "1" : "0.6";
    chipOnline.textContent = isOnline ? "ONLINE" : "OFFLINE";
}

function setMetric(name, value) {
    if (!metricEls[name]) return;
    metricEls[name].textContent = value;
}

function formatMs(v) {
    if (typeof v !== "number" || Number.isNaN(v)) return "--";
    return `${Math.round(v)}ms`;
}

function startFpsLoop() {
    if (fpsLoopStarted) return;
    if (!("requestVideoFrameCallback" in HTMLVideoElement.prototype)) return;
    fpsLoopStarted = true;
    fpsLastTs = performance.now();

    const tick = (now) => {
        fpsCounter += 1;
        const delta = now - fpsLastTs;
        if (delta >= 1000) {
            fpsCurrent = Math.round((fpsCounter * 1000) / delta);
            setMetric("fps", String(fpsCurrent));
            fpsCounter = 0;
            fpsLastTs = now;
        }
        videoEl.requestVideoFrameCallback(tick);
    };

    videoEl.requestVideoFrameCallback(tick);
}

async function waitIceComplete() {
    if (!pc) return;
    if (pc.iceGatheringState === "complete") return;
    await new Promise((resolve) => {
        const check = () => {
            if (pc.iceGatheringState === "complete") {
                pc.removeEventListener("icegatheringstatechange", check);
                resolve();
            }
        };
        pc.addEventListener("icegatheringstatechange", check);
    });
}

async function connectWebRTC() {
    setStatus("Baglanti kuruluyor...");
    setOnline(false);

    pc = new RTCPeerConnection({ sdpSemantics: "unified-plan" });
    pc.addTransceiver("video", { direction: "recvonly" });
    pc.addTransceiver("audio", { direction: "recvonly" });

    pc.addEventListener("track", (evt) => {
        if (evt.track.kind === "video") {
            videoEl.srcObject = evt.streams[0];
            startFpsLoop();
        } else {
            audioEl.srcObject = evt.streams[0];
        }
    });

    pc.addEventListener("connectionstatechange", () => {
        const state = pc.connectionState;
        if (state === "connected") {
            setOnline(true);
            setStatus("Baglandi. Konus butonuna bas.");
        } else if (state === "connecting") {
            setOnline(false);
            setStatus("Baglanti suruyor...");
        } else if (state === "failed" || state === "closed" || state === "disconnected") {
            setOnline(false);
            setStatus("Baglanti koptu.");
        }
    });

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await waitIceComplete();

    const resp = await fetch("/offer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            sdp: pc.localDescription.sdp,
            type: pc.localDescription.type,
        }),
    });
    if (!resp.ok) {
        throw new Error(`offer failed: ${resp.status}`);
    }
    const answer = await resp.json();
    sessionId = answer.sessionid;
    await pc.setRemoteDescription(answer);

    if (metricsTimer) clearInterval(metricsTimer);
    metricsTimer = setInterval(pollMetrics, 500);
}

async function pollMetrics() {
    if (!sessionId) return;
    try {
        const resp = await fetch(`/metrics?sessionid=${encodeURIComponent(sessionId)}`);
        if (!resp.ok) return;
        const payload = await resp.json();
        const m = payload.data || {};

        setMetric("llm", formatMs(m.llm_ms));
        setMetric("tts", formatMs(m.tts_ms));
        setMetric("e2e", formatMs(m.e2e_ms));
        setMetric("rtf", typeof m.rtf === "number" ? m.rtf.toFixed(2) : "--");
        if (fpsCurrent === null) setMetric("fps", "--");

        if (m.last_answer && String(m.last_answer).trim()) {
            answerBox.textContent = String(m.last_answer).trim();
        }
    } catch (err) {
        setStatus(`Metrik okunamadi: ${err.message}`);
    }
}

function pickRecorderMime() {
    const candidates = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/ogg;codecs=opus",
    ];
    for (const c of candidates) {
        if (MediaRecorder.isTypeSupported(c)) return c;
    }
    return "";
}

async function startRecording() {
    if (!sessionId) {
        setStatus("Once baglanti beklenmeli.");
        return;
    }

    recorderStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = pickRecorderMime();
    mediaRecorder = mimeType ? new MediaRecorder(recorderStream, { mimeType }) : new MediaRecorder(recorderStream);
    recordedChunks = [];

    mediaRecorder.ondataavailable = (evt) => {
        if (evt.data && evt.data.size > 0) {
            recordedChunks.push(evt.data);
        }
    };
    mediaRecorder.onstop = async () => {
        try {
            const audioBlob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || "audio/webm" });
            await sendAudioForChat(audioBlob);
        } catch (err) {
            setStatus(`Ses gonderilemedi: ${err.message}`);
        } finally {
            if (recorderStream) {
                recorderStream.getTracks().forEach((t) => t.stop());
            }
            recorderStream = null;
            mediaRecorder = null;
            recordedChunks = [];
        }
    };

    mediaRecorder.start();
    isRecording = true;
    talkBtn.classList.add("recording");
    talkBtn.textContent = "Birak";
    setStatus("Dinleniyor... tekrar basinca gonderecek.");
}

function stopRecording() {
    if (!mediaRecorder || mediaRecorder.state === "inactive") return;
    mediaRecorder.stop();
    isRecording = false;
    talkBtn.classList.remove("recording");
    talkBtn.textContent = "Konus";
    setStatus("Ses gonderiliyor...");
}

async function sendAudioForChat(audioBlob) {
    const form = new FormData();
    form.append("file", audioBlob, "speech.webm");
    form.append("sessionid", String(sessionId));
    form.append("type", "chat");
    form.append("language", "en");
    form.append("tts", JSON.stringify({ language: "en", stream_chunk_size: 2 }));

    const resp = await fetch("/humanaudio", {
        method: "POST",
        body: form,
    });
    if (!resp.ok) {
        throw new Error(`humanaudio failed: ${resp.status}`);
    }
    setStatus("Istek alindi. Asistan cevap hazirliyor...");
}

async function toggleRecord() {
    try {
        if (!isRecording) {
            await startRecording();
        } else {
            stopRecording();
        }
    } catch (err) {
        isRecording = false;
        talkBtn.classList.remove("recording");
        talkBtn.textContent = "Konus";
        setStatus(`Mikrofon acilamadi: ${err.message}`);
    }
}

window.addEventListener("beforeunload", () => {
    if (metricsTimer) clearInterval(metricsTimer);
    if (pc) {
        try {
            pc.close();
        } catch (_err) {
            // ignore
        }
    }
});

talkBtn.addEventListener("click", toggleRecord);

window.addEventListener("DOMContentLoaded", async () => {
    setMetric("fps", "--");
    setMetric("llm", "--");
    setMetric("tts", "--");
    setMetric("rtf", "--");
    setMetric("e2e", "--");
    try {
        await connectWebRTC();
    } catch (err) {
        setStatus(`Baslangic hatasi: ${err.message}`);
        setOnline(false);
    }
});
