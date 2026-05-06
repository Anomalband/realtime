$ErrorActionPreference = "Stop"

$root = "C:\yazilim"
$liveTalkingDir = Join-Path $root "livetalking"
$xttsServerDir = Join-Path $root "xtts-streaming-server\server"

$nerfPython = "C:\Users\adbot\AppData\Local\miniforge3\envs\nerfstream\python.exe"
$xttsPython = "C:\Users\adbot\AppData\Local\miniforge3\envs\xttsserver\python.exe"

$xttsPort = 9880
$xttsUrl = "http://127.0.0.1:$xttsPort"
$xttsOut = Join-Path $root "xttsserver.log"
$xttsErr = Join-Path $root "xttsserver.err.log"

function Test-XttsReady {
    try {
        $r = Invoke-RestMethod -Uri "$xttsUrl/languages" -TimeoutSec 3
        return $true
    } catch {
        return $false
    }
}

Write-Host "Checking XTTS server on $xttsUrl ..."
$conn = Get-NetTCPConnection -LocalPort $xttsPort -State Listen -ErrorAction SilentlyContinue
if (-not $conn) {
    Write-Host "Starting XTTS streaming server ..."
    if (Test-Path $xttsOut) { Remove-Item $xttsOut -Force }
    if (Test-Path $xttsErr) { Remove-Item $xttsErr -Force }

    $env:COQUI_TOS_AGREED = "1"
    $env:XTTS_USE_DEEPSPEED = "0"
    Start-Process -WindowStyle Hidden -FilePath $xttsPython `
        -ArgumentList @("-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "$xttsPort") `
        -WorkingDirectory $xttsServerDir `
        -RedirectStandardOutput $xttsOut `
        -RedirectStandardError $xttsErr | Out-Null
}

Write-Host "Waiting XTTS readiness ..."
$deadline = (Get-Date).AddMinutes(4)
while ((Get-Date) -lt $deadline) {
    if (Test-XttsReady) {
        Write-Host "XTTS ready."
        break
    }
    Start-Sleep -Seconds 2
}

if (-not (Test-XttsReady)) {
    Write-Host "XTTS did not become ready. Check logs:"
    Write-Host "  $xttsOut"
    Write-Host "  $xttsErr"
    exit 1
}

Write-Host "Starting LiveTalking low-latency pipeline ..."
Set-Location $liveTalkingDir

& $nerfPython app.py `
    --model wav2lip `
    --avatar_id wav2lip256_avatar1 `
    --tts xtts `
    --TTS_SERVER $xttsUrl `
    --xtts_language en `
    --xtts_stream_chunk_size 2 `
    --xtts_prewarm `
    --xtts_prewarm_rounds 2 `
    --asr_whisper_model medium `
    --asr_language en `
    --asr_device cuda `
    --asr_compute_type int8_float16 `
    --llm_backend local_qwen `
    --llm_model_id Qwen/Qwen2.5-3B-Instruct `
    --llm_quant_bits 4 `
    --llm_chunk_min_chars 2 `
    --llm_chunk_max_chars 16 `
    --llm_chunk_max_wait_ms 40 `
    --pipeline_prewarm `
    --llm_generation_prewarm
