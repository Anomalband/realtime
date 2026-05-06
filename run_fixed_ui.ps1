$ErrorActionPreference = "Stop"

$root = "C:\yazilim"
$repo = Join-Path $root "livetalking"
$xttsServerDir = Join-Path $root "xtts-streaming-server\server"

$nerfPython = "C:\Users\adbot\AppData\Local\miniforge3\envs\nerfstream\python.exe"
$xttsPython = "C:\Users\adbot\AppData\Local\miniforge3\envs\xttsserver\python.exe"

$uiPort = 8040
$xttsPort = 9880
$xttsUrl = "http://127.0.0.1:$xttsPort"

$uiOut = Join-Path $root "livetalking_8040.log"
$uiErr = Join-Path $root "livetalking_8040.err.log"
$xttsOut = Join-Path $root "xttsserver.log"
$xttsErr = Join-Path $root "xttsserver.err.log"

function Test-UrlReady([string]$url) {
    try {
        $r = Invoke-WebRequest -Uri $url -TimeoutSec 3 -UseBasicParsing
        return $r.StatusCode -ge 200
    } catch {
        return $false
    }
}

Write-Host "Checking XTTS server..."
if (-not (Test-UrlReady "$xttsUrl/languages")) {
    Write-Host "Starting XTTS server on $xttsUrl ..."
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

$xttsDeadline = (Get-Date).AddMinutes(4)
while ((Get-Date) -lt $xttsDeadline) {
    if (Test-UrlReady "$xttsUrl/languages") { break }
    Start-Sleep -Seconds 2
}
if (-not (Test-UrlReady "$xttsUrl/languages")) {
    Write-Host "XTTS not ready. Check:"
    Write-Host "  $xttsOut"
    Write-Host "  $xttsErr"
    exit 1
}

Write-Host "Restarting fixed UI server on port $uiPort ..."
$old = Get-NetTCPConnection -State Listen -LocalPort $uiPort -ErrorAction SilentlyContinue
if ($old) {
    Stop-Process -Id $old.OwningProcess -Force
    Start-Sleep -Seconds 1
}

if (Test-Path $uiOut) { Remove-Item $uiOut -Force }
if (Test-Path $uiErr) { Remove-Item $uiErr -Force }

Start-Process -WindowStyle Hidden -FilePath $nerfPython `
    -ArgumentList @(
        "app.py",
        "--transport", "webrtc",
        "--listenport", "$uiPort",
        "--model", "wav2lip",
        "--avatar_id", "wav2lip256_avatar1",
        "--tts", "xtts",
        "--TTS_SERVER", "$xttsUrl",
        "--xtts_language", "en",
        "--xtts_stream_chunk_size", "2",
        "--xtts_prewarm",
        "--xtts_prewarm_rounds", "2",
        "--asr_whisper_model", "medium",
        "--asr_language", "en",
        "--asr_device", "cuda",
        "--asr_compute_type", "int8_float16",
        "--llm_backend", "local_qwen",
        "--llm_model_id", "Qwen/Qwen2.5-0.5B-Instruct",
        "--llm_quant_bits", "4",
        "--llm_chunk_min_chars", "2",
        "--llm_chunk_max_chars", "16",
        "--llm_chunk_max_wait_ms", "40",
        "--pipeline_prewarm",
        "--no_llm_generation_prewarm"
    ) `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $uiOut `
    -RedirectStandardError $uiErr | Out-Null

$uiDeadline = (Get-Date).AddMinutes(4)
while ((Get-Date) -lt $uiDeadline) {
    if (Test-UrlReady "http://127.0.0.1:$uiPort/") { break }
    Start-Sleep -Seconds 2
}

if (-not (Test-UrlReady "http://127.0.0.1:$uiPort/")) {
    Write-Host "UI server not ready. Check:"
    Write-Host "  $uiOut"
    Write-Host "  $uiErr"
    exit 1
}

Write-Host "Ready: http://127.0.0.1:$uiPort/"
