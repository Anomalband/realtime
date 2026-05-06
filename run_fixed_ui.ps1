param(
    [string]$LiveEnv = "nerfstream",
    [string]$XttsEnv = "xttsserver",
    [int]$UiPort = 8040,
    [int]$XttsPort = 9880,
    [switch]$SkipSetup
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$runtimeRoot = Join-Path $repoRoot "_runtime"
$xttsServerDir = Join-Path $runtimeRoot "xtts-streaming-server\server"
$xttsUrl = "http://127.0.0.1:$XttsPort"

$uiOut = Join-Path $repoRoot ("livetalking_{0}.log" -f $UiPort)
$uiErr = Join-Path $repoRoot ("livetalking_{0}.err.log" -f $UiPort)
$xttsOut = Join-Path $repoRoot "xttsserver.log"
$xttsErr = Join-Path $repoRoot "xttsserver.err.log"

function Test-UrlReady([string]$url) {
    try {
        $r = Invoke-WebRequest -Uri $url -TimeoutSec 3 -UseBasicParsing
        return $r.StatusCode -ge 200
    } catch {
        return $false
    }
}

function Require-Conda {
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "conda not found. Please install Miniforge/Anaconda first."
    }
    return $cmd.Source
}

$null = Require-Conda

if (-not $SkipSetup) {
    $setupScript = Join-Path $repoRoot "setup_end2end.ps1"
    if (-not (Test-Path $setupScript)) {
        throw "setup_end2end.ps1 not found: $setupScript"
    }
    Write-Host "Running setup script before start..."
    powershell -ExecutionPolicy Bypass -File $setupScript -LiveEnv $LiveEnv -XttsEnv $XttsEnv
}

if (-not (Test-Path $xttsServerDir)) {
    $legacyXttsServerDir = Join-Path (Split-Path $repoRoot -Parent) "xtts-streaming-server\server"
    if (Test-Path $legacyXttsServerDir) {
        $xttsServerDir = $legacyXttsServerDir
    }
}

if (-not (Test-Path $xttsServerDir)) {
    throw "XTTS server folder missing: $xttsServerDir. Run setup_end2end.ps1 first."
}

Write-Host "Checking XTTS server..."
if (-not (Test-UrlReady "$xttsUrl/languages")) {
    Write-Host "Starting XTTS server on $xttsUrl ..."
    if (Test-Path $xttsOut) { Remove-Item $xttsOut -Force }
    if (Test-Path $xttsErr) { Remove-Item $xttsErr -Force }

    $env:COQUI_TOS_AGREED = "1"
    $env:XTTS_USE_DEEPSPEED = "0"
    Start-Process -WindowStyle Hidden -FilePath "conda" `
        -ArgumentList @("run", "-n", $XttsEnv, "python", "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "$XttsPort") `
        -WorkingDirectory $xttsServerDir `
        -RedirectStandardOutput $xttsOut `
        -RedirectStandardError $xttsErr | Out-Null
}

$xttsDeadline = (Get-Date).AddMinutes(5)
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

Write-Host "Restarting fixed UI server on port $UiPort ..."
$old = Get-NetTCPConnection -State Listen -LocalPort $UiPort -ErrorAction SilentlyContinue
if ($old) {
    Stop-Process -Id $old.OwningProcess -Force
    Start-Sleep -Seconds 1
}

if (Test-Path $uiOut) { Remove-Item $uiOut -Force }
if (Test-Path $uiErr) { Remove-Item $uiErr -Force }

Start-Process -WindowStyle Hidden -FilePath "conda" `
    -ArgumentList @(
        "run", "-n", $LiveEnv, "python", "app.py",
        "--transport", "webrtc",
        "--listenport", "$UiPort",
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
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $uiOut `
    -RedirectStandardError $uiErr | Out-Null

$uiDeadline = (Get-Date).AddMinutes(5)
while ((Get-Date) -lt $uiDeadline) {
    if (Test-UrlReady "http://127.0.0.1:$UiPort/") { break }
    Start-Sleep -Seconds 2
}

if (-not (Test-UrlReady "http://127.0.0.1:$UiPort/")) {
    Write-Host "UI server not ready. Check:"
    Write-Host "  $uiOut"
    Write-Host "  $uiErr"
    exit 1
}

Write-Host "Ready: http://127.0.0.1:$UiPort/"
