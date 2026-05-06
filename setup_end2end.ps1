param(
    [string]$LiveEnv = "nerfstream",
    [string]$XttsEnv = "xttsserver",
    [switch]$SkipModelDownload
)

$ErrorActionPreference = "Stop"

$repoRoot = $PSScriptRoot
$runtimeRoot = Join-Path $repoRoot "_runtime"
$xttsRoot = Join-Path $runtimeRoot "xtts-streaming-server"
$xttsServerDir = Join-Path $xttsRoot "server"
$downloadsDir = Join-Path $repoRoot "_downloads"

$driveFolderUrl = "https://drive.google.com/drive/folders/1FOC_MD6wdogyyX_7V1d4NDIO7P9NlSAJ?usp=sharing"
$s3fdUrl = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"

function Require-Conda {
    $cmd = Get-Command conda -ErrorAction SilentlyContinue
    if (-not $cmd) {
        throw "conda not found. Please install Miniforge/Anaconda first, then rerun setup_end2end.ps1."
    }
    return $cmd.Source
}

function Get-CondaEnvs {
    $json = (conda env list --json | Out-String)
    return (ConvertFrom-Json $json).envs
}

function Ensure-CondaEnv([string]$envName) {
    $envs = Get-CondaEnvs
    $exists = $false
    foreach ($p in $envs) {
        if ($p -match "[\\/]" + [regex]::Escape($envName) + "$") {
            $exists = $true
            break
        }
    }
    if (-not $exists) {
        Write-Host "Creating conda env: $envName (python 3.10)"
        conda create -y -n $envName python=3.10 | Out-Host
    } else {
        Write-Host "Conda env exists: $envName"
    }
}

function CondaRun([string]$envName, [string[]]$cmdArgs) {
    conda run -n $envName @cmdArgs | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed in env '$envName': $($cmdArgs -join ' ')"
    }
}

function CondaRunWithRetry([string]$envName, [string[]]$cmdArgs, [int]$MaxAttempts = 3, [int]$SleepSeconds = 5) {
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        try {
            CondaRun $envName $cmdArgs
            return
        } catch {
            if ($attempt -ge $MaxAttempts) {
                throw
            }
            Write-Host "Retry $attempt/$MaxAttempts failed for: $($cmdArgs -join ' ')"
            Write-Host "Waiting $SleepSeconds seconds before retry..."
            Start-Sleep -Seconds $SleepSeconds
        }
    }
}

function Ensure-XttsPatch([string]$mainPath) {
    $content = Get-Content -Raw $mainPath

    if ($content -notmatch "import numbers") {
        $content = $content -replace "import wave(\r?\n)", "import wave`nimport numbers`n"
    }

    if ($content -notmatch "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD") {
        $patch = @"
# Torch >=2.6 defaults torch.load(..., weights_only=True), but XTTS checkpoints
# include trusted config classes and need full unpickling.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# Torch strict keyword handling can reject older transformers' `test_elements`
# keyword usage; provide a tiny compatibility shim.
_orig_torch_isin = torch.isin
def _compat_torch_isin(*args, **kwargs):
    elements = kwargs.get("elements", args[0] if len(args) > 0 else None)
    test_elements = kwargs.get("test_elements", args[1] if len(args) > 1 else None)
    test_element = kwargs.get("test_element", None)
    if test_element is None:
        test_element = test_elements

    if isinstance(elements, numbers.Number) and isinstance(test_element, numbers.Number):
        result = torch.tensor(elements == test_element)
        if kwargs.get("invert", False):
            result = ~result
        if "out" in kwargs and kwargs["out"] is not None:
            kwargs["out"].copy_(result)
            return kwargs["out"]
        return result

    if "test_element" in kwargs and "test_elements" not in kwargs and not isinstance(test_element, numbers.Number):
        kwargs["test_elements"] = kwargs.pop("test_element")
    elif "test_elements" in kwargs and "test_element" not in kwargs and isinstance(test_elements, numbers.Number):
        kwargs["test_element"] = kwargs.pop("test_elements")
    return _orig_torch_isin(*args, **kwargs)
torch.isin = _compat_torch_isin

"@
        $content = $content -replace "torch\.set_num_threads\(", ($patch + "torch.set_num_threads(")
    }

    if ($content -notmatch "XTTS_USE_DEEPSPEED") {
        $content = $content -replace "model = Xtts\.init_from_config\(config\)\r?\nmodel\.load_checkpoint\(config, checkpoint_dir=model_path, eval=True, use_deepspeed=True\)", "model = Xtts.init_from_config(config)`nuse_deepspeed = bool(device == `"cuda`" and os.environ.get(`"XTTS_USE_DEEPSPEED`", `"0`") == `"1`")`nmodel.load_checkpoint(config, checkpoint_dir=model_path, eval=True, use_deepspeed=use_deepspeed)"
    }

    Set-Content -Path $mainPath -Value $content -Encoding UTF8
}

function Ensure-Wav2LipAssets {
    New-Item -ItemType Directory -Path $downloadsDir -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $repoRoot "models") -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $repoRoot "data\avatars") -Force | Out-Null

    $rawModel = Join-Path $downloadsDir "wav2lip256.pth"
    $avatarTar = Join-Path $downloadsDir "wav2lip256_avatar1.tar.gz"
    if ((-not (Test-Path $rawModel)) -or (-not (Test-Path $avatarTar))) {
        Write-Host "Downloading wav2lip quickstart assets from Google Drive..."
        CondaRun $LiveEnv @("python", "-m", "gdown", $driveFolderUrl, "-O", $downloadsDir, "--folder", "--fuzzy")
    }
    if (-not (Test-Path $rawModel)) {
        throw "Missing file after download: $rawModel"
    }
    if (-not (Test-Path $avatarTar)) {
        throw "Missing file after download: $avatarTar"
    }

    Copy-Item -Path $rawModel -Destination (Join-Path $repoRoot "models\wav2lip.pth") -Force

    $avatarDir = Join-Path $repoRoot "data\avatars\wav2lip256_avatar1"
    if (-not (Test-Path $avatarDir)) {
        Write-Host "Extracting avatar package..."
        tar -xzf $avatarTar -C (Join-Path $repoRoot "data\avatars")
    }
    if (-not (Test-Path $avatarDir)) {
        throw "Avatar directory not found after extraction: $avatarDir"
    }

    $s3fdPath = Join-Path $downloadsDir "s3fd-619a316812.pth"
    if (-not (Test-Path $s3fdPath)) {
        Write-Host "Downloading s3fd detector model..."
        Invoke-WebRequest -Uri $s3fdUrl -OutFile $s3fdPath -UseBasicParsing
    }

    $dst1 = Join-Path $repoRoot "avatars\wav2lip\face_detection\detection\sfd\s3fd.pth"
    $dst2 = Join-Path $repoRoot "avatars\musetalk\utils\face_detection\detection\sfd\s3fd.pth"
    Copy-Item -Path $s3fdPath -Destination $dst1 -Force
    Copy-Item -Path $s3fdPath -Destination $dst2 -Force
}

Write-Host "Checking prerequisites..."
$null = Require-Conda
New-Item -ItemType Directory -Path $runtimeRoot -Force | Out-Null

Ensure-CondaEnv $LiveEnv
Ensure-CondaEnv $XttsEnv

Write-Host "Installing LiveTalking dependencies in env: $LiveEnv"
CondaRun $LiveEnv @("python", "-m", "pip", "install", "--upgrade", "pip")
CondaRun $LiveEnv @("python", "-m", "pip", "install", "setuptools<82", "wheel")
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    CondaRun $LiveEnv @("python", "-m", "pip", "install", "torch==2.5.0", "torchvision==0.20.0", "torchaudio==2.5.0", "--index-url", "https://download.pytorch.org/whl/cu121")
} else {
    CondaRun $LiveEnv @("python", "-m", "pip", "install", "torch==2.5.0", "torchvision==0.20.0", "torchaudio==2.5.0", "--index-url", "https://download.pytorch.org/whl/cpu")
}
CondaRunWithRetry $LiveEnv @("python", "-m", "pip", "install", "-r", (Join-Path $repoRoot "requirements.txt"))
CondaRun $LiveEnv @("python", "-m", "pip", "install", "gdown")

if (-not $SkipModelDownload) {
    Ensure-Wav2LipAssets
} else {
    Write-Host "Skipping model download by request."
}

if (-not (Test-Path $xttsRoot)) {
    Write-Host "Cloning xtts-streaming-server..."
    git clone https://github.com/coqui-ai/xtts-streaming-server.git $xttsRoot | Out-Host
}

$xttsMain = Join-Path $xttsServerDir "main.py"
if (-not (Test-Path $xttsMain)) {
    throw "XTTS server main.py not found at: $xttsMain"
}
Write-Host "Patching xtts-streaming-server for Windows compatibility..."
Ensure-XttsPatch $xttsMain

Write-Host "Installing XTTS server dependencies in env: $XttsEnv"
CondaRun $XttsEnv @("python", "-m", "pip", "install", "--upgrade", "pip")
CondaRun $XttsEnv @("python", "-m", "pip", "install", "setuptools<82", "wheel")
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    CondaRun $XttsEnv @("python", "-m", "pip", "install", "torch==2.5.0", "torchaudio==2.5.0", "--index-url", "https://download.pytorch.org/whl/cu121")
} else {
    CondaRun $XttsEnv @("python", "-m", "pip", "install", "torch==2.5.0", "torchaudio==2.5.0", "--index-url", "https://download.pytorch.org/whl/cpu")
}
CondaRun $XttsEnv @("python", "-m", "pip", "install", "TTS@git+https://github.com/coqui-ai/TTS@fa28f99f1508b5b5366539b2149963edcb80ba62")
CondaRun $XttsEnv @("python", "-m", "pip", "install", "uvicorn[standard]==0.23.2", "fastapi==0.95.2", "pydantic==1.10.13", "python-multipart==0.0.6", "typing-extensions>=4.8.0", "numpy==1.24.3", "cutlet", "mecab-python3==1.0.6", "unidic-lite==1.0.8", "unidic==1.1.0", "transformers==4.41.2", "tokenizers==0.19.1")
CondaRun $XttsEnv @("python", "-m", "unidic", "download")

Write-Host ""
Write-Host "Setup complete."
Write-Host "Next step:"
Write-Host "  powershell -ExecutionPolicy Bypass -File `"$repoRoot\run_fixed_ui.ps1`""
